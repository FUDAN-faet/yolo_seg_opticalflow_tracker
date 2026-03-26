#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

namespace
{
double toSec(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

template<typename T>
T clampValue(T v, T lo, T hi)
{
  return std::max(lo, std::min(v, hi));
}
}  // namespace

struct BufferedColorFrame
{
  double stamp_sec{};
  sensor_msgs::msg::Image::SharedPtr msg;
  cv::Mat frame_bgr;
};

struct BufferedDepthFrame
{
  double stamp_sec{};
  sensor_msgs::msg::Image::SharedPtr msg;
  cv::Mat depth;
  std::string encoding;
};

class CatchupMaskTracker
{
public:
  struct Params
  {
    int max_corners = 200;
    double quality_level = 0.01;
    double min_distance = 5.0;
    int block_size = 7;
    cv::Size lk_win_size = cv::Size(21, 21);
    int lk_max_level = 3;
    int min_good_points = 8;
    int re_detect_interval = 3;
    int morph_kernel = 5;
  };

  explicit CatchupMaskTracker(const Params & params)
  : params_(params)
  {
    kernel_ = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(params_.morph_kernel, params_.morph_kernel));
  }

  void reset()
  {
    prev_gray_.release();
    prev_points_.release();
    current_mask_.release();
    last_stamp_sec_ = 0.0;
    active_ = false;
    track_step_count_ = 0;
  }

  bool active() const { return active_; }
  double lastStampSec() const { return last_stamp_sec_; }
  const cv::Mat & currentMask() const { return current_mask_; }
  const cv::Mat & prevPoints() const { return prev_points_; }

  bool initFromMask(const cv::Mat & frame_bgr, const cv::Mat & mask, double stamp_sec)
  {
    if (frame_bgr.empty() || mask.empty()) {
      reset();
      return false;
    }

    cv::Mat gray;
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat cleaned = cleanMask(mask);
    cv::Mat pts = detectPoints(gray, cleaned);

    if (pts.empty() || pts.rows < params_.min_good_points) {
      reset();
      return false;
    }

    prev_gray_ = gray;
    prev_points_ = pts;
    current_mask_ = cleaned;
    last_stamp_sec_ = stamp_sec;
    active_ = true;
    track_step_count_ = 0;
    return true;
  }

  bool updateToFrame(const cv::Mat & frame_bgr, double stamp_sec, cv::Mat & tracked_mask)
  {
    tracked_mask.release();
    if (!active_ || prev_gray_.empty() || current_mask_.empty() || frame_bgr.empty()) {
      return false;
    }
    if (stamp_sec <= last_stamp_sec_ + 1e-9) {
      tracked_mask = current_mask_;
      return true;
    }

    cv::Mat gray;
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);

    cv::Mat good_old, good_new;
    if (prev_points_.empty() || prev_points_.rows < params_.min_good_points) {
      if (!redetectAndFlow(gray, good_old, good_new)) {
        reset();
        return false;
      }
    } else {
      cv::Mat next_points, status, err;
      cv::calcOpticalFlowPyrLK(
        prev_gray_, gray, prev_points_, next_points, status, err,
        params_.lk_win_size, params_.lk_max_level,
        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 0.03));

      if (next_points.empty() || status.empty()) {
        reset();
        return false;
      }

      collectGoodPoints(prev_points_, next_points, status, good_old, good_new);
      if (good_new.rows < params_.min_good_points) {
        if (!redetectAndFlow(gray, good_old, good_new)) {
          reset();
          return false;
        }
      }
    }

    cv::Mat inliers;
    cv::Mat affine = cv::estimateAffinePartial2D(
      good_old, good_new, inliers, cv::RANSAC, 3.0, 2000, 0.99, 10);
    if (affine.empty()) {
      reset();
      return false;
    }

    cv::Mat warped;
    cv::warpAffine(
      current_mask_, warped, affine, gray.size(), cv::INTER_NEAREST,
      cv::BORDER_CONSTANT, cv::Scalar(0));
    warped = cleanMask(warped);
    if (cv::countNonZero(warped) == 0) {
      reset();
      return false;
    }

    ++track_step_count_;
    if (track_step_count_ % params_.re_detect_interval == 0) {
      cv::Mat fresh = detectPoints(gray, warped);
      prev_points_ = (!fresh.empty() && fresh.rows >= params_.min_good_points) ? fresh : good_new;
    } else {
      prev_points_ = good_new;
    }

    prev_gray_ = gray;
    current_mask_ = warped;
    last_stamp_sec_ = stamp_sec;
    tracked_mask = current_mask_;
    return true;
  }

private:
  static void collectGoodPoints(
    const cv::Mat & old_pts, const cv::Mat & new_pts, const cv::Mat & status,
    cv::Mat & good_old, cv::Mat & good_new)
  {
    std::vector<cv::Point2f> olds;
    std::vector<cv::Point2f> news;
    for (int i = 0; i < status.rows; ++i) {
      const auto ok = status.at<unsigned char>(i, 0);
      if (ok) {
        olds.push_back(old_pts.at<cv::Point2f>(i, 0));
        news.push_back(new_pts.at<cv::Point2f>(i, 0));
      }
    }
    if (!olds.empty()) {
      good_old = cv::Mat(static_cast<int>(olds.size()), 1, CV_32FC2, olds.data()).clone();
      good_new = cv::Mat(static_cast<int>(news.size()), 1, CV_32FC2, news.data()).clone();
    } else {
      good_old.release();
      good_new.release();
    }
  }

  cv::Mat binarizeMask(const cv::Mat & mask) const
  {
    cv::Mat mono;
    if (mask.channels() == 3) {
      cv::cvtColor(mask, mono, cv::COLOR_BGR2GRAY);
    } else {
      mono = mask.clone();
    }
    cv::threshold(mono, mono, 0, 255, cv::THRESH_BINARY);
    mono.convertTo(mono, CV_8UC1);
    return mono;
  }

  cv::Mat cleanMask(const cv::Mat & mask) const
  {
    cv::Mat out = binarizeMask(mask);
    cv::morphologyEx(out, out, cv::MORPH_OPEN, kernel_);
    cv::morphologyEx(out, out, cv::MORPH_CLOSE, kernel_);
    return out;
  }

  cv::Mat detectPoints(const cv::Mat & gray, const cv::Mat & mask) const
  {
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(
      gray, corners, params_.max_corners, params_.quality_level,
      params_.min_distance, mask, params_.block_size);
    if (corners.empty()) {
      return {};
    }
    return cv::Mat(static_cast<int>(corners.size()), 1, CV_32FC2, corners.data()).clone();
  }

  bool redetectAndFlow(const cv::Mat & gray, cv::Mat & good_old, cv::Mat & good_new)
  {
    if (prev_gray_.empty() || current_mask_.empty()) {
      return false;
    }
    cv::Mat redetected = detectPoints(prev_gray_, current_mask_);
    if (redetected.empty() || redetected.rows < params_.min_good_points) {
      return false;
    }

    cv::Mat next_points, status, err;
    cv::calcOpticalFlowPyrLK(
      prev_gray_, gray, redetected, next_points, status, err,
      params_.lk_win_size, params_.lk_max_level,
      cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 0.03));

    if (next_points.empty() || status.empty()) {
      return false;
    }
    collectGoodPoints(redetected, next_points, status, good_old, good_new);
    return good_new.rows >= params_.min_good_points;
  }

  Params params_;
  cv::Mat kernel_;
  cv::Mat prev_gray_;
  cv::Mat prev_points_;
  cv::Mat current_mask_;
  double last_stamp_sec_{0.0};
  bool active_{false};
  int track_step_count_{0};
};

class DelayedMaskFusionNode : public rclcpp::Node
{
public:
  explicit DelayedMaskFusionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("delayed_mask_fusion_node", options),
    tracker_(loadTrackerParams())
  {
    declare_parameter<std::string>("color_topic", "/camera_dcw2/sensor_color");
    declare_parameter<std::string>("depth_topic", "/camera_dcw2/depth/image_raw");
    declare_parameter<std::string>("yolo_mask_topic", "/tracking/bottle_mask");
    declare_parameter<std::string>("tracked_mask_topic", "/tracking/bottle_mask_tracked");
    declare_parameter<std::string>("filtered_depth_topic", "/tracking/depth_filtered_tracked");
    declare_parameter<std::string>("visualization_topic", "/tracking/bottle_mask_tracked_vis");
    declare_parameter<int>("buffer_max_frames", 180);
    declare_parameter<int>("depth_buffer_max_frames", 180);
    declare_parameter<double>("lookup_tolerance_sec", 0.05);
    declare_parameter<double>("depth_lookup_tolerance_sec", 0.05);
    declare_parameter<bool>("publish_overlay", false);
    declare_parameter<int>("mask_threshold", 1);
    declare_parameter<std::string>("depth_fill_mode", "infinity");
    declare_parameter<int>("masked_value_16uc1", 65535);
    declare_parameter<int>("log_every_n_frames", 30);

    color_topic_ = get_parameter("color_topic").as_string();
    depth_topic_ = get_parameter("depth_topic").as_string();
    yolo_mask_topic_ = get_parameter("yolo_mask_topic").as_string();
    tracked_mask_topic_ = get_parameter("tracked_mask_topic").as_string();
    filtered_depth_topic_ = get_parameter("filtered_depth_topic").as_string();
    visualization_topic_ = get_parameter("visualization_topic").as_string();
    buffer_max_frames_ = get_parameter("buffer_max_frames").as_int();
    depth_buffer_max_frames_ = get_parameter("depth_buffer_max_frames").as_int();
    lookup_tolerance_sec_ = get_parameter("lookup_tolerance_sec").as_double();
    depth_lookup_tolerance_sec_ = get_parameter("depth_lookup_tolerance_sec").as_double();
    publish_overlay_ = get_parameter("publish_overlay").as_bool();
    mask_threshold_ = get_parameter("mask_threshold").as_int();
    depth_fill_mode_ = get_parameter("depth_fill_mode").as_string();
    masked_value_16uc1_ = get_parameter("masked_value_16uc1").as_int();
    log_every_n_frames_ = get_parameter("log_every_n_frames").as_int();

    rclcpp::QoS qos(rclcpp::KeepLast(5));
    qos.best_effort();

    pub_mask_ = create_publisher<sensor_msgs::msg::Image>(tracked_mask_topic_, qos);
    pub_depth_filtered_ = create_publisher<sensor_msgs::msg::Image>(filtered_depth_topic_, qos);
    pub_vis_ = create_publisher<sensor_msgs::msg::Image>(visualization_topic_, qos);

    sub_color_ = create_subscription<sensor_msgs::msg::Image>(
      color_topic_, qos,
      std::bind(&DelayedMaskFusionNode::onColor, this, std::placeholders::_1));
    sub_depth_ = create_subscription<sensor_msgs::msg::Image>(
      depth_topic_, qos,
      std::bind(&DelayedMaskFusionNode::onDepth, this, std::placeholders::_1));
    sub_yolo_mask_ = create_subscription<sensor_msgs::msg::Image>(
      yolo_mask_topic_, qos,
      std::bind(&DelayedMaskFusionNode::onYoloMask, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "Started. color=%s depth=%s yolo_mask=%s tracked_mask=%s",
      color_topic_.c_str(), depth_topic_.c_str(), yolo_mask_topic_.c_str(), tracked_mask_topic_.c_str());
  }

private:
  CatchupMaskTracker::Params loadTrackerParams()
  {
    declare_parameter<int>("feature_max_corners", 200);
    declare_parameter<double>("feature_quality_level", 0.01);
    declare_parameter<double>("feature_min_distance", 5.0);
    declare_parameter<int>("feature_block_size", 7);
    declare_parameter<int>("lk_win_size", 21);
    declare_parameter<int>("lk_max_level", 3);
    declare_parameter<int>("min_good_points", 8);
    declare_parameter<int>("re_detect_interval", 3);
    declare_parameter<int>("morph_kernel", 5);

    CatchupMaskTracker::Params p;
    p.max_corners = get_parameter("feature_max_corners").as_int();
    p.quality_level = get_parameter("feature_quality_level").as_double();
    p.min_distance = get_parameter("feature_min_distance").as_double();
    p.block_size = get_parameter("feature_block_size").as_int();
    const int lk_win = get_parameter("lk_win_size").as_int();
    p.lk_win_size = cv::Size(lk_win, lk_win);
    p.lk_max_level = get_parameter("lk_max_level").as_int();
    p.min_good_points = get_parameter("min_good_points").as_int();
    p.re_detect_interval = get_parameter("re_detect_interval").as_int();
    p.morph_kernel = get_parameter("morph_kernel").as_int();
    return p;
  }

  void onColor(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv::Mat frame_bgr;
    try {
      frame_bgr = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Failed to convert color image: %s", e.what());
      return;
    }

    const double stamp_sec = toSec(msg->header.stamp);
    std::optional<PublishPack> pack;

    {
      std::scoped_lock<std::mutex> lock(mutex_);
      color_buffer_.push_back(BufferedColorFrame{stamp_sec, msg, frame_bgr.clone()});
      while (static_cast<int>(color_buffer_.size()) > buffer_max_frames_) {
        color_buffer_.pop_front();
      }

      if (!tracker_.active() || stamp_sec <= tracker_.lastStampSec() + 1e-9) {
        return;
      }

      cv::Mat tracked;
      if (!tracker_.updateToFrame(frame_bgr, stamp_sec, tracked)) {
        RCLCPP_WARN(get_logger(), "Tracker lost on color callback. Waiting for next YOLO correction.");
        return;
      }

      auto depth_item = findClosestDepthLocked(stamp_sec, depth_lookup_tolerance_sec_);
      if (!depth_item.has_value()) {
        return;
      }
      pack = buildPublishPackLocked(tracked, frame_bgr, *depth_item, msg->header);
    }

    if (pack.has_value()) {
      publish(pack.value(), "online_track");
    }
  }

  void onDepth(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv::Mat depth;
    try {
      depth = cv_bridge::toCvCopy(msg, msg->encoding)->image;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Failed to convert depth image: %s", e.what());
      return;
    }

    const double stamp_sec = toSec(msg->header.stamp);
    std::scoped_lock<std::mutex> lock(mutex_);
    depth_buffer_.push_back(BufferedDepthFrame{stamp_sec, msg, depth.clone(), msg->encoding});
    while (static_cast<int>(depth_buffer_.size()) > depth_buffer_max_frames_) {
      depth_buffer_.pop_front();
    }
  }

  void onYoloMask(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv::Mat mask;
    try {
      mask = cv_bridge::toCvCopy(msg, "mono8")->image;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Failed to convert YOLO mask: %s", e.what());
      return;
    }

    if (cv::countNonZero(mask) == 0) {
      return;
    }

    const double yolo_stamp_sec = toSec(msg->header.stamp);
    std::optional<PublishPack> pack;

    {
      std::scoped_lock<std::mutex> lock(mutex_);
      const auto color_idx = findClosestColorIndexLocked(yolo_stamp_sec, lookup_tolerance_sec_);
      if (!color_idx.has_value()) {
        RCLCPP_WARN(
          get_logger(),
          "No historical color frame matched this YOLO mask stamp. Increase buffer_max_frames or lookup_tolerance_sec.");
        return;
      }

      const auto & base = color_buffer_.at(*color_idx);
      if (!tracker_.initFromMask(base.frame_bgr, mask, base.stamp_sec)) {
        RCLCPP_WARN(get_logger(), "Tracker init failed from YOLO mask.");
        return;
      }

      cv::Mat latest_mask = tracker_.currentMask().clone();
      cv::Mat latest_color = base.frame_bgr;
      auto latest_header = base.msg->header;

      int replay_count = 0;
      for (size_t i = *color_idx + 1; i < color_buffer_.size(); ++i) {
        const auto & item = color_buffer_[i];
        if (!tracker_.updateToFrame(item.frame_bgr, item.stamp_sec, latest_mask)) {
          RCLCPP_WARN(get_logger(), "Tracker failed during catch-up replay.");
          return;
        }
        latest_color = item.frame_bgr;
        latest_header = item.msg->header;
        ++replay_count;
      }

      auto depth_item = findClosestDepthLocked(toSec(latest_header.stamp), depth_lookup_tolerance_sec_);
      if (!depth_item.has_value()) {
        return;
      }

      RCLCPP_INFO(
        get_logger(),
        "YOLO catch-up: yolo_stamp=%.6f replayed_frames=%d latest_stamp=%.6f",
        yolo_stamp_sec, replay_count, toSec(latest_header.stamp));

      pack = buildPublishPackLocked(latest_mask, latest_color, *depth_item, latest_header);
    }

    if (pack.has_value()) {
      publish(pack.value(), "yolo_catchup");
    }
  }

  struct PublishPack
  {
    sensor_msgs::msg::Image::SharedPtr mask_msg;
    sensor_msgs::msg::Image::SharedPtr depth_msg;
    sensor_msgs::msg::Image::SharedPtr vis_msg;
  };

  std::optional<size_t> findClosestColorIndexLocked(double stamp_sec, double tolerance_sec) const
  {
    if (color_buffer_.empty()) {
      return std::nullopt;
    }
    size_t best_idx = 0;
    double best_dt = std::numeric_limits<double>::max();
    for (size_t i = 0; i < color_buffer_.size(); ++i) {
      const double dt = std::fabs(color_buffer_[i].stamp_sec - stamp_sec);
      if (dt < best_dt) {
        best_dt = dt;
        best_idx = i;
      }
    }
    if (best_dt > tolerance_sec) {
      return std::nullopt;
    }
    return best_idx;
  }

  std::optional<BufferedDepthFrame> findClosestDepthLocked(double stamp_sec, double tolerance_sec) const
  {
    if (depth_buffer_.empty()) {
      return std::nullopt;
    }
    size_t best_idx = 0;
    double best_dt = std::numeric_limits<double>::max();
    for (size_t i = 0; i < depth_buffer_.size(); ++i) {
      const double dt = std::fabs(depth_buffer_[i].stamp_sec - stamp_sec);
      if (dt < best_dt) {
        best_dt = dt;
        best_idx = i;
      }
    }
    if (best_dt > tolerance_sec) {
      return std::nullopt;
    }
    return depth_buffer_[best_idx];
  }

  cv::Mat filterDepth(const cv::Mat & depth, const std::string & encoding, const cv::Mat & mask) const
  {
    if (depth.empty() || mask.empty()) {
      return {};
    }
    cv::Mat out = depth.clone();
    cv::Mat delete_region;
    cv::threshold(mask, delete_region, mask_threshold_ - 1, 255, cv::THRESH_BINARY);

    if (encoding == "16UC1") {
      const auto fill_value = static_cast<uint16_t>(
        clampValue(masked_value_16uc1_, 0, static_cast<int>(std::numeric_limits<uint16_t>::max())));
      out.setTo(fill_value, delete_region);
      return out;
    }

    if (encoding == "32FC1") {
      float fill_value = std::numeric_limits<float>::infinity();
      if (depth_fill_mode_ == "zero") {
        fill_value = 0.0f;
      } else if (depth_fill_mode_ == "nan") {
        fill_value = std::numeric_limits<float>::quiet_NaN();
      }
      out.setTo(fill_value, delete_region);
      return out;
    }

    RCLCPP_WARN(get_logger(), "Unsupported depth encoding: %s", encoding.c_str());
    return {};
  }

  cv::Mat buildVisualization(const cv::Mat & color_bgr, const cv::Mat & mask) const
  {
    cv::Mat vis = color_bgr.clone();
    if (!mask.empty() && cv::countNonZero(mask) > 0) {
      cv::Mat overlay = cv::Mat::zeros(color_bgr.size(), color_bgr.type());
      std::vector<cv::Mat> channels;
      cv::split(overlay, channels);
      channels[2] = mask;
      cv::merge(channels, overlay);
      cv::addWeighted(color_bgr, 1.0, overlay, 0.45, 0.0, vis);

      std::vector<cv::Point> nz;
      cv::findNonZero(mask, nz);
      if (!nz.empty()) {
        const auto rect = cv::boundingRect(nz);
        cv::rectangle(vis, rect, cv::Scalar(0, 255, 0), 2);
      }
    }

    const cv::Mat & pts = tracker_.prevPoints();
    for (int i = 0; i < pts.rows; ++i) {
      const auto p = pts.at<cv::Point2f>(i, 0);
      cv::circle(vis, p, 2, cv::Scalar(0, 255, 255), -1);
    }
    return vis;
  }

  std::optional<PublishPack> buildPublishPackLocked(
    const cv::Mat & tracked_mask,
    const cv::Mat & latest_color,
    const BufferedDepthFrame & depth_item,
    const std_msgs::msg::Header & color_header)
  {
    cv::Mat filtered_depth = filterDepth(depth_item.depth, depth_item.encoding, tracked_mask);
    if (filtered_depth.empty()) {
      return std::nullopt;
    }

    PublishPack pack;
    pack.mask_msg = cv_bridge::CvImage(color_header, "mono8", tracked_mask).toImageMsg();
    pack.depth_msg = cv_bridge::CvImage(depth_item.msg->header, depth_item.encoding, filtered_depth).toImageMsg();

    if (publish_overlay_) {
      cv::Mat vis = buildVisualization(latest_color, tracked_mask);
      pack.vis_msg = cv_bridge::CvImage(color_header, "bgr8", vis).toImageMsg();
    }
    return pack;
  }

  void publish(const PublishPack & pack, const std::string & source)
  {
    pub_mask_->publish(*pack.mask_msg);
    pub_depth_filtered_->publish(*pack.depth_msg);
    if (publish_overlay_ && pack.vis_msg) {
      pub_vis_->publish(*pack.vis_msg);
    }

    ++processed_frames_;
    if (log_every_n_frames_ > 0 && processed_frames_ % log_every_n_frames_ == 0) {
      RCLCPP_INFO(get_logger(), "Published tracked output. source=%s", source.c_str());
    }
  }

  std::mutex mutex_;
  std::deque<BufferedColorFrame> color_buffer_;
  std::deque<BufferedDepthFrame> depth_buffer_;

  CatchupMaskTracker tracker_;

  std::string color_topic_;
  std::string depth_topic_;
  std::string yolo_mask_topic_;
  std::string tracked_mask_topic_;
  std::string filtered_depth_topic_;
  std::string visualization_topic_;
  int buffer_max_frames_{180};
  int depth_buffer_max_frames_{180};
  double lookup_tolerance_sec_{0.05};
  double depth_lookup_tolerance_sec_{0.05};
  bool publish_overlay_{false};
  int mask_threshold_{1};
  std::string depth_fill_mode_{"infinity"};
  int masked_value_16uc1_{65535};
  int log_every_n_frames_{30};
  int processed_frames_{0};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_color_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_yolo_mask_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_mask_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_filtered_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_vis_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);

  auto node = std::make_shared<DelayedMaskFusionNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
