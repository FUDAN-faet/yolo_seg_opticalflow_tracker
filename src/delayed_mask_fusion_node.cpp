// 标准库：算法、并发、容器、字符串和数值工具。
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// ROS / OpenCV 相关头文件：
// - cv_bridge 负责 ROS Image 和 cv::Mat 之间的转换
// - imgproc / tracking 提供图像处理和 LK 光流
// - MultiThreadedExecutor 让 color / depth / yolo 回调可以并发执行
// - sensor_msgs::msg::Image / std_msgs::msg::Header 是我们使用的消息类型
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

namespace
{
// 把 ROS 的 sec + nanosec 统一转成 double 秒，便于时间比较和最近邻查找。
double toSec(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

// 通用钳制函数：把数值约束在 [lo, hi] 范围内。
template<typename T>
T clampValue(T v, T lo, T hi)
{
  return std::max(lo, std::min(v, hi));
}

// 把编码字符串统一转成大写，避免 "16UC1" / "16uc1" 这种大小写差异导致判断失败。
std::string normalizeEncoding(std::string encoding)
{
  std::transform(
    encoding.begin(), encoding.end(), encoding.begin(),
    [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return encoding;
}
}  // namespace

// color 缓冲项：
// - stamp_sec 用来按时间戳查找对应历史帧
// - header 用来发布输出消息时继承原始时间戳 / frame_id
// - gray 是追踪真正需要的图像
// - frame_bgr 只在需要可视化 overlay 时才填充，避免平时多占内存
struct BufferedColorFrame
{
  double stamp_sec{};
  std_msgs::msg::Header header;
  cv::Mat gray;
  cv::Mat frame_bgr;
};

// depth 缓冲项：
// - 同样保存时间戳和 header，便于后续寻找最接近的深度帧并发布
// - depth 是已经转换成 cv::Mat 的深度图
// - encoding 记录原始深度编码，发布时要原样带回去
struct BufferedDepthFrame
{
  double stamp_sec{};
  std_msgs::msg::Header header;
  cv::Mat depth;
  std::string encoding;
};

// 这个类只做一件事：
// 从某一帧 YOLO 给出的 mask 出发，用稀疏 LK 光流 + 仿射模型，把 mask 一路追到后续帧。
class CatchupMaskTracker
{
public:
  // 这里集中管理跟踪器的可调参数。
  struct Params
  {
    int max_corners = 200;                 // 最多保留多少个角点
    double quality_level = 0.01;          // goodFeaturesToTrack 的质量阈值
    double min_distance = 5.0;            // 相邻角点的最小距离
    int block_size = 7;                   // 角点检测窗口大小
    cv::Size lk_win_size = cv::Size(21, 21);  // LK 金字塔光流窗口大小
    int lk_max_level = 3;                 // LK 金字塔层数
    int min_good_points = 8;              // 能继续估计仿射所需的最少有效点数
    int re_detect_interval = 3;           // 每隔多少帧重新在 mask 区域内找角点
    int morph_kernel = 5;                 // mask 开闭运算核大小
  };

  // 构造时把形态学核预先建好，避免每帧重复创建。
  explicit CatchupMaskTracker(const Params & params)
  : params_(params)
  {
    kernel_ = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(params_.morph_kernel, params_.morph_kernel));
  }

  // 清空内部状态，表示 tracker 现在不可用，需要下一次 YOLO mask 重新初始化。
  void reset()
  {
    prev_gray_.release();
    prev_points_.release();
    current_mask_.release();
    last_stamp_sec_ = 0.0;
    active_ = false;
    track_step_count_ = 0;
  }

  // 当前 tracker 是否处于可用状态。
  bool active() const { return active_; }

  // tracker 当前追到的最新时间戳。
  double lastStampSec() const { return last_stamp_sec_; }

  // 当前追踪得到的最新 mask。
  const cv::Mat & currentMask() const { return current_mask_; }

  // 当前保存的角点，用于调试可视化。
  const cv::Mat & prevPoints() const { return prev_points_; }

  // 用某一帧的灰度图 + mask 作为新起点，初始化 tracker。
  bool initFromMask(const cv::Mat & frame_gray, const cv::Mat & mask, double stamp_sec)
  {
    // 初始化前先保证输入是合法的单通道灰度图，且 mask 尺寸匹配。
    if (
      frame_gray.empty() || frame_gray.channels() != 1 || mask.empty() ||
      frame_gray.size() != mask.size())
    {
      reset();
      return false;
    }

    // 先把 mask 清理干净，去掉小噪点和孔洞，减少后续角点漂移。
    cv::Mat cleaned = cleanMask(mask);

    // 只在目标区域内找角点，让 LK 跟踪集中在目标本身。
    cv::Mat pts = detectPoints(frame_gray, cleaned);

    // 如果一开始可用点都不够，说明这帧不适合初始化。
    if (pts.empty() || pts.rows < params_.min_good_points) {
      reset();
      return false;
    }

    // 记录初始化状态，之后 updateToGray 会从这里继续往后追。
    prev_gray_ = frame_gray;
    prev_points_ = pts;
    current_mask_ = cleaned;
    last_stamp_sec_ = stamp_sec;
    active_ = true;
    track_step_count_ = 0;
    return true;
  }

  // 把当前 tracker 从上一帧推进到新的灰度图所在帧。
  bool updateToGray(const cv::Mat & gray, double stamp_sec, cv::Mat & tracked_mask)
  {
    // 先清空输出，避免调用方误用旧结果。
    tracked_mask.release();

    // 只接受合法单通道灰度图，且尺寸必须和 tracker 的历史状态一致。
    if (
      !active_ || prev_gray_.empty() || current_mask_.empty() || gray.empty() ||
      gray.channels() != 1 || gray.size() != prev_gray_.size())
    {
      return false;
    }

    // 如果目标时间戳不比当前新，直接返回当前 mask，不再重复追踪。
    if (stamp_sec <= last_stamp_sec_ + 1e-9) {
      tracked_mask = current_mask_;
      return true;
    }

    // good_old / good_new 用来保存“上一帧有效点”和“当前帧对应点”。
    cv::Mat good_old;
    cv::Mat good_new;

    // 如果现有点已经太少，就先在上一帧 mask 区域内重找角点，再做一次 LK。
    if (prev_points_.empty() || prev_points_.rows < params_.min_good_points) {
      if (!redetectAndFlow(gray, good_old, good_new)) {
        reset();
        return false;
      }
    } else {
      // 正常路径：直接用上一帧保存的角点做 LK 光流。
      cv::Mat next_points;
      cv::Mat status;
      cv::Mat err;
      cv::calcOpticalFlowPyrLK(
        prev_gray_, gray, prev_points_, next_points, status, err,
        params_.lk_win_size, params_.lk_max_level,
        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 0.03));

      // LK 算不出来时，说明追踪已经失去可靠性。
      if (next_points.empty() || status.empty()) {
        reset();
        return false;
      }

      // 只保留 status=1 的点对。
      collectGoodPoints(prev_points_, next_points, status, good_old, good_new);

      // 如果有效点数不足，就退回“重找角点再做一次 LK”的备用路径。
      if (good_new.rows < params_.min_good_points) {
        if (!redetectAndFlow(gray, good_old, good_new)) {
          reset();
          return false;
        }
      }
    }

    // 用 RANSAC 拟合 2D 部分仿射变换，过滤离群点。
    cv::Mat inliers;
    cv::Mat affine = cv::estimateAffinePartial2D(
      good_old, good_new, inliers, cv::RANSAC, 3.0, 2000, 0.99, 10);
    if (affine.empty()) {
      reset();
      return false;
    }

    // 把上一帧 mask 按估计到的仿射关系 warp 到当前帧。
    cv::Mat warped;
    cv::warpAffine(
      current_mask_, warped, affine, gray.size(), cv::INTER_NEAREST,
      cv::BORDER_CONSTANT, cv::Scalar(0));

    // warp 后再做一次清理，抹掉边缘锯齿和孤立小块。
    warped = cleanMask(warped);

    // 如果 warp 完已经全空，说明追踪基本失效。
    if (cv::countNonZero(warped) == 0) {
      reset();
      return false;
    }

    // 已成功推进一帧，累加计数。
    ++track_step_count_;

    // 定期在新 mask 内重新检测角点，防止长期只靠旧点导致漂移。
    if (track_step_count_ % params_.re_detect_interval == 0) {
      cv::Mat fresh = detectPoints(gray, warped);
      prev_points_ = (!fresh.empty() && fresh.rows >= params_.min_good_points) ? fresh : good_new;
    } else {
      // 非重检测帧就直接沿用本次 LK 的输出点。
      prev_points_ = good_new;
    }

    // 更新内部状态，为下一次 update 做准备。
    prev_gray_ = gray;
    current_mask_ = warped;
    last_stamp_sec_ = stamp_sec;

    // 输出当前最新 mask。
    tracked_mask = current_mask_;
    return true;
  }

private:
  // 从 LK 输出中筛出状态有效的点对。
  static void collectGoodPoints(
    const cv::Mat & old_pts, const cv::Mat & new_pts, const cv::Mat & status,
    cv::Mat & good_old, cv::Mat & good_new)
  {
    std::vector<cv::Point2f> olds;
    std::vector<cv::Point2f> news;

    // 先按最大可能数目预留空间，减少 vector 扩容。
    olds.reserve(status.rows);
    news.reserve(status.rows);

    // 逐点检查 status，只有跟踪成功的点才保留。
    for (int i = 0; i < status.rows; ++i) {
      const auto ok = status.at<unsigned char>(i, 0);
      if (ok) {
        olds.push_back(old_pts.at<cv::Point2f>(i, 0));
        news.push_back(new_pts.at<cv::Point2f>(i, 0));
      }
    }

    // OpenCV 后续接口更方便吃 cv::Mat，这里把 vector 包成 Nx1 的 CV_32FC2。
    if (!olds.empty()) {
      good_old = cv::Mat(static_cast<int>(olds.size()), 1, CV_32FC2, olds.data()).clone();
      good_new = cv::Mat(static_cast<int>(news.size()), 1, CV_32FC2, news.data()).clone();
    } else {
      good_old.release();
      good_new.release();
    }
  }

  // 把输入 mask 统一转成 0/255 的单通道 uint8 图。
  cv::Mat binarizeMask(const cv::Mat & mask) const
  {
    cv::Mat mono;

    // 三通道 mask 先转灰度；单通道则直接拷贝。
    if (mask.channels() == 3) {
      cv::cvtColor(mask, mono, cv::COLOR_BGR2GRAY);
    } else {
      mono = mask.clone();
    }

    // 非零像素全设成 255，确保后续形态学和 threshold 行为一致。
    cv::threshold(mono, mono, 0, 255, cv::THRESH_BINARY);
    mono.convertTo(mono, CV_8UC1);
    return mono;
  }

  // 对二值 mask 做开运算 + 闭运算，去噪并填小洞。
  cv::Mat cleanMask(const cv::Mat & mask) const
  {
    cv::Mat out = binarizeMask(mask);
    cv::morphologyEx(out, out, cv::MORPH_OPEN, kernel_);
    cv::morphologyEx(out, out, cv::MORPH_CLOSE, kernel_);
    return out;
  }

  // 只在 mask 区域内找角点，保证点都落在目标上。
  cv::Mat detectPoints(const cv::Mat & gray, const cv::Mat & mask) const
  {
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(
      gray, corners, params_.max_corners, params_.quality_level,
      params_.min_distance, mask, params_.block_size);

    // 没找到角点时直接返回空 Mat。
    if (corners.empty()) {
      return {};
    }

    // 把 vector 包装成 OpenCV 常用的 Nx1 CV_32FC2 格式。
    return cv::Mat(static_cast<int>(corners.size()), 1, CV_32FC2, corners.data()).clone();
  }

  // 备用路径：当旧点太少时，先在上一帧 mask 内重找角点，再把它们流到当前帧。
  bool redetectAndFlow(const cv::Mat & gray, cv::Mat & good_old, cv::Mat & good_new)
  {
    // 没有上一帧状态时，无法做重检测。
    if (prev_gray_.empty() || current_mask_.empty()) {
      return false;
    }

    // 在上一帧 mask 区域内重新找角点。
    cv::Mat redetected = detectPoints(prev_gray_, current_mask_);
    if (redetected.empty() || redetected.rows < params_.min_good_points) {
      return false;
    }

    // 把这些新角点从 prev_gray_ 流到当前 gray。
    cv::Mat next_points;
    cv::Mat status;
    cv::Mat err;
    cv::calcOpticalFlowPyrLK(
      prev_gray_, gray, redetected, next_points, status, err,
      params_.lk_win_size, params_.lk_max_level,
      cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 0.03));

    // 光流失败则直接失败。
    if (next_points.empty() || status.empty()) {
      return false;
    }

    // 提取出成功对应的点对。
    collectGoodPoints(redetected, next_points, status, good_old, good_new);
    return good_new.rows >= params_.min_good_points;
  }

  // 参数集合。
  Params params_;

  // 形态学核，构造时预先生成。
  cv::Mat kernel_;

  // 上一帧灰度图。
  cv::Mat prev_gray_;

  // 上一帧保留下来的角点。
  cv::Mat prev_points_;

  // 当前最新 mask。
  cv::Mat current_mask_;

  // tracker 当前追到的时间戳。
  double last_stamp_sec_{0.0};

  // 当前 tracker 是否有效。
  bool active_{false};

  // 已推进的帧数，用于控制定期重检测角点。
  int track_step_count_{0};
};

// 这个节点把三件事合在一起做：
// 1. 缓存 color / depth 历史帧
// 2. 接收延迟到达的 YOLO mask，做 catch-up replay
// 3. 在线持续追踪，并把 mask 用到 depth 上做过滤
class DelayedMaskFusionNode : public rclcpp::Node
{
public:
  explicit DelayedMaskFusionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("delayed_mask_fusion_node", options),
    // 先读取 tracker 参数，再用这些参数构造共享 tracker。
    tracker_params_(loadTrackerParams()),
    tracker_(tracker_params_)
  {
    // 先声明节点参数，给外部 launch / 命令行提供可配置入口。
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

    // 读取参数值，后续整个节点都用这些成员变量。
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

    // 传感器类数据只保留最新一帧：
    // 应用层自己已经维护了历史缓冲，DDS 队列没必要再堆旧帧。
    auto sensor_qos = rclcpp::SensorDataQoS();
    sensor_qos.keep_last(1);

    // 输出：
    // - tracked mask
    // - 过滤后的 depth
    // - 可选 overlay 可视化
    pub_mask_ = create_publisher<sensor_msgs::msg::Image>(tracked_mask_topic_, sensor_qos);
    pub_depth_filtered_ = create_publisher<sensor_msgs::msg::Image>(filtered_depth_topic_, sensor_qos);
    pub_vis_ = create_publisher<sensor_msgs::msg::Image>(visualization_topic_, sensor_qos);

    // 为三类输入各建一个 callback group，配合 MultiThreadedExecutor 并发执行。
    color_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    depth_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    yolo_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // 把每个订阅绑到它自己的 group 上。
    rclcpp::SubscriptionOptions color_options;
    color_options.callback_group = color_callback_group_;
    rclcpp::SubscriptionOptions depth_options;
    depth_options.callback_group = depth_callback_group_;
    rclcpp::SubscriptionOptions yolo_options;
    yolo_options.callback_group = yolo_callback_group_;

    // color 输入：既要缓存，也可能驱动在线追踪。
    sub_color_ = create_subscription<sensor_msgs::msg::Image>(
      color_topic_, sensor_qos,
      std::bind(&DelayedMaskFusionNode::onColor, this, std::placeholders::_1),
      color_options);

    // depth 输入：只负责缓存，之后按最近时间戳取出来做过滤。
    sub_depth_ = create_subscription<sensor_msgs::msg::Image>(
      depth_topic_, sensor_qos,
      std::bind(&DelayedMaskFusionNode::onDepth, this, std::placeholders::_1),
      depth_options);

    // YOLO mask 输入：触发一次 catch-up replay，并用结果纠正共享 tracker。
    sub_yolo_mask_ = create_subscription<sensor_msgs::msg::Image>(
      yolo_mask_topic_, sensor_qos,
      std::bind(&DelayedMaskFusionNode::onYoloMask, this, std::placeholders::_1),
      yolo_options);

    // 启动日志，方便确认 topic 是否配置正确。
    RCLCPP_INFO(
      get_logger(),
      "Started. color=%s depth=%s yolo_mask=%s tracked_mask=%s",
      color_topic_.c_str(), depth_topic_.c_str(), yolo_mask_topic_.c_str(), tracked_mask_topic_.c_str());
  }

private:
  // 读取并返回 tracker 参数。
  CatchupMaskTracker::Params loadTrackerParams()
  {
    // 这些参数都属于 tracker 本体，不属于 ROS topic / buffering 配置。
    declare_parameter<int>("feature_max_corners", 200);
    declare_parameter<double>("feature_quality_level", 0.01);
    declare_parameter<double>("feature_min_distance", 5.0);
    declare_parameter<int>("feature_block_size", 7);
    declare_parameter<int>("lk_win_size", 21);
    declare_parameter<int>("lk_max_level", 3);
    declare_parameter<int>("min_good_points", 8);
    declare_parameter<int>("re_detect_interval", 3);
    declare_parameter<int>("morph_kernel", 5);

    // 把参数逐项填进 Params 结构体。
    CatchupMaskTracker::Params p;
    p.max_corners = get_parameter("feature_max_corners").as_int();
    p.quality_level = get_parameter("feature_quality_level").as_double();
    p.min_distance = get_parameter("feature_min_distance").as_double();
    p.block_size = get_parameter("feature_block_size").as_int();

    // LK 窗口参数是一个方窗，这里把单个 int 组装成 cv::Size。
    const int lk_win = get_parameter("lk_win_size").as_int();
    p.lk_win_size = cv::Size(lk_win, lk_win);

    p.lk_max_level = get_parameter("lk_max_level").as_int();
    p.min_good_points = get_parameter("min_good_points").as_int();
    p.re_detect_interval = get_parameter("re_detect_interval").as_int();
    p.morph_kernel = get_parameter("morph_kernel").as_int();
    return p;
  }

  // color 回调：
  // 1. 把 color 缓进内部队列
  // 2. 如果共享 tracker 已经激活，就把它往当前帧推进一帧
  void onColor(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv::Mat frame_gray;
    cv::Mat frame_bgr;

    try {
      // 如果需要 overlay，就保留 BGR 并顺手转一次灰度。
      if (publish_overlay_) {
        auto color_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        frame_bgr = color_ptr->image;
        cv::cvtColor(frame_bgr, frame_gray, cv::COLOR_BGR2GRAY);
      } else {
        // 不需要 overlay 时，直接转成 mono8，避免额外 BGR 常驻内存。
        auto gray_ptr = cv_bridge::toCvCopy(msg, "mono8");
        frame_gray = gray_ptr->image;
      }
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Failed to convert color image: %s", e.what());
      return;
    }

    // 当前 color 帧的时间戳（秒）。
    const double stamp_sec = toSec(msg->header.stamp);

    {
      // 先把 color 压入缓冲区，供后续 YOLO catch-up 使用。
      std::scoped_lock<std::mutex> lock(buffer_mutex_);
      color_buffer_.push_back(BufferedColorFrame{stamp_sec, msg->header, frame_gray, frame_bgr});

      // 缓冲区超长时，从最老的一端丢弃。
      while (static_cast<int>(color_buffer_.size()) > buffer_max_frames_) {
        color_buffer_.pop_front();
      }
    }

    // 这里不直接拿共享 tracker_ 在锁内算，
    // 而是先复制一份本地副本，在锁外完成重计算。
    CatchupMaskTracker local_tracker(tracker_params_);
    std::uint64_t snapshot_version = 0;
    {
      std::scoped_lock<std::mutex> lock(tracker_mutex_);

      // 如果 tracker 还没激活，或者这帧时间戳不更新，就只缓存不追踪。
      if (!tracker_.active() || stamp_sec <= tracker_.lastStampSec() + 1e-9) {
        return;
      }

      // 复制共享 tracker 当前状态，后面在锁外更新这份副本。
      local_tracker = tracker_;

      // 记下版本号，后面提交时用来检测“有没有人先一步改过 tracker_”。
      snapshot_version = tracker_version_;
    }

    // 在锁外做真正的光流更新，避免长时间阻塞其它回调。
    cv::Mat tracked;
    const bool ok = local_tracker.updateToGray(frame_gray, stamp_sec, tracked);
    if (!ok) {
      RCLCPP_WARN(get_logger(), "Tracker lost on color callback. Waiting for next YOLO correction.");
    }

    bool committed = false;
    {
      std::scoped_lock<std::mutex> lock(tracker_mutex_);

      // 只有当共享 tracker 从我们快照出来之后没有被别人改过，
      // 才允许把本地结果写回去，防止旧结果覆盖新结果。
      if (tracker_version_ == snapshot_version) {
        tracker_ = local_tracker;
        ++tracker_version_;
        committed = true;
      }
    }

    // 如果提交失败、追踪失败、或者没有有效 mask，就到这里结束。
    if (!committed || !ok || tracked.empty()) {
      return;
    }

    // 再找一张时间上最接近当前 color 的 depth，用它来生成 filtered depth。
    auto depth_item = findClosestDepth(stamp_sec, depth_lookup_tolerance_sec_);
    if (!depth_item.has_value()) {
      return;
    }

    // 构建输出包：tracked mask / filtered depth / 可选 vis。
    auto pack = buildPublishPack(
      tracked, frame_bgr, *depth_item, msg->header, local_tracker.prevPoints());
    if (pack.has_value()) {
      publish(pack.value(), "online_track");
    }
  }

  // depth 回调只做一件事：缓存最新深度图。
  void onDepth(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv::Mat depth;
    try {
      // 深度这里保持原始 encoding，不做额外改动。
      auto depth_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
      depth = depth_ptr->image;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Failed to convert depth image: %s", e.what());
      return;
    }

    const double stamp_sec = toSec(msg->header.stamp);

    // 压入 depth 缓冲区，供 onColor / onYoloMask 查找最近深度帧。
    std::scoped_lock<std::mutex> lock(buffer_mutex_);
    depth_buffer_.push_back(BufferedDepthFrame{stamp_sec, msg->header, depth, msg->encoding});
    while (static_cast<int>(depth_buffer_.size()) > depth_buffer_max_frames_) {
      depth_buffer_.pop_front();
    }
  }

  // YOLO mask 回调：
  // 1. 从历史 color 缓冲里找到与 mask 时间戳最接近的那一帧
  // 2. 在该帧重新初始化 tracker
  // 3. 逐帧 replay 到最新
  // 4. 把 replay 结果提交为新的共享 tracker 状态
  void onYoloMask(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv::Mat mask;
    try {
      // YOLO mask 强制按 mono8 读取，方便后续二值处理。
      auto mask_ptr = cv_bridge::toCvCopy(msg, "mono8");
      mask = mask_ptr->image;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Failed to convert YOLO mask: %s", e.what());
      return;
    }

    // 空 mask 直接忽略，让当前 tracker 继续跑。
    if (mask.empty() || cv::countNonZero(mask) == 0) {
      return;
    }

    const double yolo_stamp_sec = toSec(msg->header.stamp);

    // 在锁内拿一份从“匹配帧”到“当前缓冲尾部”的 color 快照，
    // 后面的 replay 全部在锁外完成。
    auto color_frames = snapshotColorFramesForCatchup(yolo_stamp_sec, lookup_tolerance_sec_);
    if (!color_frames.has_value() || color_frames->empty()) {
      RCLCPP_WARN(
        get_logger(),
        "No historical color frame matched this YOLO mask stamp. Increase buffer_max_frames or lookup_tolerance_sec.");
      return;
    }

    // 如果 YOLO mask 分辨率和 color 缓冲分辨率不一致，直接拒绝，避免后面尺寸错位。
    if (mask.size() != color_frames->front().gray.size()) {
      RCLCPP_WARN(
        get_logger(),
        "YOLO mask size (%dx%d) does not match buffered color frame size (%dx%d).",
        mask.cols, mask.rows, color_frames->front().gray.cols, color_frames->front().gray.rows);
      return;
    }

    // YOLO correction 使用独立的本地 tracker，不直接动共享 tracker_。
    CatchupMaskTracker local_tracker(tracker_params_);
    if (!local_tracker.initFromMask(color_frames->front().gray, mask, color_frames->front().stamp_sec)) {
      RCLCPP_WARN(get_logger(), "Tracker init failed from YOLO mask.");
      return;
    }

    // latest_frame / latest_mask 会随着 replay 向后不断更新。
    BufferedColorFrame latest_frame = color_frames->front();
    cv::Mat latest_mask = local_tracker.currentMask();
    int replay_count = 0;

    // 第一段 replay：把快照里已有的历史帧一路追到当时的最新帧。
    for (size_t i = 1; i < color_frames->size(); ++i) {
      const auto & item = color_frames->at(i);
      if (!local_tracker.updateToGray(item.gray, item.stamp_sec, latest_mask)) {
        RCLCPP_WARN(get_logger(), "Tracker failed during catch-up replay.");
        return;
      }
      latest_frame = item;
      ++replay_count;
    }

    // 第二段 replay：为了减少 YOLO 回调执行期间又有新 color 到来的影响，
    // 再额外做最多 2 轮“补追最新帧”。
    for (int round = 0; round < 2; ++round) {
      auto newer_frames = snapshotColorFramesAfter(local_tracker.lastStampSec());
      if (newer_frames.empty()) {
        break;
      }

      for (const auto & item : newer_frames) {
        if (!local_tracker.updateToGray(item.gray, item.stamp_sec, latest_mask)) {
          RCLCPP_WARN(get_logger(), "Tracker failed while extending catch-up replay.");
          return;
        }
        latest_frame = item;
        ++replay_count;
      }
    }

    bool committed = false;
    {
      std::scoped_lock<std::mutex> lock(tracker_mutex_);

      // YOLO correction 一般比单纯在线 tracking 更可信，
      // 只要它不比当前共享 tracker 更旧，就允许写回。
      if (!tracker_.active() || local_tracker.lastStampSec() >= tracker_.lastStampSec() - 1e-9) {
        tracker_ = local_tracker;
        ++tracker_version_;
        committed = true;
      }
    }

    // 如果 replay 结果已经落后于当前共享 tracker，就不再发布。
    if (!committed) {
      return;
    }

    // 用 catch-up 后追到的最新时间戳，去找最接近的 depth。
    auto depth_item = findClosestDepth(latest_frame.stamp_sec, depth_lookup_tolerance_sec_);
    if (!depth_item.has_value()) {
      return;
    }

    // 打印一次 catch-up 统计，方便观察 YOLO 延迟和 replay 长度。
    RCLCPP_INFO(
      get_logger(),
      "YOLO catch-up: yolo_stamp=%.6f replayed_frames=%d latest_stamp=%.6f",
      yolo_stamp_sec, replay_count, latest_frame.stamp_sec);

    // 构建并发布 catch-up 的最终输出。
    auto pack = buildPublishPack(
      latest_mask, latest_frame.frame_bgr, *depth_item, latest_frame.header, local_tracker.prevPoints());
    if (pack.has_value()) {
      publish(pack.value(), "yolo_catchup");
    }
  }

  // 打包一次发布需要的三种消息。
  struct PublishPack
  {
    sensor_msgs::msg::Image::SharedPtr mask_msg;   // tracked mask
    sensor_msgs::msg::Image::SharedPtr depth_msg;  // filtered depth
    sensor_msgs::msg::Image::SharedPtr vis_msg;    // 可选可视化
  };

  // 在已加 buffer 锁的前提下，找和给定时间戳最接近的 color 索引。
  std::optional<size_t> findClosestColorIndexLocked(double stamp_sec, double tolerance_sec) const
  {
    if (color_buffer_.empty()) {
      return std::nullopt;
    }

    size_t best_idx = 0;
    double best_dt = std::numeric_limits<double>::max();

    // 线性扫描，找到时间差最小的那一帧。
    for (size_t i = 0; i < color_buffer_.size(); ++i) {
      const double dt = std::fabs(color_buffer_[i].stamp_sec - stamp_sec);
      if (dt < best_dt) {
        best_dt = dt;
        best_idx = i;
      }
    }

    // 时间差超过容忍阈值时，认为没匹配到。
    if (best_dt > tolerance_sec) {
      return std::nullopt;
    }
    return best_idx;
  }

  // 在已加 buffer 锁的前提下，找与给定时间戳最接近的 depth。
  std::optional<BufferedDepthFrame> findClosestDepthLocked(double stamp_sec, double tolerance_sec) const
  {
    if (depth_buffer_.empty()) {
      return std::nullopt;
    }

    size_t best_idx = 0;
    double best_dt = std::numeric_limits<double>::max();

    // 同样做线性最近邻搜索。
    for (size_t i = 0; i < depth_buffer_.size(); ++i) {
      const double dt = std::fabs(depth_buffer_[i].stamp_sec - stamp_sec);
      if (dt < best_dt) {
        best_dt = dt;
        best_idx = i;
      }
    }

    // 超过容忍阈值则放弃。
    if (best_dt > tolerance_sec) {
      return std::nullopt;
    }
    return depth_buffer_[best_idx];
  }

  // 对外包装版：自己加 buffer 锁，再调用 locked 版本。
  std::optional<BufferedDepthFrame> findClosestDepth(double stamp_sec, double tolerance_sec)
  {
    std::scoped_lock<std::mutex> lock(buffer_mutex_);
    return findClosestDepthLocked(stamp_sec, tolerance_sec);
  }

  // 取一份 color 快照：
  // 从与 yolo_stamp 最接近的那一帧开始，一直到当前缓冲尾部。
  std::optional<std::vector<BufferedColorFrame>> snapshotColorFramesForCatchup(
    double stamp_sec, double tolerance_sec)
  {
    std::scoped_lock<std::mutex> lock(buffer_mutex_);

    const auto color_idx = findClosestColorIndexLocked(stamp_sec, tolerance_sec);
    if (!color_idx.has_value()) {
      return std::nullopt;
    }

    std::vector<BufferedColorFrame> frames;
    frames.reserve(color_buffer_.size() - *color_idx);
    for (size_t i = *color_idx; i < color_buffer_.size(); ++i) {
      frames.push_back(color_buffer_[i]);
    }
    return frames;
  }

  // 取一份“比某个时间戳更晚”的 color 快照，用于 YOLO replay 末尾补追最新帧。
  std::vector<BufferedColorFrame> snapshotColorFramesAfter(double stamp_sec)
  {
    std::scoped_lock<std::mutex> lock(buffer_mutex_);

    std::vector<BufferedColorFrame> frames;
    for (const auto & item : color_buffer_) {
      if (item.stamp_sec > stamp_sec + 1e-9) {
        frames.push_back(item);
      }
    }
    return frames;
  }

  // 把 tracked mask 应用到 depth 上，生成过滤后的深度图。
  cv::Mat filterDepth(const cv::Mat & depth, const std::string & encoding, const cv::Mat & mask) const
  {
    // 空输入直接失败。
    if (depth.empty() || mask.empty()) {
      return {};
    }

    // 深度图必须是二维单通道矩阵。
    if (depth.dims != 2) {
      RCLCPP_WARN(get_logger(), "Unsupported depth image rank: %d", depth.dims);
      return {};
    }

    // mask 必须是单通道。
    if (mask.channels() != 1) {
      RCLCPP_WARN(get_logger(), "Tracked mask must be single-channel, but got %d channels.", mask.channels());
      return {};
    }

    // 只有在尺寸一致时，才能按像素位置把目标区域从 depth 中抹掉。
    if (depth.size() != mask.size()) {
      RCLCPP_WARN(
        get_logger(),
        "Depth/mask size mismatch. mask=%dx%d depth=%dx%d",
        mask.cols, mask.rows, depth.cols, depth.rows);
      return {};
    }

    // 输出深度图从输入深度复制一份开始，后面只改目标区域。
    cv::Mat out = depth.clone();

    // 根据 mask_threshold 生成要删除的区域。
    cv::Mat delete_region;
    cv::threshold(mask, delete_region, mask_threshold_ - 1, 255, cv::THRESH_BINARY);

    // 深度编码统一转大写后再判断，避免大小写差异。
    const std::string normalized_encoding = normalizeEncoding(encoding);

    // 16UC1 用 uint16_t 的最大值或用户指定值表示“远处/无效”。
    if (normalized_encoding == "16UC1") {
      const auto fill_value = static_cast<uint16_t>(
        clampValue(masked_value_16uc1_, 0, static_cast<int>(std::numeric_limits<uint16_t>::max())));
      out.setTo(fill_value, delete_region);
      return out;
    }

    // 32FC1 可以直接写 0 / NaN / inf。
    if (normalized_encoding == "32FC1") {
      float fill_value = std::numeric_limits<float>::infinity();
      if (depth_fill_mode_ == "zero") {
        fill_value = 0.0f;
      } else if (depth_fill_mode_ == "nan") {
        fill_value = std::numeric_limits<float>::quiet_NaN();
      }
      out.setTo(fill_value, delete_region);
      return out;
    }

    // 其它深度编码暂不支持。
    RCLCPP_WARN(get_logger(), "Unsupported depth encoding: %s", encoding.c_str());
    return {};
  }

  // 生成可视化图：
  // 在原始 color 上叠加红色 mask，再画出包围框和角点。
  cv::Mat buildVisualization(
    const cv::Mat & color_bgr, const cv::Mat & mask, const cv::Mat & tracker_points) const
  {
    // 没有 color 图时，就没法做 overlay。
    if (color_bgr.empty()) {
      return {};
    }

    // 先复制一张原图作为可视化底图。
    cv::Mat vis = color_bgr.clone();

    // 如果 mask 非空，就把 mask 用半透明红色叠加上去。
    if (!mask.empty() && cv::countNonZero(mask) > 0) {
      cv::Mat overlay = cv::Mat::zeros(color_bgr.size(), color_bgr.type());
      std::vector<cv::Mat> channels;
      cv::split(overlay, channels);
      channels[2] = mask;
      cv::merge(channels, overlay);
      cv::addWeighted(color_bgr, 1.0, overlay, 0.45, 0.0, vis);

      // 额外找出 mask 的外接矩形，方便观察目标大致位置。
      std::vector<cv::Point> nz;
      cv::findNonZero(mask, nz);
      if (!nz.empty()) {
        const auto rect = cv::boundingRect(nz);
        cv::rectangle(vis, rect, cv::Scalar(0, 255, 0), 2);
      }
    }

    // 把当前 tracker 保存的角点画出来，便于观察光流是否稳定。
    for (int i = 0; i < tracker_points.rows; ++i) {
      const auto p = tracker_points.at<cv::Point2f>(i, 0);
      cv::circle(vis, p, 2, cv::Scalar(0, 255, 255), -1);
    }
    return vis;
  }

  // 根据当前 tracked mask + 最近 depth + 可选 color，打包出待发布消息。
  std::optional<PublishPack> buildPublishPack(
    const cv::Mat & tracked_mask,
    const cv::Mat & latest_color,
    const BufferedDepthFrame & depth_item,
    const std_msgs::msg::Header & color_header,
    const cv::Mat & tracker_points) const
  {
    // 先生成 filtered depth；失败就直接放弃本次发布。
    cv::Mat filtered_depth = filterDepth(depth_item.depth, depth_item.encoding, tracked_mask);
    if (filtered_depth.empty()) {
      return std::nullopt;
    }

    PublishPack pack;

    // mask 消息沿用 color header，因为 mask 的坐标系和时间戳来自 color 帧。
    pack.mask_msg = cv_bridge::CvImage(color_header, "mono8", tracked_mask).toImageMsg();

    // depth 消息沿用 depth header，因为它本质上还是 depth 图。
    pack.depth_msg = cv_bridge::CvImage(depth_item.header, depth_item.encoding, filtered_depth).toImageMsg();

    // 只有在用户要求 overlay 且当前真的有 BGR 图时才构建可视化。
    if (publish_overlay_ && !latest_color.empty()) {
      cv::Mat vis = buildVisualization(latest_color, tracked_mask, tracker_points);
      if (!vis.empty()) {
        pack.vis_msg = cv_bridge::CvImage(color_header, "bgr8", vis).toImageMsg();
      }
    }
    return pack;
  }

  // 实际发布输出消息。
  void publish(const PublishPack & pack, const std::string & source)
  {
    // 始终发布 tracked mask 和 filtered depth。
    pub_mask_->publish(*pack.mask_msg);
    pub_depth_filtered_->publish(*pack.depth_msg);

    // overlay 是可选项，只有打开时才发。
    if (publish_overlay_ && pack.vis_msg) {
      pub_vis_->publish(*pack.vis_msg);
    }

    // 原子计数发布帧数，便于多线程下安全统计。
    const int processed = ++processed_frames_;
    if (log_every_n_frames_ > 0 && processed % log_every_n_frames_ == 0) {
      RCLCPP_INFO(get_logger(), "Published tracked output. source=%s", source.c_str());
    }
  }

  // tracker 的静态参数配置。
  CatchupMaskTracker::Params tracker_params_;

  // buffer_mutex_ 只保护 color / depth 两个历史缓冲区。
  std::mutex buffer_mutex_;
  std::deque<BufferedColorFrame> color_buffer_;
  std::deque<BufferedDepthFrame> depth_buffer_;

  // tracker_mutex_ 只保护共享 tracker 状态。
  std::mutex tracker_mutex_;
  CatchupMaskTracker tracker_;

  // 每次共享 tracker 被写回时都递增，用于检测“旧快照结果回写”。
  std::uint64_t tracker_version_{0};

  // topic 参数。
  std::string color_topic_;
  std::string depth_topic_;
  std::string yolo_mask_topic_;
  std::string tracked_mask_topic_;
  std::string filtered_depth_topic_;
  std::string visualization_topic_;

  // 缓冲和查找相关参数。
  int buffer_max_frames_{180};
  int depth_buffer_max_frames_{180};
  double lookup_tolerance_sec_{0.05};
  double depth_lookup_tolerance_sec_{0.05};

  // 输出行为和 depth 过滤参数。
  bool publish_overlay_{false};
  int mask_threshold_{1};
  std::string depth_fill_mode_{"infinity"};
  int masked_value_16uc1_{65535};

  // 日志统计频率。
  int log_every_n_frames_{30};

  // 多线程下安全统计发布帧数。
  std::atomic<int> processed_frames_{0};

  // 三个回调各自独立的 callback group。
  rclcpp::CallbackGroup::SharedPtr color_callback_group_;
  rclcpp::CallbackGroup::SharedPtr depth_callback_group_;
  rclcpp::CallbackGroup::SharedPtr yolo_callback_group_;

  // 三个输入订阅。
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_color_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_yolo_mask_;

  // 三个输出 publisher。
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_mask_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_filtered_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_vis_;
};

int main(int argc, char ** argv)
{
  // 初始化 ROS。
  rclcpp::init(argc, argv);

  // 打开同进程通信优化：如果未来上下游也在同进程，可进一步减少拷贝。
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);

  // 创建节点实例。
  auto node = std::make_shared<DelayedMaskFusionNode>(options);

  // 使用 3 线程执行器，让 color / depth / yolo 三类回调能并发调度。
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
  executor.add_node(node);
  executor.spin();

  // 正常退出时关闭 ROS。
  rclcpp::shutdown();
  return 0;
}
