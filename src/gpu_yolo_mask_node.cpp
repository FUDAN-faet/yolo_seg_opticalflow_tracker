// 标准库：算法、并发、文件检查、时间统计、线程同步等基础设施。
#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// ROS / OpenCV / ONNX Runtime 相关头文件：
// - builtin_interfaces::msg::Time 用于时间戳换算
// - cv_bridge 负责 ROS Image 和 cv::Mat 的双向转换
// - OpenCV DNN 这里只保留 blobFromImage / NMSBoxes 这些实用函数
// - ONNX Runtime 负责真正的 ONNX 推理，并优先调度 CUDAExecutionProvider
// - rclcpp / sensor_msgs / std_msgs 是 ROS 2 节点和消息定义
#include <builtin_interfaces/msg/time.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

namespace
{
// 把 ROS 时间戳统一转换成 double 秒，便于日志和比较。
double toSec(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

// 全部转小写，便于做不区分大小写的后缀判断。
std::string toLower(std::string text)
{
  std::transform(
    text.begin(), text.end(), text.begin(),
    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

// 判断字符串是否以指定后缀结尾，忽略大小写。
bool endsWithCaseInsensitive(const std::string & text, const std::string & suffix)
{
  if (suffix.size() > text.size()) {
    return false;
  }
  return toLower(text.substr(text.size() - suffix.size())) == toLower(suffix);
}

// 把模型路径扩展名替换成新的后缀，例如 .pt -> .onnx。
std::string swapExtension(const std::string & path, const std::string & new_ext)
{
  const auto pos = path.find_last_of('.');
  if (pos == std::string::npos) {
    return path + new_ext;
  }
  return path.substr(0, pos) + new_ext;
}

// 简单检查文件是否存在。
bool fileExists(const std::string & path)
{
  std::ifstream fin(path);
  return fin.good();
}

// 通用钳制函数：保证数值始终落在 [low, high]。
template<typename T>
T clampValue(T value, T low, T high)
{
  return std::max(low, std::min(value, high));
}
}  // namespace

// 这是推理线程实际消费的“待处理帧”结构。
// 这里只保留三样东西：
// 1. header：后面发布 mask / 可视化时要把原时间戳和 frame_id 带回去
// 2. frame_bgr：真实做 YOLO 的彩色图像
// 3. stamp_sec：便于打印日志和延迟观察
struct PendingFrame
{
  std_msgs::msg::Header header;
  cv::Mat frame_bgr;
  double stamp_sec{};
};

// 预处理阶段记录下来的几何信息。
// 后处理时需要靠这些信息把网络输出重新映射回原图坐标系。
struct PreprocessInfo
{
  double scale{1.0};      // 原 ROI 缩放到网络输入尺寸时使用的缩放系数
  int pad_x{0};           // letterbox 左右填充量
  int pad_y{0};           // letterbox 上下填充量
  int resized_w{0};       // 缩放后的 ROI 宽度
  int resized_h{0};       // 缩放后的 ROI 高度
  cv::Size roi_size;      // 实际参与推理的 ROI 尺寸
  cv::Rect roi_bounds;    // ROI 在整张图上的位置
  bool has_roi{false};    // 当前是否真的启用了局部 ROI
};

// 单个检测目标的最小表达：
// - box 是还原回 ROI 坐标系后的边界框
// - score 是该类别置信度
// - mask_coeffs 是分割原型图的线性组合系数
struct YoloDetection
{
  cv::Rect box;
  float score{0.0F};
  std::vector<float> mask_coeffs;
};

// 这个节点的职责很单一：
// 订阅 RGB -> 做 YOLO 分割 -> 发布 mono8 mask -> 可选发布可视化。
// 光流追踪和 depth 过滤不在这里做，而是交给下游 delayed_mask_fusion_node。
class GpuYoloMaskNode : public rclcpp::Node
{
public:
  explicit GpuYoloMaskNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("gpu_yolo_mask_node", options)
  {
    // 先声明所有参数，方便 ros2 run / launch / yaml 外部配置。
    declare_parameter<bool>("gpu", true);
    declare_parameter<std::string>("model_path", "/home/zme/yolo_ws/src/models/150epoches-yolo11s-seg.onnx");
    declare_parameter<int>("target_class_id", 0);
    declare_parameter<std::string>("color_topic", "/camera_dcw2/sensor_color");
    declare_parameter<std::string>("mask_topic", "/tracking/bottle_mask");
    declare_parameter<std::string>("visualization_topic", "/tracking/visualization");
    declare_parameter<double>("conf_threshold", 0.35);
    declare_parameter<double>("nms_threshold", 0.45);
    declare_parameter<double>("mask_prob_threshold", 0.5);
    declare_parameter<int>("inference_imgsz", 320);
    declare_parameter<int>("max_det", 1);
    declare_parameter<bool>("enable_hand_roi", false);
    declare_parameter<double>("hand_roi_x_min", 0.0);
    declare_parameter<double>("hand_roi_y_min", 0.0);
    declare_parameter<double>("hand_roi_x_max", 1.0);
    declare_parameter<double>("hand_roi_y_max", 1.0);
    declare_parameter<bool>("gpu_warmup", true);
    declare_parameter<bool>("publish_visualization", true);
    // Ultralytics 的 result.masks 本质上是实例级 mask，
    // 后处理里已经按检测框做过裁剪。
    // 这里默认打开同样的行为，避免 proto 组合后在整张图上出现大面积“串色”。
    declare_parameter<bool>("crop_mask_to_box", true);
    declare_parameter<int>("min_inference_interval_ms", 0);
    declare_parameter<int>("log_every_n_frames", 30);

    // 读取参数到成员变量，后续所有行为都基于这些配置。
    use_gpu_ = get_parameter("gpu").as_bool();
    requested_model_path_ = get_parameter("model_path").as_string();
    target_class_id_ = get_parameter("target_class_id").as_int();
    color_topic_ = get_parameter("color_topic").as_string();
    mask_topic_ = get_parameter("mask_topic").as_string();
    visualization_topic_ = get_parameter("visualization_topic").as_string();
    conf_threshold_ = static_cast<float>(get_parameter("conf_threshold").as_double());
    nms_threshold_ = static_cast<float>(get_parameter("nms_threshold").as_double());
    mask_prob_threshold_ = static_cast<float>(get_parameter("mask_prob_threshold").as_double());
    input_size_ = get_parameter("inference_imgsz").as_int();
    max_det_ = get_parameter("max_det").as_int();
    enable_hand_roi_ = get_parameter("enable_hand_roi").as_bool();
    hand_roi_x_min_ = get_parameter("hand_roi_x_min").as_double();
    hand_roi_y_min_ = get_parameter("hand_roi_y_min").as_double();
    hand_roi_x_max_ = get_parameter("hand_roi_x_max").as_double();
    hand_roi_y_max_ = get_parameter("hand_roi_y_max").as_double();
    gpu_warmup_ = get_parameter("gpu_warmup").as_bool();
    publish_visualization_ = get_parameter("publish_visualization").as_bool();
    crop_mask_to_box_ = get_parameter("crop_mask_to_box").as_bool();
    min_inference_interval_ms_ = get_parameter("min_inference_interval_ms").as_int();
    log_every_n_frames_ = get_parameter("log_every_n_frames").as_int();

    // 传感器类话题只保留最新一帧，目标是低延迟而不是完整处理所有旧帧。
    auto sensor_qos = rclcpp::SensorDataQoS();
    sensor_qos.keep_last(1);

    // 发布 mono8 mask。
    pub_mask_ = create_publisher<sensor_msgs::msg::Image>(mask_topic_, sensor_qos);

    // 发布叠加边框和 mask 的可视化图。
    pub_visualization_ =
      create_publisher<sensor_msgs::msg::Image>(visualization_topic_, sensor_qos);

    // 订阅 RGB 图像。
    sub_color_ = create_subscription<sensor_msgs::msg::Image>(
      color_topic_, sensor_qos,
      std::bind(&GpuYoloMaskNode::onColor, this, std::placeholders::_1));

    // 如果用户给的是 .pt，这里会尝试自动推导同名 .onnx。
    model_path_ = resolveModelPath(requested_model_path_);

    // 初始化 ONNX Runtime 会话和推理后端。
    initSession();

    // 如果推理后端没有成功就绪，就不要继续启动工作线程，
    // 也不要打印“Started”，避免把初始化失败误显示成正常启动。
    if (!net_ready_) {
      backend_name_ = "init_failed";
      RCLCPP_ERROR(
        get_logger(),
        "YOLO node initialization failed. model=%s backend=%s",
        model_path_.c_str(),
        backend_name_.c_str());
      return;
    }

    // 单独起一条工作线程做推理，避免 ROS 回调线程直接被前向推理阻塞。
    worker_thread_ = std::thread(&GpuYoloMaskNode::workerLoop, this);

    // 启动日志，方便确认当前使用的 topic、模型路径和后端。
    RCLCPP_INFO(
      get_logger(),
      "Started. color=%s mask=%s vis=%s model=%s backend=%s",
      color_topic_.c_str(),
      mask_topic_.c_str(),
      visualization_topic_.c_str(),
      model_path_.c_str(),
      backend_name_.c_str());
  }

  // 析构时要优雅地停掉工作线程。
  ~GpuYoloMaskNode() override
  {
    {
      // 先把停止标志设起来。
      std::scoped_lock<std::mutex> lock(queue_mutex_);
      stop_worker_ = true;
    }

    // 唤醒可能还在等待队列的工作线程。
    queue_cv_.notify_all();

    // 等待工作线程彻底退出。
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
  }

private:
  // 初始化 ONNX Runtime 会话：
  // 1. 校验模型路径
  // 2. 创建 SessionOptions
  // 3. 优先追加 CUDAExecutionProvider
  // 4. 再追加 CPUExecutionProvider 作为兜底
  // 5. 载入模型并缓存输入输出名字
  // 6. 可选做一次 warmup
  void initSession()
  {
    // 当前 C++ 实现只接受 ONNX 模型，不直接解释 Ultralytics 的 .pt。
    if (!endsWithCaseInsensitive(model_path_, ".onnx")) {
      RCLCPP_ERROR(
        get_logger(),
        "Current C++ YOLO implementation expects an ONNX model. requested=%s resolved=%s",
        requested_model_path_.c_str(),
        model_path_.c_str());
      return;
    }

    // 模型文件不存在就直接报错，后面不再继续初始化。
    if (!fileExists(model_path_)) {
      RCLCPP_ERROR(
        get_logger(),
        "Model file not found: %s. Export your Ultralytics .pt model to .onnx first.",
        model_path_.c_str());
      return;
    }

    try {
      // SessionOptions 控制图优化级别、线程数和 EP 选择策略。
      session_options_ = Ort::SessionOptions{};
      session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
      session_options_.SetExecutionMode(ORT_SEQUENTIAL);
      session_options_.SetIntraOpNumThreads(1);
      session_options_.SetInterOpNumThreads(1);
      session_options_.DisableMemPattern();

      // 先尝试 CUDAExecutionProvider；失败时不让节点崩溃，而是回退到 CPU。
      bool cuda_enabled = false;
      if (use_gpu_) {
        try {
          // 这里改用旧版的公开 C 结构体而不是 V2 的字符串 Update 接口。
          // 之前的 V2 路径在你的环境里会在解析 "device_id" 时触发 cudaGetDeviceCount()
          // 并直接报 CUDA 999，导致 provider 还没真正挂上就回退到 CPU。
          // 这一版直接填字段，先把字符串解析这一层不稳定因素拿掉。
          OrtCUDAProviderOptions cuda_options;
          cuda_options.device_id = 0;
          cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
          cuda_options.do_copy_in_default_stream = 1;
          session_options_.AppendExecutionProvider_CUDA(cuda_options);
          cuda_enabled = true;
        } catch (const Ort::Exception & e) {
          RCLCPP_WARN(
            get_logger(),
            "Failed to enable CUDAExecutionProvider (%s). Falling back to CPU.",
            e.what());
        }
      }

      // 这套 ORT 头文件里 CPU provider 的 C++ 包装不可用，
      // 这里直接依赖 ORT 默认自带的 CPU EP 作为兜底。
      if (!cuda_enabled) {
        backend_name_ = "onnxruntime_cpu";
      }

      // 正式创建 Session。
      session_ = std::make_unique<Ort::Session>(ort_env_, model_path_.c_str(), session_options_);

      // 读取模型的输入输出名字，后续每次 Run 都直接复用。
      loadModelIoNames();

      // 打印 ORT 当前可见的 provider，方便确认运行环境。
      const auto providers = Ort::GetAvailableProviders();
      std::string provider_list;
      for (size_t i = 0; i < providers.size(); ++i) {
        provider_list += providers[i];
        if (i + 1 < providers.size()) {
          provider_list += ",";
        }
      }
      RCLCPP_INFO(get_logger(), "ORT available providers: %s", provider_list.c_str());

      // 输入 tensor 形状固定为 [1, 3, input_size, input_size]。
      input_tensor_shape_ = {1, 3, input_size_, input_size_};

      // 如果模型自己声明了固定输入宽高，这里优先让运行参数跟模型对齐。
      if (session_->GetInputCount() > 0) {
        auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape.size() == 4 && input_shape[2] > 0 && input_shape[3] > 0) {
          if (input_shape[2] != input_shape[3]) {
            RCLCPP_WARN(
              get_logger(),
              "Model input shape is not square (%ld x %ld). Current node still assumes square preprocessing.",
              static_cast<long>(input_shape[2]),
              static_cast<long>(input_shape[3]));
          }

          if (input_size_ != static_cast<int>(input_shape[2])) {
            RCLCPP_WARN(
              get_logger(),
              "inference_imgsz=%d does not match model input height=%ld. Overriding to match model.",
              input_size_,
              static_cast<long>(input_shape[2]));
            input_size_ = static_cast<int>(input_shape[2]);
            input_tensor_shape_ = {1, 3, input_size_, input_size_};
          }
        }
      }

      // 只有 Session 真正创建成功后，才把后端名定下来。
      if (cuda_enabled) {
        backend_name_ = "onnxruntime_cuda";
      }

      net_ready_ = true;
    } catch (const Ort::Exception & e) {
      backend_name_ = "init_failed";
      RCLCPP_ERROR(get_logger(), "Failed to initialize ONNX Runtime session: %s", e.what());
      return;
    } catch (const std::exception & e) {
      backend_name_ = "init_failed";
      RCLCPP_ERROR(get_logger(), "Failed to initialize ONNX Runtime session: %s", e.what());
      return;
    }

    // 可选预热：第一次 Run 往往慢，先用 dummy 图跑一遍能减少正式首帧抖动。
    if (gpu_warmup_ && net_ready_) {
      warmup();
    }
  }

  // 如果用户传的是 .pt，就尝试找同目录同名的 .onnx。
  // 否则直接返回原路径。
  std::string resolveModelPath(const std::string & requested) const
  {
    if (endsWithCaseInsensitive(requested, ".pt")) {
      const std::string candidate = swapExtension(requested, ".onnx");
      if (fileExists(candidate)) {
        return candidate;
      }
    }
    return requested;
  }

  // 读取并缓存模型的输入输出节点名字。
  void loadModelIoNames()
  {
    input_name_storage_.clear();
    output_name_storage_.clear();
    input_names_.clear();
    output_names_.clear();

    Ort::AllocatorWithDefaultOptions allocator;

    const size_t input_count = session_->GetInputCount();
    const size_t output_count = session_->GetOutputCount();

    input_name_storage_.reserve(input_count);
    output_name_storage_.reserve(output_count);
    input_names_.reserve(input_count);
    output_names_.reserve(output_count);

    for (size_t i = 0; i < input_count; ++i) {
      auto name = session_->GetInputNameAllocated(i, allocator);
      input_name_storage_.emplace_back(name.get());
      input_names_.push_back(input_name_storage_.back().c_str());
    }

    for (size_t i = 0; i < output_count; ++i) {
      auto name = session_->GetOutputNameAllocated(i, allocator);
      output_name_storage_.emplace_back(name.get());
      output_names_.push_back(output_name_storage_.back().c_str());
    }

    RCLCPP_INFO(
      get_logger(),
      "Loaded ORT model IO. inputs=%zu outputs=%zu first_input=%s",
      input_count,
      output_count,
      input_names_.empty() ? "<none>" : input_names_.front());
  }

  // 真正执行一次 ONNX Runtime 推理。
  // 这里保留两份输出：
  // 1. ort_outputs：负责真正持有 ORT 输出张量内存
  // 2. cv_outputs：把 ORT 输出包装成 cv::Mat 视图，方便复用现有后处理代码
  bool runModel(
    const cv::Mat & blob,
    std::vector<Ort::Value> & ort_outputs,
    std::vector<cv::Mat> & cv_outputs)
  {
    if (!net_ready_ || session_ == nullptr) {
      return false;
    }

    // ONNX Runtime 的输入 tensor 需要连续内存。
    const cv::Mat blob_contiguous = blob.isContinuous() ? blob : blob.clone();

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      const_cast<float *>(blob_contiguous.ptr<float>()),
      blob_contiguous.total(),
      input_tensor_shape_.data(),
      input_tensor_shape_.size());

    std::array<Ort::Value, 1> input_values{std::move(input_tensor)};
    Ort::RunOptions run_options;

    ort_outputs = session_->Run(
      run_options,
      input_names_.data(),
      input_values.data(),
      input_values.size(),
      output_names_.data(),
      output_names_.size());

    cv_outputs.clear();
    cv_outputs.reserve(ort_outputs.size());

    for (auto & output : ort_outputs) {
      if (!output.IsTensor()) {
        RCLCPP_ERROR(get_logger(), "ORT output is not a tensor.");
        return false;
      }

      auto tensor_info = output.GetTensorTypeAndShapeInfo();
      if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        RCLCPP_ERROR(get_logger(), "ORT output tensor element type is not float.");
        return false;
      }

      auto shape = tensor_info.GetShape();
      if (shape.empty()) {
        RCLCPP_ERROR(get_logger(), "ORT output tensor has empty shape.");
        return false;
      }

      std::vector<int> sizes;
      sizes.reserve(shape.size());
      for (const auto dim : shape) {
        if (dim <= 0) {
          RCLCPP_ERROR(
            get_logger(),
            "ORT output tensor has non-positive dimension: %ld",
            static_cast<long>(dim));
          return false;
        }
        sizes.push_back(static_cast<int>(dim));
      }

      cv_outputs.emplace_back(
        static_cast<int>(sizes.size()),
        sizes.data(),
        CV_32F,
        output.GetTensorMutableData<float>());
    }

    return true;
  }

  // 用一张固定 dummy 图做一次前向推理，完成后端预热。
  void warmup()
  {
    // 114 是很多 YOLO letterbox 的常用填充值。
    cv::Mat dummy(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));

    try {
      // 生成网络输入 blob。
      cv::Mat blob = cv::dnn::blobFromImage(
        dummy, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);

      // 只要能成功跑一遍 Run，就说明 session 和 provider 已经可用。
      std::vector<Ort::Value> ort_outputs;
      std::vector<cv::Mat> outputs;
      if (runModel(blob, ort_outputs, outputs)) {
        RCLCPP_INFO(get_logger(), "YOLO backend warmup completed.");
      }
    } catch (const Ort::Exception & e) {
      RCLCPP_WARN(get_logger(), "YOLO warmup skipped because inference failed: %s", e.what());
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "YOLO warmup skipped because inference failed: %s", e.what());
    }
  }

  // RGB 回调本身不做推理，只负责：
  // 1. 把 ROS Image 转成 BGR
  // 2. 丢进单槽队列
  // 3. 唤醒后台推理线程
  void onColor(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    // 网络没准备好时，收到图像也不处理。
    if (!net_ready_) {
      return;
    }

    cv::Mat frame_bgr;
    try {
      // 这里 clone 一次，确保推理线程拿到独立图像副本。
      frame_bgr = cv_bridge::toCvCopy(msg, "bgr8")->image.clone();
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Failed to convert color image: %s", e.what());
      return;
    }

    std::scoped_lock<std::mutex> lock(queue_mutex_);

    // 取当前单调时钟，用来限制最小推理间隔。
    const auto now = std::chrono::steady_clock::now();

    // 如果设置了最小推理间隔，并且还没到时间，就直接跳过这帧。
    if (
      min_inference_interval_ms_ > 0 &&
      now - last_enqueue_time_ < std::chrono::milliseconds(min_inference_interval_ms_))
    {
      return;
    }

    // 单槽队列设计：
    // 如果后台线程还没消费掉旧帧，新帧会直接覆盖旧帧，这样总延迟最低。
    if (pending_frame_.has_value()) {
      ++dropped_frames_;
    }

    // 把当前帧放进待处理槽位。
    pending_frame_ = PendingFrame{msg->header, std::move(frame_bgr), toSec(msg->header.stamp)};
    last_enqueue_time_ = now;

    // 唤醒后台推理线程。
    queue_cv_.notify_one();
  }

  // 后台工作线程主循环：
  // 只要节点还活着，就不断取最新的一帧做推理。
  void workerLoop()
  {
    while (rclcpp::ok()) {
      std::optional<PendingFrame> work;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // 没有帧时阻塞等待；收到新帧或停止请求时被唤醒。
        queue_cv_.wait(lock, [this]() { return stop_worker_ || pending_frame_.has_value(); });

        // 如果已经要求退出，而且队列里也没有待处理帧，就直接结束线程。
        if (stop_worker_ && !pending_frame_.has_value()) {
          return;
        }

        // 取走当前待处理帧，并清空槽位。
        work = std::move(pending_frame_);
        pending_frame_.reset();
      }

      // 理论上不会进这里，但保险起见还是防一下空值。
      if (!work.has_value()) {
        continue;
      }

      // 真正的预处理、推理、后处理和发布都在这里完成。
      processFrame(*work);
    }
  }

  // 根据配置决定是否只在手部 ROI 内跑推理。
  // 如果没有启用 ROI，就直接返回整张图。
  cv::Mat extractRoi(const cv::Mat & color_frame, PreprocessInfo & prep) const
  {
    // 先默认 ROI 就是整张图。
    prep.roi_size = color_frame.size();
    prep.roi_bounds = cv::Rect(0, 0, color_frame.cols, color_frame.rows);
    prep.has_roi = false;

    // 不启用 ROI 时直接返回原图。
    if (!enable_hand_roi_) {
      return color_frame;
    }

    const int h = color_frame.rows;
    const int w = color_frame.cols;

    // ROI 参数以 0~1 的相对比例传入，这里转换成像素坐标。
    const int x1 = clampValue(static_cast<int>(std::round(hand_roi_x_min_ * w)), 0, w);
    const int y1 = clampValue(static_cast<int>(std::round(hand_roi_y_min_ * h)), 0, h);
    const int x2 = clampValue(static_cast<int>(std::round(hand_roi_x_max_ * w)), 0, w);
    const int y2 = clampValue(static_cast<int>(std::round(hand_roi_y_max_ * h)), 0, h);

    // 非法 ROI 直接退回整图。
    if (x2 <= x1 || y2 <= y1) {
      RCLCPP_WARN(get_logger(), "Invalid hand ROI, fallback to full frame.");
      return color_frame;
    }

    // 记录 ROI 真正的矩形边界。
    prep.roi_bounds = cv::Rect(x1, y1, x2 - x1, y2 - y1);
    prep.roi_size = prep.roi_bounds.size();
    prep.has_roi = true;

    // 返回 ROI 视图，避免不必要拷贝。
    return color_frame(prep.roi_bounds);
  }

  // 预处理：
  // 1. 把 ROI 缩放到网络输入大小
  // 2. 做 letterbox 填充
  // 3. 转成 NCHW blob
  cv::Mat makeInputBlob(const cv::Mat & roi_frame, PreprocessInfo & prep) const
  {
    // 计算保持长宽比时的缩放比例。
    prep.scale = std::min(
      static_cast<double>(input_size_) / static_cast<double>(roi_frame.cols),
      static_cast<double>(input_size_) / static_cast<double>(roi_frame.rows));

    // 缩放后的尺寸。
    prep.resized_w = std::max(1, static_cast<int>(std::round(roi_frame.cols * prep.scale)));
    prep.resized_h = std::max(1, static_cast<int>(std::round(roi_frame.rows * prep.scale)));

    // 计算左右 / 上下填充量。
    prep.pad_x = std::max(0, (input_size_ - prep.resized_w) / 2);
    prep.pad_y = std::max(0, (input_size_ - prep.resized_h) / 2);

    // 先缩放 ROI。
    cv::Mat resized;
    cv::resize(
      roi_frame, resized, cv::Size(prep.resized_w, prep.resized_h), 0.0, 0.0, cv::INTER_LINEAR);

    // 再放到固定大小的画布中央，完成 letterbox。
    cv::Mat canvas(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(prep.pad_x, prep.pad_y, prep.resized_w, prep.resized_h)));

    // BGR -> RGB、归一化到 0~1、转 NCHW blob。
    return cv::dnn::blobFromImage(
      canvas, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
  }

  // 在多个输出里找到“检测预测张量”。
  // YOLO-seg 常见导出里它通常是一个 3 维张量。
  static const cv::Mat * findPredictionOutput(const std::vector<cv::Mat> & outputs)
  {
    for (const auto & output : outputs) {
      if (output.dims == 3) {
        return &output;
      }
    }
    return nullptr;
  }

  // 在多个输出里找到“mask proto 原型图”。
  // 它通常是一个 4 维张量。
  static const cv::Mat * findProtoOutput(const std::vector<cv::Mat> & outputs)
  {
    for (const auto & output : outputs) {
      if (output.dims == 4) {
        return &output;
      }
    }
    return nullptr;
  }

  // 从网络输出里解码出“最优的那个目标”。
  // 这里沿用了 Python 版本的低计算模式：只保留一个目标实例。
  std::optional<YoloDetection> decodeBestDetection(
    const std::vector<cv::Mat> & outputs,
    const PreprocessInfo & prep) const
  {
    // 先分别找到检测输出和 mask proto 输出。
    const cv::Mat * pred_raw = findPredictionOutput(outputs);
    const cv::Mat * proto_raw = findProtoOutput(outputs);
    if (pred_raw == nullptr || proto_raw == nullptr) {
      return std::nullopt;
    }

    // 推断输出布局：
    // pred_channels = 4(box) + class_count + mask_dim。
    const int pred_channels = pred_raw->size[1];
    const int num_preds = pred_raw->size[2];
    const int mask_dim = proto_raw->size[1];
    const int class_count = pred_channels - 4 - mask_dim;

    // 如果类别数都不够，说明当前解析逻辑和模型输出不匹配。
    if (class_count <= target_class_id_) {
      RCLCPP_ERROR(
        get_logger(),
        "Model output layout mismatch. pred_channels=%d mask_dim=%d class_count=%d target_class_id=%d",
        pred_channels, mask_dim, class_count, target_class_id_);
      return std::nullopt;
    }

    // 把 (C, N) 视图转成 (N, C)，方便按每个候选框逐行解析。
    cv::Mat pred_mat(
      pred_channels, num_preds, CV_32F,
      const_cast<float *>(pred_raw->ptr<float>()));
    cv::Mat pred_t;
    cv::transpose(pred_mat, pred_t);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<std::vector<float>> coeffs;

    boxes.reserve(num_preds);
    scores.reserve(num_preds);
    coeffs.reserve(num_preds);

    // 逐个候选框解析。
    for (int i = 0; i < pred_t.rows; ++i) {
      const float * row = pred_t.ptr<float>(i);

      // 当前目标类别的置信度。
      const float score = row[4 + target_class_id_];
      if (score < conf_threshold_) {
        continue;
      }

      // YOLO 原始框参数：中心点 + 宽高。
      const float cx = row[0];
      const float cy = row[1];
      const float w = row[2];
      const float h = row[3];

      // 先去掉 letterbox padding，再按缩放比例还原回 ROI 坐标。
      const float left = static_cast<float>((cx - 0.5F * w - prep.pad_x) / prep.scale);
      const float top = static_cast<float>((cy - 0.5F * h - prep.pad_y) / prep.scale);
      const float right = static_cast<float>((cx + 0.5F * w - prep.pad_x) / prep.scale);
      const float bottom = static_cast<float>((cy + 0.5F * h - prep.pad_y) / prep.scale);

      // 裁剪到 ROI 范围内，避免出现负坐标或越界框。
      const int x1 = clampValue(static_cast<int>(std::floor(left)), 0, prep.roi_size.width - 1);
      const int y1 = clampValue(static_cast<int>(std::floor(top)), 0, prep.roi_size.height - 1);
      const int x2 = clampValue(static_cast<int>(std::ceil(right)), 0, prep.roi_size.width);
      const int y2 = clampValue(static_cast<int>(std::ceil(bottom)), 0, prep.roi_size.height);

      // 无效框直接跳过。
      if (x2 <= x1 || y2 <= y1) {
        continue;
      }

      // 保存框、分数以及对应的 mask 系数。
      boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
      scores.push_back(score);
      coeffs.emplace_back(row + 4 + class_count, row + pred_channels);
    }

    // 一个框都没有时，说明当前帧无目标。
    if (boxes.empty()) {
      return std::nullopt;
    }

    // 先对候选框做 NMS。
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, indices, 1.0F, max_det_);
    if (indices.empty()) {
      return std::nullopt;
    }

    // 在通过 NMS 的候选里再选分数最高的那个。
    int best_index = indices.front();
    float best_score = scores[best_index];
    for (const int idx : indices) {
      if (scores[idx] > best_score) {
        best_score = scores[idx];
        best_index = idx;
      }
    }

    return YoloDetection{boxes[best_index], scores[best_index], coeffs[best_index]};
  }

  // 根据最优检测的 mask 系数和 proto 输出，还原整张图的二值 mask。
  cv::Mat buildMask(
    const std::vector<cv::Mat> & outputs,
    const YoloDetection & detection,
    const PreprocessInfo & prep,
    const cv::Size & full_image_size) const
  {
    const cv::Mat * proto_raw = findProtoOutput(outputs);
    if (proto_raw == nullptr) {
      return cv::Mat::zeros(full_image_size, CV_8UC1);
    }

    // proto 的尺寸通常是 [1, mask_dim, mask_h, mask_w]。
    const int mask_dim = proto_raw->size[1];
    const int mask_h = proto_raw->size[2];
    const int mask_w = proto_raw->size[3];

    // 把 proto 视作 (mask_dim, mask_h * mask_w)。
    cv::Mat proto(
      mask_dim, mask_h * mask_w, CV_32F,
      const_cast<float *>(proto_raw->ptr<float>()));

    // 检测框自带的 mask 系数是一个 1 x mask_dim 向量。
    cv::Mat coeff(1, mask_dim, CV_32F, const_cast<float *>(detection.mask_coeffs.data()));

    // 线性组合 proto，得到这个检测框自己的 mask logits。
    cv::Mat mask_flat = coeff * proto;
    cv::Mat mask_logits = mask_flat.reshape(1, mask_h);

    // 做 sigmoid，把 logits 变成概率图。
    cv::Mat exp_mask;
    cv::exp(-mask_logits, exp_mask);
    cv::Mat mask_prob = 1.0F / (1.0F + exp_mask);

    // 先把概率图缩回网络输入大小。
    cv::Mat mask_input;
    cv::resize(
      mask_prob, mask_input, cv::Size(input_size_, input_size_), 0.0, 0.0, cv::INTER_LINEAR);

    // 去掉 letterbox 填充区域，只保留真实内容。
    cv::Rect valid(prep.pad_x, prep.pad_y, prep.resized_w, prep.resized_h);
    valid &= cv::Rect(0, 0, input_size_, input_size_);
    cv::Mat mask_unpadded = mask_input(valid);

    // 再把去 padding 后的 mask 缩放回 ROI 原始大小。
    cv::Mat mask_roi;
    cv::resize(mask_unpadded, mask_roi, prep.roi_size, 0.0, 0.0, cv::INTER_LINEAR);

    // 按概率阈值二值化，得到 0/255 的单通道 mask。
    cv::Mat mask_binary;
    cv::threshold(mask_roi, mask_binary, mask_prob_threshold_, 255.0, cv::THRESH_BINARY);
    mask_binary.convertTo(mask_binary, CV_8UC1);

    // Ultralytics 的分割后处理会把实例 mask 限制在检测框内。
    // 如果不做这一步，proto 线性组合很容易在目标外的区域激活，
    // 看起来就像“框是对的，但整幅图被错误涂绿了”。
    if (crop_mask_to_box_) {
      cv::Mat cropped = cv::Mat::zeros(mask_binary.size(), mask_binary.type());
      const cv::Rect clipped_box =
        detection.box & cv::Rect(0, 0, prep.roi_size.width, prep.roi_size.height);
      if (clipped_box.area() > 0) {
        mask_binary(clipped_box).copyTo(cropped(clipped_box));
      }
      mask_binary = std::move(cropped);
    }

    // 如果没有启用 ROI，当前 mask 就已经是整图尺寸。
    if (!prep.has_roi) {
      return mask_binary;
    }

    // 如果启用了 ROI，需要把 ROI 内的 mask 放回整图对应位置。
    cv::Mat full_mask = cv::Mat::zeros(full_image_size, CV_8UC1);
    mask_binary.copyTo(full_mask(prep.roi_bounds));
    return full_mask;
  }

  // 构建可视化图：
  // 1. 把 mask 叠加到原图
  // 2. 画出检测框
  // 3. 写上类别和分数
  cv::Mat buildVisualization(
    const cv::Mat & color_frame,
    const cv::Mat & mask,
    const std::optional<YoloDetection> & detection,
    const PreprocessInfo & prep) const
  {
    // 先复制一张原图。
    cv::Mat vis = color_frame.clone();

    // 如果当前帧有 mask，就把它用绿色半透明方式叠加上去。
    if (!mask.empty() && cv::countNonZero(mask) > 0) {
      cv::Mat overlay = cv::Mat::zeros(color_frame.size(), color_frame.type());
      std::vector<cv::Mat> channels;
      cv::split(overlay, channels);
      channels[1] = mask;
      cv::merge(channels, overlay);
      cv::addWeighted(vis, 1.0, overlay, 0.5, 0.0, vis);
    }

    // 如果有检测结果，就把框也画出来。
    if (detection.has_value()) {
      cv::Rect box = detection->box;

      // 如果推理时用了 ROI，需要把框偏移回整图坐标。
      if (prep.has_roi) {
        box.x += prep.roi_bounds.x;
        box.y += prep.roi_bounds.y;
      }

      // 最后再做一次边界裁剪。
      box &= cv::Rect(0, 0, color_frame.cols, color_frame.rows);

      if (box.area() > 0) {
        cv::rectangle(vis, box, cv::Scalar(0, 255, 0), 2);
        cv::putText(
          vis,
          "class_" + std::to_string(target_class_id_) + ": " + cv::format("%.2f", detection->score),
          cv::Point(box.x, std::max(0, box.y - 10)),
          cv::FONT_HERSHEY_SIMPLEX,
          0.7,
          cv::Scalar(0, 255, 0),
          2);
      }
    }

    return vis;
  }

  // 对单帧执行完整流程：
  // 取 ROI -> 预处理 -> 前向推理 -> 解码检测 -> 生成 mask -> 发布。
  void processFrame(const PendingFrame & frame)
  {
    // 用来记录预处理几何信息。
    PreprocessInfo prep;

    // 根据配置可能取局部 ROI，也可能直接用整图。
    cv::Mat roi = extractRoi(frame.frame_bgr, prep);

    // 记录本次推理起始时间，用于统计延迟。
    const auto start = std::chrono::steady_clock::now();

    // 把 ROI 变成网络输入。
    cv::Mat blob = makeInputBlob(roi, prep);

    std::vector<Ort::Value> ort_outputs;
    std::vector<cv::Mat> outputs;
    try {
      // 把 blob 送进 ONNX Runtime，并把输出重新包装成 cv::Mat 视图。
      if (!runModel(blob, ort_outputs, outputs)) {
        return;
      }
    } catch (const Ort::Exception & e) {
      RCLCPP_ERROR(get_logger(), "YOLO forward failed: %s", e.what());
      return;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "YOLO forward failed: %s", e.what());
      return;
    }

    // 只解码当前帧最优的那个目标。
    const auto detection = decodeBestDetection(outputs, prep);

    // 默认先给一张全黑 mask；只有检测到目标时才重建真实 mask。
    cv::Mat mask = cv::Mat::zeros(frame.frame_bgr.size(), CV_8UC1);
    if (detection.has_value()) {
      mask = buildMask(outputs, *detection, prep, frame.frame_bgr.size());
    }

    // 把 mask 转回 ROS Image 并发布。
    auto mask_msg = cv_bridge::CvImage(frame.header, "mono8", mask).toImageMsg();
    pub_mask_->publish(*mask_msg);

    // 如果开启可视化，就额外发布可视化图。
    if (publish_visualization_) {
      cv::Mat vis = buildVisualization(frame.frame_bgr, mask, detection, prep);
      auto vis_msg = cv_bridge::CvImage(frame.header, "bgr8", vis).toImageMsg();
      pub_visualization_->publish(*vis_msg);
    }

    // 计算本次推理总耗时。
    const double inference_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();

    // 累加处理帧计数。
    const int processed = ++processed_frames_;

    // 按指定频率打印统计日志。
    if (log_every_n_frames_ > 0 && processed % log_every_n_frames_ == 0) {
      RCLCPP_INFO(
        get_logger(),
        "Published YOLO mask. score=%.3f nonzero=%d latency=%.2fms dropped=%d backend=%s stamp=%.6f",
        detection.has_value() ? detection->score : 0.0F,
        cv::countNonZero(mask),
        inference_ms,
        dropped_frames_.load(),
        backend_name_.c_str(),
        frame.stamp_sec);
    }
  }

  // 是否优先尝试 GPU。
  bool use_gpu_{true};

  // 当前关注的目标类别 id。
  int target_class_id_{0};

  // 网络输入大小，例如 320。
  int input_size_{320};

  // NMS 后最多保留多少个候选框参与选择。
  int max_det_{1};

  // 是否启用局部 ROI 推理。
  bool enable_hand_roi_{false};

  // ROI 的归一化坐标。
  double hand_roi_x_min_{0.0};
  double hand_roi_y_min_{0.0};
  double hand_roi_x_max_{1.0};
  double hand_roi_y_max_{1.0};

  // 是否在启动时做预热。
  bool gpu_warmup_{true};

  // 是否发布可视化图。
  bool publish_visualization_{true};

  // 是否把 mask 再裁到 box 内部。
  bool crop_mask_to_box_{false};

  // 相邻两次入队之间的最短时间间隔，单位毫秒。
  int min_inference_interval_ms_{0};

  // 每处理多少帧打印一次日志。
  int log_every_n_frames_{30};

  // 置信度阈值。
  float conf_threshold_{0.35F};

  // NMS 阈值。
  float nms_threshold_{0.45F};

  // mask 概率二值化阈值。
  float mask_prob_threshold_{0.5F};

  // 用户配置的原始模型路径。
  std::string requested_model_path_;

  // 实际解析后使用的模型路径。
  std::string model_path_;

  // 输入 / 输出 topic。
  std::string color_topic_;
  std::string mask_topic_;
  std::string visualization_topic_;

  // 当前后端名字，用于日志打印。
  std::string backend_name_{"uninitialized"};

  // 网络是否已经成功就绪。
  bool net_ready_{false};

  // ONNX Runtime 环境对象。一个进程里通常只需要一个 Env。
  Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING, "gpu_yolo_mask_node"};

  // SessionOptions 保存图优化级别、线程和 provider 配置。
  Ort::SessionOptions session_options_;

  // 真正执行推理的 ORT Session。
  std::unique_ptr<Ort::Session> session_;

  // 模型的输入 / 输出名字。
  std::vector<std::string> input_name_storage_;
  std::vector<std::string> output_name_storage_;
  std::vector<const char *> input_names_;
  std::vector<const char *> output_names_;

  // 输入 tensor 固定使用 [1, 3, input_size, input_size]。
  std::array<int64_t, 4> input_tensor_shape_{1, 3, 320, 320};

  // 单槽队列相关同步原语。
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::optional<PendingFrame> pending_frame_;
  bool stop_worker_{false};
  std::thread worker_thread_;
  std::chrono::steady_clock::time_point last_enqueue_time_{};

  // 统计量：
  // - processed_frames_ 表示成功处理并发布的帧数
  // - dropped_frames_ 表示因为“最新帧覆盖旧帧”策略而丢掉的帧数
  std::atomic<int> processed_frames_{0};
  std::atomic<int> dropped_frames_{0};

  // ROS 订阅 / 发布器。
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_color_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_mask_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_visualization_;
};

int main(int argc, char ** argv)
{
  // 初始化 ROS 2。
  rclcpp::init(argc, argv);

  // 创建节点实例。
  auto node = std::make_shared<GpuYoloMaskNode>();

  // 进入事件循环。
  rclcpp::spin(node);

  // 正常退出时关闭 ROS 2。
  rclcpp::shutdown();
  return 0;
}
