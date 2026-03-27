# yolo_seg_opticalflow_tracker

一个用于 **ROS 2 图像分割后处理** 的 C++ 节点工程：
- 接收 YOLO 延迟输出的分割 mask；
- 对 RGB 历史帧做光流追踪（catch-up replay）；
- 同步 depth 并做掩膜过滤；
- 发布追踪后的 mask / 过滤后的 depth / 可选可视化结果。

当前核心实现文件为 `src/delayed_mask_fusion_node.cpp`。

## 功能概览

节点 `delayed_mask_fusion_node` 的输入输出如下：

### 订阅
- 彩色图：`/camera_dcw2/sensor_color`（参数 `color_topic`）
- 深度图：`/camera_dcw2/depth/image_raw`（参数 `depth_topic`）
- YOLO mask：`/tracking/bottle_mask`（参数 `yolo_mask_topic`，`mono8`）

### 发布
- 追踪后 mask：`/tracking/bottle_mask_tracked`（参数 `tracked_mask_topic`）
- 过滤后 depth：`/tracking/depth_filtered_tracked`（参数 `filtered_depth_topic`）
- 可视化图（可选）：`/tracking/bottle_mask_tracked_vis`（参数 `visualization_topic`，需 `publish_overlay:=true`）

## 处理流程

1. 缓存 RGB / Depth 历史帧（环形缓冲）。
2. 收到 YOLO mask 后，按时间戳在缓存中找到最接近的 RGB 帧作为基准帧。
3. 用基准帧 + YOLO mask 初始化特征点追踪器。
4. 对基准帧之后的历史 RGB 帧做光流回放（catch-up），把 mask 推进到“最新帧时刻”。
5. 在最新时刻匹配最近的 depth 帧，对 mask 区域做 depth 填充（删除目标区域深度）。
6. 发布追踪 mask 与过滤 depth（以及可视化叠加图）。

## 依赖

至少需要：
- ROS 2（`rclcpp`、`sensor_msgs`）
- OpenCV
- `cv_bridge`

可参考仓库中的依赖片段：
- `CMakeLists_snippet.txt`
- `package_xml_snippet.txt`

## 集成到你的 ROS 2 包

将 `src/delayed_mask_fusion_node.cpp` 放入目标包后，在 `CMakeLists.txt` 和 `package.xml` 补充对应依赖（可直接参考两个 snippet 文件）。

### CMake 最小示例

```cmake
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(delayed_mask_fusion_node src/delayed_mask_fusion_node.cpp)
target_compile_features(delayed_mask_fusion_node PUBLIC cxx_std_17)

ament_target_dependencies(
  delayed_mask_fusion_node
  rclcpp
  sensor_msgs
  cv_bridge
)

target_include_directories(delayed_mask_fusion_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(delayed_mask_fusion_node ${OpenCV_LIBRARIES})

install(TARGETS delayed_mask_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)
```

## 运行示例

先运行你的 YOLO 节点，确保其持续发布 `/tracking/bottle_mask`。然后运行：

```bash
ros2 run <your_pkg> delayed_mask_fusion_node --ros-args \
  -p color_topic:=/camera_dcw2/sensor_color \
  -p depth_topic:=/camera_dcw2/depth/image_raw \
  -p yolo_mask_topic:=/tracking/bottle_mask \
  -p tracked_mask_topic:=/tracking/bottle_mask_tracked \
  -p filtered_depth_topic:=/tracking/depth_filtered_tracked \
  -p publish_overlay:=false \
  -p buffer_max_frames:=180
```

## 关键参数说明

### 同步与缓存
- `buffer_max_frames`：RGB 缓冲长度。
- `depth_buffer_max_frames`：Depth 缓冲长度。
- `lookup_tolerance_sec`：YOLO mask 对齐 RGB 时允许的最大时间误差（秒）。
- `depth_lookup_tolerance_sec`：输出时对齐 depth 的最大时间误差（秒）。

### 光流追踪
- `feature_max_corners`：初始/重检测最大特征点数量。
- `feature_quality_level`、`feature_min_distance`、`feature_block_size`：`goodFeaturesToTrack` 参数。
- `lk_win_size`、`lk_max_level`：金字塔 LK 光流参数。
- `min_good_points`：估计仿射变换所需的最小有效点数。
- `re_detect_interval`：每隔多少帧触发一次特征点重检测。
- `morph_kernel`：mask 形态学开闭运算核大小。

### 深度过滤
- `mask_threshold`：mask 二值阈值。
- `depth_fill_mode`：`32FC1` depth 的填充值模式，支持 `infinity` / `zero` / `nan`。
- `masked_value_16uc1`：`16UC1` depth 的填充值（默认 `65535`）。

### 日志与可视化
- `publish_overlay`：是否发布可视化叠加图。
- `log_every_n_frames`：每 N 帧输出一次发布日志。

## 工程现状与迁移建议

仓库附带了 `migration_notes.md`，给出了从 Python tracker + depth filter 迁移到当前 C++ 单进程版本的建议路径；并建议先保持 YOLO 推理节点独立，后续再考虑 ONNX Runtime / TensorRT 全 C++ 化。

---
如果你希望，我也可以继续帮你补一个 `launch.py`（含所有参数默认值）和推荐的 `rviz2` 显示配置。
