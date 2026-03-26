# 迁移建议

## 你现在立刻能落地的版本
这个 `delayed_mask_fusion_node.cpp` 做的是：

- 订阅 RGB
- 订阅 Depth
- 订阅 **YOLO 延迟输出的 mono8 mask**
- 在同一个进程里完成：
  - 历史帧缓存
  - 光流补帧
  - catch-up replay
  - depth mask removal
  - 最终 tracked mask / filtered depth / overlay 发布

这样你先把 **原来的 Python tracker 节点 + Python depth filter 输出链路** 合成一个 C++ 进程，
通常就能明显减少图像序列化、Python 回调开销和跨进程通信负担。

## 还没有完全并进去的部分
你当前 YOLO 仍然建议先保留为单独推理源，继续发布 `/tracking/bottle_mask`。

原因不是做不到，而是你当前用的是 Ultralytics Python + Torch 直接推理 `.pt`。
如果要全 C++，建议走下面二选一：

### 路线 A：ONNX Runtime
1. 先把 `.pt` 导出成 `.onnx`
2. 在 C++ 节点里内置 ONNX Runtime 推理
3. 用同一个进程完成 YOLO + tracker + depth filter

### 路线 B：TensorRT
1. 导出 engine
2. C++ 节点里直接 TensorRT 推理
3. 适合 NVIDIA GPU 上追求最低延迟

## 推荐落地顺序
1. 先把这个 C++ 节点替换掉 Python tracker 节点
2. 保留现有 Python YOLO 节点不动，验证整体链路
3. 确认卡顿明显下降后，再把 YOLO 模型导出到 ONNX/TensorRT
4. 最后再把 YOLO 推理一起并进 C++

## 推荐启动方式
先继续跑你现有的 YOLO 节点，让它发布 `/tracking/bottle_mask`。

然后运行：

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
