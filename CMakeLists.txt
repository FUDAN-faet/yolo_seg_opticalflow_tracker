cmake_minimum_required(VERSION 3.10)
project(yolo_seg_opticalflow_tracker)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc video dnn)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(ONNXRUNTIME_ROOT "/opt/onnxruntime-cuda13" CACHE PATH "Path to ONNX Runtime installation")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")

if(EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
  set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
elseif(EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1")
  set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1")
elseif(EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.25.0")
  set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.25.0")
else()
  message(FATAL_ERROR "Could not find libonnxruntime under ${ONNXRUNTIME_ROOT}/lib")
endif()

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime/core/session/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "Could not find onnxruntime_cxx_api.h under ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

add_executable(delayed_mask_fusion_node src/delayed_mask_fusion_node.cpp)
target_compile_features(delayed_mask_fusion_node PUBLIC cxx_std_17)
ament_target_dependencies(
  delayed_mask_fusion_node
  builtin_interfaces
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
)
target_include_directories(delayed_mask_fusion_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(delayed_mask_fusion_node ${OpenCV_LIBRARIES})

add_executable(gpu_yolo_mask_node src/gpu_yolo_mask_node.cpp)
target_compile_features(gpu_yolo_mask_node PUBLIC cxx_std_17)
ament_target_dependencies(
  gpu_yolo_mask_node
  builtin_interfaces
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
)
target_include_directories(
  gpu_yolo_mask_node
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(
  gpu_yolo_mask_node
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_LIB}
)
set_target_properties(
  gpu_yolo_mask_node
  PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

install(
  TARGETS
    delayed_mask_fusion_node
    gpu_yolo_mask_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    launch
    rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
