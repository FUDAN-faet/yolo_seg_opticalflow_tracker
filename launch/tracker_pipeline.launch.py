from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("yolo_seg_opticalflow_tracker")

    model_path = LaunchConfiguration("model_path")
    color_topic = LaunchConfiguration("color_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    yolo_mask_topic = LaunchConfiguration("yolo_mask_topic")
    visualization_topic = LaunchConfiguration("visualization_topic")
    tracked_mask_topic = LaunchConfiguration("tracked_mask_topic")
    tracked_vis_topic = LaunchConfiguration("tracked_vis_topic")
    depth_filtered_topic = LaunchConfiguration("depth_filtered_topic")
    crop_mask_to_box = LaunchConfiguration("crop_mask_to_box")
    ort_root = LaunchConfiguration("ort_root")
    cudnn_lib_dir = LaunchConfiguration("cudnn_lib_dir")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    ld_library_path = [
        ort_root,
        "/lib:",
        cudnn_lib_dir,
        ":",
        EnvironmentVariable("LD_LIBRARY_PATH", default_value=""),
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model_path",
                default_value="/home/zme/yolo_ws/src/models/150epoches-yolo11s-seg.onnx",
                description="Path to the YOLO segmentation ONNX model.",
            ),
            DeclareLaunchArgument(
                "color_topic",
                default_value="/camera_dcw2/sensor_color",
                description="RGB image topic used by both nodes.",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/camera_dcw2/depth/image_raw",
                description="Depth image topic used by delayed_mask_fusion_node.",
            ),
            DeclareLaunchArgument(
                "yolo_mask_topic",
                default_value="/tracking/bottle_mask",
                description="Mask topic published by GPU YOLO and consumed by fusion node.",
            ),
            DeclareLaunchArgument(
                "visualization_topic",
                default_value="/tracking/visualization",
                description="Visualization topic published by gpu_yolo_mask_node.",
            ),
            DeclareLaunchArgument(
                "tracked_mask_topic",
                default_value="/tracking/bottle_mask_tracked",
                description="Tracked mask topic published by delayed_mask_fusion_node.",
            ),
            DeclareLaunchArgument(
                "tracked_vis_topic",
                default_value="/tracking/bottle_mask_tracked_vis",
                description="Tracked visualization topic published by delayed_mask_fusion_node.",
            ),
            DeclareLaunchArgument(
                "depth_filtered_topic",
                default_value="/tracking/depth_filtered_tracked",
                description="Depth output topic published by delayed_mask_fusion_node.",
            ),
            DeclareLaunchArgument(
                "crop_mask_to_box",
                default_value="true",
                description="Crop YOLO segmentation mask to the selected detection box.",
            ),
            DeclareLaunchArgument(
                "ort_root",
                default_value="/opt/onnxruntime-cuda13",
                description="ONNX Runtime installation root.",
            ),
            DeclareLaunchArgument(
                "cudnn_lib_dir",
                default_value="/home/zme/venv/ros_yolo_env/lib/python3.12/site-packages/nvidia/cudnn/lib",
                description="Directory containing libcudnn.so.9.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="false",
                description="Launch RViz alongside the tracking pipeline.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [package_share, "rviz", "tracker_pipeline.rviz"]
                ),
                description="RViz config file path.",
            ),
            SetEnvironmentVariable(name="LD_LIBRARY_PATH", value=ld_library_path),
            Node(
                package="yolo_seg_opticalflow_tracker",
                executable="gpu_yolo_mask_node",
                name="gpu_yolo_mask_node",
                output="screen",
                parameters=[
                    {
                        "model_path": model_path,
                        "color_topic": color_topic,
                        "mask_topic": yolo_mask_topic,
                        "visualization_topic": visualization_topic,
                        "crop_mask_to_box": crop_mask_to_box,
                    }
                ],
            ),
            Node(
                package="yolo_seg_opticalflow_tracker",
                executable="delayed_mask_fusion_node",
                name="delayed_mask_fusion_node",
                output="screen",
                parameters=[
                    {
                        "color_topic": color_topic,
                        "depth_topic": depth_topic,
                        "yolo_mask_topic": yolo_mask_topic,
                        "tracked_mask_topic": tracked_mask_topic,
                        "visualization_topic": tracked_vis_topic,
                        "filtered_depth_topic": depth_filtered_topic,
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(use_rviz),
            ),
        ]
    )
