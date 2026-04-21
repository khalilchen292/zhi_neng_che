import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # 1. 启动深度相机 (光鉴 Aurora 930) - 独占相机，不再有冲突
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('deptrum-ros-driver-aurora930'),
                'launch',
                'aurora930_launch.py'
            )
        )
    )

    # 2. 启动小车底盘通讯 (纯净版！)
    # 只启动串口通信，剥离所有 Nav2 雷达建图和多余的相机节点，把 /cmd_vel 控制权 100% 交给 YOLO
    chassis_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('origincar_base'),
                'launch',
                'base_serial.launch.py'
            )
        )
    )

    # 3. 启动你写的 YOLOv5 推理与控制节点
    vision_inference_node = Node(
        package='task_vision',
        executable='yolo_node',
        name='yolo_inference_node',
        output='screen'
    )

    # 将所有节点加入启动描述中并返回
    return LaunchDescription([
        camera_launch,
        chassis_launch,
        vision_inference_node
    ])