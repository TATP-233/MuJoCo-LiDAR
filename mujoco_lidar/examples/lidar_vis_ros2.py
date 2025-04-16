import os
import time
import argparse
import threading
import traceback

import mujoco
import mujoco.viewer
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray


from mujoco_lidar.lidar_wrapper import MjLidarWrapper
from mujoco_lidar.scan_gen import LivoxGenerator, generate_vlp32, generate_HDL64, generate_os128

from mujoco_lidar.mj_lidar_utils import create_demo_scene, KeyboardListener, create_marker_from_geom


def publish_scene(publisher, mj_scene, frame_id, stamp):
    """将MuJoCo场景发布为ROS可视化标记数组"""
    marker_array = MarkerArray()

    # 记录当前使用的标记ID
    current_id = 0

    # 创建每个几何体的标记
    for i in range(mj_scene.ngeom):
        geom = mj_scene.geoms[i]
        # 创建标记并返回一个标记列表
        markers = create_marker_from_geom(geom, current_id, frame_id)

        # 添加所有返回的标记到标记数组
        for marker in markers:
            # 在ROS2中，需要设置stamp为ROS2的时间类型
            marker.header.stamp = stamp
            marker_array.markers.append(marker)
            current_id += 1

    # 发布标记数组
    publisher.publish(marker_array)


def publish_point_cloud(publisher, points, frame_id, stamp):
    """将点云数据发布为ROS PointCloud2消息"""

    # 定义点云字段
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
    ]

    # 添加强度值
    if len(points.shape) == 2:
        # 如果是(N, 3)形状，转换为(3, N)以便处理
        points_transposed = points.T if points.shape[1] == 3 else points

        if points_transposed.shape[0] == 3:
            # 添加强度通道
            points_with_intensity = np.vstack([
                points_transposed, 
                np.ones(points_transposed.shape[1], dtype=np.float32)
            ])
        else:
            points_with_intensity = points_transposed
    else:
        # 如果点云已经是(3, N)形状
        if points.shape[0] == 3:
            points_with_intensity = np.vstack([
                points, 
                np.ones(points.shape[1], dtype=np.float32)
            ])
        else:
            points_with_intensity = points

    # 创建ROS2 PointCloud2消息
    pc_msg = PointCloud2()
    pc_msg.header.frame_id = frame_id
    pc_msg.header.stamp = stamp
    pc_msg.fields = fields
    pc_msg.is_bigendian = False
    pc_msg.point_step = 16  # 4 个 float32 (x,y,z,intensity)
    pc_msg.row_step = pc_msg.point_step * points_with_intensity.shape[1]
    pc_msg.height = 1
    pc_msg.width = points_with_intensity.shape[1]
    pc_msg.is_dense = True

    # 转置回(N, 4)格式并转换为字节数组
    pc_msg.data = np.transpose(points_with_intensity).astype(np.float32).tobytes()

    publisher.publish(pc_msg)


def broadcast_tf(broadcaster, parent_frame, child_frame, translation, rotation, stamp):
    """广播TF变换"""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame

    t.transform.translation.x = float(translation[0])
    t.transform.translation.y = float(translation[1])
    t.transform.translation.z = float(translation[2])

    t.transform.rotation.x = float(rotation[0])
    t.transform.rotation.y = float(rotation[1])
    t.transform.rotation.z = float(rotation[2])
    t.transform.rotation.w = float(rotation[3])

    broadcaster.sendTransform(t)

class LidarVisualizer(Node):
    def __init__(self, args):
        super().__init__('mujoco_lidar_test')

        # 创建点云发布者
        self.pub_taichi = self.create_publisher(PointCloud2, '/lidar_points_taichi', 1)

        # 创建场景可视化标记发布者
        self.pub_scene = self.create_publisher(MarkerArray, '/mujoco_scene', 1)

        # 创建TF广播者
        self.tf_broadcaster = TransformBroadcaster(self)

        # 创建MuJoCo场景
        self.mj_model, self.mj_data = create_demo_scene()

        self.use_livox_lidar = False
        if args.lidar in {"avia", "mid40", "mid70", "mid360", "tele"}:
            self.livox_generator = LivoxGenerator(args.lidar)
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
            self.use_livox_lidar = True
        elif args.lidar == "HDL64":
            self.rays_theta, self.rays_phi = generate_HDL64()
        elif args.lidar == "vlp32":
            self.rays_theta, self.rays_phi = generate_vlp32()
        elif args.lidar == "os128":
            self.rays_theta, self.rays_phi = generate_os128()
        else:
            raise ValueError(f"不支持的LiDAR型号: {args.lidar}")

        # 优化内存布局
        self.rays_theta = np.ascontiguousarray(self.rays_theta)
        self.rays_phi = np.ascontiguousarray(self.rays_phi)

        # 创建激光雷达传感器
        self.lidar = MjLidarWrapper(self.mj_model, self.mj_data, site_name="lidar_site", args={"enable_profiling": args.profiling, "verbose": args.verbose})

        self.get_logger().info(f"射线数量: {len(self.rays_phi)}")

        # 获取激光雷达初始位置和方向
        lidar_base_position = self.mj_model.body("lidar_base").pos
        lidar_base_orientation = self.mj_model.body("lidar_base").quat[[1,2,3,0]]

        # 创建键盘监听器
        self.kb_listener = KeyboardListener(lidar_base_position, lidar_base_orientation)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MuJoCo LiDAR可视化与ROS2集成')
    parser.add_argument('--lidar', type=str, default='mid360', help='LiDAR型号 (mid360, HDL64, vlp32, os128)', \
                        choices=['avia', 'HAP', 'horizon', 'mid40', 'mid70', 'mid360', 'tele', 'HDL64', 'vlp32', 'os128'])
    parser.add_argument('--profiling', action='store_true', help='启用性能分析')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出信息')
    parser.add_argument('--rate', type=int, default=12, help='循环频率 (Hz) (默认: 12)')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MuJoCo LiDAR可视化与ROS2集成")
    print("=" * 60)
    print(f"配置：")
    print(f"- LiDAR型号: {args.lidar}")
    print(f"- 循环频率: {args.rate} Hz")
    print(f"- 性能分析: {'启用' if args.profiling else '禁用'}")
    print(f"- 详细输出: {'启用' if args.verbose else '禁用'}")

    forder_path = os.path.dirname(os.path.abspath(__file__))
    cmd = f"ros2 run rviz2 rviz2 -d {forder_path}/config/rviz2_config.rviz"
    print(f"在终端执行命令以开启rviz可视化:\n {cmd}")
    print("=" * 60)

    # 初始化ROS2
    rclpy.init()

    # 创建节点并运行
    node = LidarVisualizer(args)

    spin_thread = threading.Thread(target=lambda:rclpy.spin(node))
    spin_thread.start()

    # 创建定时器
    step_cnt = 0
    step_gap = 60 // args.rate
    rate = node.create_rate(60)

    try:
        with mujoco.viewer.launch_passive(node.mj_model, node.mj_data) as viewer:
            # 设置视图模式为site
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE.value
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE.value

            while rclpy.ok() and node.kb_listener.running and viewer.is_running:

                # 更新激光雷达位置和方向
                site_position, site_orientation = node.kb_listener.update_lidar_pose(1./60.)
                node.mj_model.body("lidar_base").pos[:] = site_position[:]
                node.mj_model.body("lidar_base").quat[:] = site_orientation[[3,0,1,2]]

                # 更新模拟
                mujoco.mj_step(node.mj_model, node.mj_data)
                step_cnt += 1
                viewer.sync()
                rate.sleep()

                if step_cnt % step_gap == 0:
                    # 发布场景可视化标记
                    publish_scene(node.pub_scene, node.lidar.scene, "world", node.get_clock().now().to_msg())

                    # 执行光线追踪
                    start_time = time.time()

                    if node.use_livox_lidar:
                        node.rays_theta, node.rays_phi = node.livox_generator.sample_ray_angles()

                    points = node.lidar.get_lidar_points(node.rays_phi, node.rays_theta, node.mj_data)
                    ti.sync()
                    end_time = time.time()

                    # 获取激光雷达位置和方向
                    lidar_position = node.lidar.sensor_position
                    lidar_orientation = Rotation.from_matrix(node.lidar.sensor_rotation).as_quat()

                    # 打印性能信息和当前位置
                    if args.verbose:
                        # 格式化欧拉角为度数
                        euler_deg = np.degrees(Rotation.from_quat(lidar_orientation).as_euler('xyz', degrees=True))
                        node.get_logger().info(f"位置: [{lidar_position[0]:.2f}, {lidar_position[1]:.2f}, {lidar_position[2]:.2f}], "
                            f"欧拉角: [{euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°], "
                            f"耗时: {(end_time - start_time)*1000:.2f} ms")

                        if args.profiling:
                            node.get_logger().info(f"  准备时间: {node.lidar.lidar_sensor.prepare_time:.2f}ms, 内核时间: {node.lidar.lidar_sensor.kernel_time:.2f}ms")

                    # 广播激光雷达的TF
                    broadcast_tf(node.tf_broadcaster, "world", "lidar", lidar_position, lidar_orientation, node.get_clock().now().to_msg())

                    # 发布点云
                    publish_point_cloud(node.pub_taichi, points, "lidar", node.get_clock().now().to_msg()
)

    except KeyboardInterrupt:
        print("用户中断，正在退出...")
    except Exception as e:
        print(f"发生错误: {e}")
        traceback.print_exc()
    finally:
        # 清理资源
        spin_thread.join()
        node.destroy_node()
        rclpy.shutdown()
        print("程序结束")

if __name__ == "__main__":
    main()