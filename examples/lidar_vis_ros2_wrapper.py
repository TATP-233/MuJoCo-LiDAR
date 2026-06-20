import argparse
import os
import signal
import subprocess
import threading
import time
import traceback

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from lidar_vis_ros2 import broadcast_tf, publish_point_cloud, publish_scene
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import MarkerArray

from mujoco_lidar import MjLidarWrapper, scan_gen
from mujoco_lidar.mj_lidar_utils import KeyboardListener, create_demo_scene


class LidarVisualizer(Node):
    def __init__(self, args):
        super().__init__("mujoco_lidar_test")
        self.site_name = "lidar_site"

        # 创建点云发布者
        self.pub_taichi = self.create_publisher(PointCloud2, "/lidar_points_taichi", 1)

        # 创建场景可视化标记发布者
        self.pub_scene = self.create_publisher(MarkerArray, "/mujoco_scene", 1)

        # 创建TF广播者
        self.tf_broadcaster = TransformBroadcaster(self)

        # 创建MuJoCo场景
        self.mj_model, self.mj_data = create_demo_scene("primitive")

        self.scene = mujoco.MjvScene(self.mj_model, maxgeom=10000)

        self.use_livox_lidar = False
        if args.lidar in {"avia", "mid40", "mid70", "mid360", "tele"}:
            self.livox_generator = scan_gen.LivoxGenerator(args.lidar)
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
            self.use_livox_lidar = True
        elif args.lidar == "HDL64":
            self.rays_theta, self.rays_phi = scan_gen.generate_HDL64()
        elif args.lidar == "vlp32":
            self.rays_theta, self.rays_phi = scan_gen.generate_vlp32()
        elif args.lidar == "os128":
            self.rays_theta, self.rays_phi = scan_gen.generate_os128()
        elif args.lidar == "custom":
            self.rays_theta, self.rays_phi = scan_gen.generate_grid_scan_pattern(
                360, 64, phi_range=(0.0, np.pi / 2.0)
            )
        else:
            raise ValueError(f"不支持的LiDAR型号: {args.lidar}")

        # 优化内存布局
        self.rays_theta = np.ascontiguousarray(self.rays_theta).astype(np.float32)
        self.rays_phi = np.ascontiguousarray(self.rays_phi).astype(np.float32)

        self.lidar = MjLidarWrapper(self.mj_model, site_name=self.site_name, backend=args.backend)

        n_rays = len(self.rays_theta)

        self.get_logger().info(f"射线数量: {n_rays}")

        # 获取激光雷达初始位置和方向
        lidar_base_position = self.mj_model.body("lidar_base").pos
        lidar_base_orientation = self.mj_model.body("lidar_base").quat[[1, 2, 3, 0]]

        # 创建键盘监听器
        self.kb_listener = KeyboardListener(lidar_base_position, lidar_base_orientation)

    def update_scene(self):
        mujoco.mjv_updateScene(
            self.mj_model,
            self.mj_data,
            mujoco.MjvOption(),
            None,
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MuJoCo LiDAR可视化与ROS2集成")
    parser.add_argument(
        "--lidar",
        type=str,
        default="mid360",
        help="LiDAR型号 (mid360, HDL64, vlp32, os128)",
        choices=[
            "avia",
            "HAP",
            "horizon",
            "mid40",
            "mid70",
            "mid360",
            "tele",
            "HDL64",
            "vlp32",
            "os128",
            "custom",
        ],
    )
    parser.add_argument("--verbose", action="store_true", help="显示详细输出信息")
    parser.add_argument("--rate", type=int, default=12, help="循环频率 (Hz) (默认: 12)")
    parser.add_argument(
        "--backend",
        type=str,
        default="taichi",
        help="LiDAR后端 (cpu, taichi, jax, warp)",
        choices=["cpu", "taichi", "jax", "warp"],
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MuJoCo LiDAR可视化与ROS2集成")
    print("=" * 60)
    print("配置：")
    print(f"- LiDAR型号: {args.lidar}")
    print(f"- LiDAR后端: {args.backend}")
    print(f"- 循环频率: {args.rate} Hz")
    print(f"- 详细输出: {'启用' if args.verbose else '禁用'}")

    forder_path = os.path.dirname(os.path.abspath(__file__))
    cmd = f"ros2 run rviz2 rviz2 -d {forder_path}/config/rviz2_config.rviz"
    print(f"正在启动rviz2可视化:\n {cmd}")
    print("=" * 60)

    # 启动 rviz2 进程
    rviz_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

    # 初始化ROS2
    rclpy.init()

    # 创建节点并运行
    node = LidarVisualizer(args)

    spin_thread = threading.Thread(target=lambda: rclpy.spin(node))
    spin_thread.start()

    # 创建定时器
    step_cnt = 0
    render_fps = 60
    step_gap = render_fps // args.rate
    rate = node.create_rate(render_fps)

    try:
        with mujoco.viewer.launch_passive(node.mj_model, node.mj_data) as viewer:
            # 设置视图模式为site
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE.value
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE.value

            lidar_pose = np.eye(4, dtype=np.float32)
            while rclpy.ok() and node.kb_listener.running and viewer.is_running():
                # 更新激光雷达位置和方向
                site_position, site_orientation = node.kb_listener.update_lidar_pose(1.0 / 60.0)
                node.mj_model.body("lidar_base").pos[:] = site_position[:]
                node.mj_model.body("lidar_base").quat[:] = site_orientation[[3, 0, 1, 2]]

                # 更新模拟
                for _ in range(int(1.0 / (render_fps * node.mj_model.opt.timestep))):
                    mujoco.mj_step(node.mj_model, node.mj_data)
                step_cnt += 1
                viewer.sync()
                rate.sleep()

                if step_cnt % step_gap == 0:
                    node.update_scene()

                    # 发布场景可视化标记
                    publish_scene(
                        node.pub_scene, node.scene, "world", node.get_clock().now().to_msg()
                    )

                    if node.use_livox_lidar:
                        node.rays_theta, node.rays_phi = node.livox_generator.sample_ray_angles()

                    # 获取激光雷达位姿
                    lidar_pose[:3, 3] = node.mj_data.site(node.site_name).xpos
                    lidar_pose[:3, :3] = node.mj_data.site(node.site_name).xmat.reshape(3, 3)

                    start_time = time.time()
                    node.lidar.trace_rays(node.mj_data, node.rays_theta, node.rays_phi)
                    end_time = time.time()

                    points_local = node.lidar.get_hit_points()

                    # 获取激光雷达位置和方向
                    lidar_position = lidar_pose[:3, 3]
                    lidar_orientation = Rotation.from_matrix(lidar_pose[:3, :3]).as_quat()

                    # 广播激光雷达的TF
                    broadcast_tf(
                        node.tf_broadcaster,
                        "world",
                        "lidar",
                        lidar_position,
                        lidar_orientation,
                        node.get_clock().now().to_msg(),
                    )

                    # 发布点云
                    publish_point_cloud(
                        node.pub_taichi, points_local, "lidar", node.get_clock().now().to_msg()
                    )

                    # 打印性能信息和当前位置
                    if args.verbose:
                        # 格式化欧拉角为度数
                        euler_deg = Rotation.from_quat(lidar_orientation).as_euler(
                            "xyz", degrees=True
                        )
                        node.get_logger().info(
                            f"位置: [{lidar_position[0]:.2f}, {lidar_position[1]:.2f}, {lidar_position[2]:.2f}], "
                            f"欧拉角: [{euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°], "
                            f"耗时: {(end_time - start_time) * 1000:.2f} ms"
                        )

    except KeyboardInterrupt:
        print("用户中断，正在退出...")
    except Exception as e:
        print(f"发生错误: {e}")
        traceback.print_exc()
    finally:
        os.system("stty echo")  # 恢复终端回显
        if hasattr(node, "kb_listener"):
            del node.kb_listener
            print("键盘监听器已清理")
        # 关闭 rviz2 进程
        print("正在关闭 rviz2 进程...")
        try:
            os.killpg(os.getpgid(rviz_process.pid), signal.SIGTERM)
            rviz_process.wait(timeout=5)
            print("rviz2 进程已关闭")
        except Exception:
            print("强制关闭 rviz2 进程...")
            os.killpg(os.getpgid(rviz_process.pid), signal.SIGKILL)
            print("rviz2 进程已强制关闭")
        # 清理资源
        spin_thread.join()
        node.destroy_node()
        rclpy.shutdown()
        print("程序结束")


if __name__ == "__main__":
    main()
