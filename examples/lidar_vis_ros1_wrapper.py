import argparse
import os
import signal
import subprocess
import time
import traceback

import geometry_msgs.msg
import mujoco
import mujoco.viewer
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray

from mujoco_lidar import MjLidarWrapper, scan_gen
from mujoco_lidar.mj_lidar_utils import KeyboardListener, create_demo_scene, create_marker_from_geom


def publish_scene(publisher, mj_scene, frame_id="world"):
    """将MuJoCo场景发布为ROS可视化标记数组"""
    marker_array = MarkerArray()

    # 记录当前使用的标记ID
    current_id = 0

    # 创建每个几何体的标记
    for i in range(mj_scene.ngeom):
        geom = mj_scene.geoms[i]
        # 现在 create_marker_from_geom 返回一个标记列表
        markers = create_marker_from_geom(geom, current_id, frame_id)

        # 添加所有返回的标记到标记数组
        for marker in markers:
            marker_array.markers.append(marker)
            current_id += 1

    # 发布标记数组
    publisher.publish(marker_array)


def publish_point_cloud(publisher, points, frame_id):
    """将点云数据发布为ROS PointCloud2消息"""
    stamp = rospy.Time.now()

    # 定义点云字段
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    # 添加强度值
    if len(points.shape) == 2:
        # 如果是(N, 3)形状，转换为(3, N)以便处理
        points_transposed = points.T if points.shape[1] == 3 else points

        if points_transposed.shape[0] == 3:
            # 添加强度通道
            points_with_intensity = np.vstack(
                [points_transposed, np.ones(points_transposed.shape[1], dtype=np.float32)]
            )
        else:
            points_with_intensity = points_transposed
    else:
        # 如果点云已经是(3, N)形状
        if points.shape[0] == 3:
            points_with_intensity = np.vstack([points, np.ones(points.shape[1], dtype=np.float32)])
        else:
            points_with_intensity = points

    # 转换为ROS消息格式的点云
    pc_msg = pc2.create_cloud(
        header=rospy.Header(frame_id=frame_id, stamp=stamp),
        fields=fields,
        points=np.transpose(points_with_intensity),  # 转置回(N, 4)格式
    )

    publisher.publish(pc_msg)


def broadcast_tf(broadcaster, parent_frame, child_frame, translation, rotation, stamp=None):
    """广播TF变换"""
    if stamp is None:
        stamp = rospy.Time.now()

    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame

    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]

    t.transform.rotation.x = rotation[0]
    t.transform.rotation.y = rotation[1]
    t.transform.rotation.z = rotation[2]
    t.transform.rotation.w = rotation[3]

    broadcaster.sendTransform(t)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MuJoCo LiDAR可视化与ROS集成")
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
        ],
    )
    parser.add_argument("--profiling", action="store_true", help="启用性能分析")
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
    print("MuJoCo LiDAR可视化与ROS集成")
    print("=" * 60)
    print("配置：")
    print(f"- LiDAR型号: {args.lidar}")
    print(f"- LiDAR后端: {args.backend}")
    print(f"- 循环频率: {args.rate} Hz")
    print(f"- 性能分析: {'启用' if args.profiling else '禁用'}")
    print(f"- 详细输出: {'启用' if args.verbose else '禁用'}")

    use_livox_lidar = False
    if args.lidar in {"avia", "mid40", "mid70", "mid360", "tele"}:
        livox_generator = scan_gen.LivoxGenerator(args.lidar)
        rays_theta, rays_phi = livox_generator.sample_ray_angles()
        use_livox_lidar = True
    elif args.lidar == "HDL64":
        rays_theta, rays_phi = scan_gen.generate_HDL64()
    elif args.lidar == "vlp32":
        rays_theta, rays_phi = scan_gen.generate_vlp32()
    elif args.lidar == "os128":
        rays_theta, rays_phi = scan_gen.generate_os128()
    else:
        raise ValueError(f"不支持的LiDAR型号: {args.lidar}")

    # 优化内存布局
    rays_theta = np.ascontiguousarray(rays_theta)
    rays_phi = np.ascontiguousarray(rays_phi)

    # 打印激光雷达的参数
    print(f"射线数量: {len(rays_theta)}")
    print("=" * 60)

    forder_path = os.path.dirname(os.path.abspath(__file__))
    cmd = f"rosrun rviz rviz -d {forder_path}/config/rviz_config.rviz"
    print(f"正在启动rviz可视化:\n {cmd}")
    print("=" * 60)

    # 启动 rviz 进程
    rviz_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

    # 初始化ROS节点
    rospy.init_node("mujoco_lidar_test", anonymous=True)

    # 创建点云发布者
    pub_taichi = rospy.Publisher("/lidar_points_taichi", PointCloud2, queue_size=1)

    # 创建场景可视化标记发布者
    pub_scene = rospy.Publisher("/mujoco_scene", MarkerArray, queue_size=1)

    # 创建TF广播者
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # 创建MuJoCo场景
    mj_model, mj_data = create_demo_scene("mesh_scene")

    # 创建场景对象
    scene = mujoco.MjvScene(mj_model, maxgeom=10000)

    # 创建激光雷达传感器
    lidar = MjLidarWrapper(mj_model, site_name="lidar_site", backend=args.backend)

    # 设置激光雷达位置
    lidar_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    lidar_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # 四元数(x,y,z,w)

    # 获取激光雷达初始位置和方向
    lidar_base_position = mj_model.body("lidar_base").pos
    lidar_base_orientation = mj_model.body("lidar_base").quat[[1, 2, 3, 0]]

    # 创建键盘监听器
    kb_listener = KeyboardListener(lidar_base_position, lidar_base_orientation)

    def update_scene():
        """更新MuJoCo场景"""
        mujoco.mjv_updateScene(
            mj_model,
            mj_data,
            mujoco.MjvOption(),
            None,
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_ALL.value,
            scene,
        )

    # 主循环
    rate = rospy.Rate(60)
    step_cnt = 0
    gap_step = 60 // args.rate
    last_time = time.time()

    try:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            # 设置视图模式为site
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE.value
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE.value

            while not rospy.is_shutdown() and kb_listener.running and viewer.is_running():
                # 更新mujoco中激光雷达site的位置和方向
                site_position, site_orientation = kb_listener.update_lidar_pose(1.0 / 60.0)
                mj_model.body("lidar_base").pos[:] = site_position[:]
                mj_model.body("lidar_base").quat[:] = site_orientation[[3, 0, 1, 2]]

                # 更新模拟
                mujoco.mj_step(mj_model, mj_data)
                step_cnt += 1
                viewer.sync()
                rate.sleep()

                if step_cnt % gap_step == 0:
                    update_scene()

                    # 发布场景可视化标记
                    publish_scene(pub_scene, scene)
                    # 执行光线追踪
                    start_time = time.time()

                    # 更新livox激光雷达的扫描角度
                    if use_livox_lidar:
                        rays_theta, rays_phi = livox_generator.sample_ray_angles()

                    # 更新激光雷达数据
                    lidar.trace_rays(mj_data, rays_theta, rays_phi)

                    # 获取激光雷达点云
                    points_local = lidar.get_hit_points()
                    end_time = time.time()

                    # 获取激光雷达位置和方向
                    lidar_position = lidar.sensor_position
                    lidar_orientation = Rotation.from_matrix(lidar.sensor_rotation).as_quat()

                    # 打印性能信息和当前位置
                    if args.verbose:
                        # 格式化欧拉角为度数
                        euler_deg = np.degrees(
                            Rotation.from_quat(lidar_orientation).as_euler("xyz", degrees=True)
                        )
                        print(
                            f"位置: [{lidar_position[0]:.2f}, {lidar_position[1]:.2f}, {lidar_position[2]:.2f}], "
                            f"欧拉角: [{euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°], "
                            f"耗时: {(end_time - start_time) * 1000:.2f} ms"
                        )

                        if args.profiling:
                            print(f"  射线追踪耗时: {(end_time - start_time) * 1000:.2f} ms")

                    # 广播激光雷达的TF
                    broadcast_tf(
                        tf_broadcaster, "world", "lidar", lidar_position, lidar_orientation
                    )

                    # 发布点云
                    publish_point_cloud(pub_taichi, points_local, "lidar")

    except rospy.ROSInterruptException:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        # 关闭 rviz 进程
        print("正在关闭 rviz 进程...")
        try:
            os.killpg(os.getpgid(rviz_process.pid), signal.SIGTERM)
            rviz_process.wait(timeout=5)
            print("rviz 进程已关闭")
        except Exception:
            print("强制关闭 rviz 进程...")
            os.killpg(os.getpgid(rviz_process.pid), signal.SIGKILL)
            print("rviz 进程已强制关闭")
        print("程序结束")
