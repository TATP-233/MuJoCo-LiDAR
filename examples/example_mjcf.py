import argparse
import contextlib
import os
import queue
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
from etils import epath

matplotlib.use("TkAgg")  # 明确指定后端

# from mujoco_lidar import LidarSensor, generate_grid_scan_pattern
from mujoco_lidar import MjLidarWrapper, scan_gen

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 全局配置变量
lidar_sim_rate = 10
running = True
point_queue = queue.Queue(maxsize=5)  # 限制队列大小防止内存溢出


# MuJoCo仿真线程函数
def mujoco_simulation_thread(mj_model, mj_data, lidar_sensor, rays_theta, rays_phi):
    global running, point_queue
    lidar_sim_cnt = 0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # 设置视图模式为site
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE.value
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE.value
        viewer.cam.distance = 5.0

        try:
            while viewer.is_running() and running:
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
                time.sleep(1.0 / 60.0)

                if mj_data.time * lidar_sim_rate > lidar_sim_cnt:
                    # 更新激光雷达位置
                    lidar_sensor.trace_rays(mj_data, rays_theta, rays_phi)

                    # 执行光线投射
                    points = lidar_sensor.get_hit_points()
                    # 将点云数据放入队列，如果队列满了就清空后再放入
                    try:
                        point_queue.put_nowait(points.copy())
                    except queue.Full:
                        # 清空队列，保持最新数据
                        while not point_queue.empty():
                            try:
                                point_queue.get_nowait()
                            except queue.Empty:
                                break
                        with contextlib.suppress(queue.Full):
                            point_queue.put_nowait(points.copy())

                    lidar_sim_cnt += 1
        except KeyboardInterrupt:
            print("接收到键盘中断信号，正在退出...")
        finally:
            print("MuJoCo Viewer已关闭，正在退出...")
            running = False


# 主程序（matplotlib在主线程中运行）
def main():
    global running
    parser = argparse.ArgumentParser(description="MuJoCo LiDAR MJCF example")
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu",
        help="LiDAR后端 (cpu, taichi, jax, warp)",
        choices=["cpu", "taichi", "jax", "warp"],
    )
    args = parser.parse_args()

    # print help
    print("=" * 100)
    print("在Mujoco Viewer视图，双击选中MoCap物体（带坐标系的红色透明方块 lidar_site）")
    print("选中后，按住ctrl，按下鼠标右键拖动平移视角")
    print("按住ctrl，按下鼠标左键拖动旋转视角")
    print("关闭任意一个窗口都会退出整个程序")
    print("=" * 100)

    # 从文件加载MuJoCo模型
    mjcf_file = epath.Path(__file__).parent.parent / "models" / "demo.xml"
    mj_model = mujoco.MjModel.from_xml_path(mjcf_file.as_posix())
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    # 生成网格扫描模式
    rays_theta, rays_phi = scan_gen.generate_grid_scan_pattern(num_ray_cols=64, num_ray_rows=16)

    exclode_body_id = mj_model.body("your_robot_name").id
    print("exclude body id:", exclode_body_id)

    # 创建激光雷达传感器
    lidar_sensor = MjLidarWrapper(
        mj_model, site_name="lidar_site", backend=args.backend, args={"bodyexclude": exclode_body_id}
    )
    lidar_sensor.trace_rays(mj_data, rays_theta, rays_phi)
    points = lidar_sensor.get_hit_points()

    # 启动MuJoCo仿真线程
    sim_thread = threading.Thread(
        target=mujoco_simulation_thread,
        args=(mj_model, mj_data, lidar_sensor, rays_theta, rays_phi),
    )
    sim_thread.daemon = True
    sim_thread.start()

    # 在主线程中运行matplotlib
    plt.ion()  # 开启交互模式
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 0.3])  # 设置三个轴的比例尺相同

    try:
        while running:
            # 检查matplotlib窗口是否被关闭
            if not plt.get_fignums():
                print("Matplotlib窗口已关闭，正在退出...")
                running = False
                break

            # 从队列中获取最新的点云数据
            try:
                while not point_queue.empty():
                    points = point_queue.get_nowait()
            except queue.Empty:
                pass

            # 更新绘图
            ax.cla()  # 清除当前坐标轴
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="viridis", s=3
            )
            ax.set_title("LiDAR Point Cloud (Real-time)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            plt.draw()  # 更新绘图
            plt.pause(1.0 / lidar_sim_rate)  # 暂停以更新图形

    except KeyboardInterrupt:
        print("接收到键盘中断信号，正在退出...")
    except Exception as e:
        print(f"绘图过程中出错: {e}")
    finally:
        running = False
        plt.close("all")  # 确保关闭所有matplotlib窗口

    # 等待仿真线程结束
    sim_thread.join(timeout=3.0)
    if sim_thread.is_alive():
        print("仿真线程未能在超时时间内结束")

    print("程序已退出")


if __name__ == "__main__":
    main()
