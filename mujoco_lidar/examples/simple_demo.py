import time
import threading

import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from mujoco_lidar.lidar_wrapper import MjLidarWrapper
from mujoco_lidar.scan_gen import generate_grid_scan_pattern

simple_demo_scene = """
<mujoco model="simple_demo">
    <worldbody>
        <!-- 地面+四面墙 -->
        <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0" rgba="0.2 0.9 0.9 1"/>
        <geom name="wall1" type="box" size="1e-3 3 1" pos=" 3 0 1" rgba="0.9 0.9 0.9 1"/>
        <geom name="wall2" type="box" size="1e-3 3 1" pos="-3 0 1" rgba="0.9 0.9 0.9 1"/>
        <geom name="wall3" type="box" size="3 1e-3 1" pos="0  3 1" rgba="0.9 0.9 0.9 1"/>
        <geom name="wall4" type="box" size="3 1e-3 1" pos="0 -3 1" rgba="0.9 0.9 0.9 1"/>

        <!-- 盒子 -->
        <geom name="box1" type="box" size="0.5 0.5 0.5" pos="2 0 0.5" euler="45 -45 0" rgba="1 0 0 1"/>

        <!-- 球体 -->
        <geom name="sphere1" type="sphere" size="0.5" pos="0 2 0.5" rgba="0 1 0 1"/>
        
        <!-- 圆柱体 -->
        <geom name="cylinder1" type="cylinder" size="0.4 0.6" pos="0 -2 0.4" euler="0 90 0" rgba="0 0 1 1"/>

        <!-- 椭球体 -->
        <geom name="ellipsoid1" type="ellipsoid" size="0.4 0.3 0.5" pos="2 2 0.5" rgba="1 1 0 1"/>

        <!-- 胶囊体 -->
        <geom name="capsule1" type="capsule" size="0.3 0.5" pos="-1 1 0.8" euler="45 0 0" rgba="1 0 1 1"/>
        
        <!-- 激光雷达 -->
        <body name="your_robot_name" pos="0 0 1" quat="1 0 0 0" mocap="true">
            <inertial pos="0 0 0" mass="1e-4" diaginertia="1e-9 1e-9 1e-9"/>
            <site name="lidar_site" size="0.001" type='sphere'/>
            <geom type="box" size="0.1 0.1 0.1" density="0" contype="0" conaffinity="0" rgba="0.9 0.3 0.3 0.2"/>
        </body>
        
        </worldbody>
</mujoco>
"""

# 创建MuJoCo模型
mj_model = mujoco.MjModel.from_xml_string(simple_demo_scene)    
mj_data = mujoco.MjData(mj_model)

# 生成网格扫描模式
rays_theta, rays_phi = generate_grid_scan_pattern(num_ray_cols=64, num_ray_rows=16)

# 创建激光雷达传感器
lidar_sim = MjLidarWrapper(mj_model, mj_data, site_name="lidar_site")
points = lidar_sim.get_lidar_points(rays_phi, rays_theta, mj_data)

lidar_sim_rate = 10
lidar_sim_cnt = 0

def plot_points_thread():
    global points, lidar_sim_rate
    plt.ion()  # 开启交互模式
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 0.3])  # 设置三个轴的比例尺相同

    while True:
        ax.cla()  # 清除当前坐标轴
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=3)
        plt.draw()  # 更新绘图
        plt.pause(1./lidar_sim_rate)  # 暂停以更新图形

plot_points_thread = threading.Thread(target=plot_points_thread)
plot_points_thread.start()

# 主循环
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    # 设置视图模式为site
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE.value
    viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE.value
    viewer.cam.distance = 5.

    while viewer.is_running:
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()
        time.sleep(1./60.)

        if mj_data.time * lidar_sim_rate > lidar_sim_cnt:

            # 更新激光雷达位置
            lidar_sim.update_scene(mj_model, mj_data)

            # 执行光线追踪
            points = lidar_sim.get_lidar_points(rays_phi, rays_theta, mj_data)
            if lidar_sim_cnt == 0:
                print("points basic info:")
                print("  .shape:", points.shape)
                print("  .dtype:", points.dtype)
                print("  x.min():", points[:, 0].min(), "x.max():", points[:, 0].max())
                print("  y.min():", points[:, 1].min(), "y.max():", points[:, 1].max())
                print("  z.min():", points[:, 2].min(), "z.max():", points[:, 2].max())

            lidar_sim_cnt += 1

plot_points_thread.join()
