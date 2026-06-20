import argparse
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
from etils import epath

from mujoco_lidar import MjLidarWrapper, scan_gen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo LiDAR native viewer example")
    parser.add_argument(
        "--backend",
        type=str,
        default="taichi",
        help="LiDAR后端 (cpu, taichi, jax, warp)",
        choices=["cpu", "taichi", "jax", "warp"],
    )
    args = parser.parse_args()

    mjcf_file = epath.Path(__file__).parent.parent / "models" / "demo.xml"
    mj_model = mujoco.MjModel.from_xml_path(mjcf_file.as_posix())
    mj_data = mujoco.MjData(mj_model)

    update_rate = 12.0  # Hz
    n_substeps = int(round(1.0 / (mj_model.opt.timestep * update_rate)))
    print(f"n_substeps = {n_substeps}")

    lidar = MjLidarWrapper(
        mj_model,
        "lidar_site",
        backend=args.backend,
        args={"bodyexclude": mj_model.body("your_robot_name").id},
    )
    livox_generator = scan_gen.LivoxGenerator("mid360")
    rays_theta, rays_phi = livox_generator.sample_ray_angles()

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.user_scn.ngeom = rays_theta.shape[0]
        for i in range(viewer.user_scn.ngeom):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0],
                pos=[0, 0, 0],
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 0.8]),
            )

        print("Starting simulation...")
        print("Number of rays:", rays_theta.shape[0])
        cmap = plt.get_cmap("hsv")  # 或使用 'jet', 'viridis', 'plasma' 等

        _last_time = 1e6
        while viewer.is_running():
            mujoco.mj_step(mj_model, mj_data)

            if mj_data.time < _last_time:
                _counter = 0
                _start_time = time.time()
            _last_time = mj_data.time

            _counter += 1
            if _counter % n_substeps == 0:
                rays_theta, rays_phi = livox_generator.sample_ray_angles()
                lidar.trace_rays(mj_data, rays_theta, rays_phi)
                points = lidar.get_hit_points()
                world_points = points @ lidar.sensor_rotation.T + lidar.sensor_position

                # 根据高度设置颜色
                z_values = world_points[:, 2]
                z_min, z_max = z_values.min(), z_values.max()
                if z_max > z_min:
                    # 归一化高度值到 [0, 1]
                    z_norm = (z_values - z_min) / (z_max - z_min)
                else:
                    z_norm = np.zeros_like(z_values)

                # 使用 matplotlib 颜色映射
                colors = cmap(z_norm)  # 返回 RGBA 值，shape: (N, 4)

                for i in range(viewer.user_scn.ngeom):
                    viewer.user_scn.geoms[i].pos[:] = world_points[i]
                    viewer.user_scn.geoms[i].rgba[:] = colors[i]

            viewer.sync()
            run_time = time.time() - _start_time
            if run_time < mj_data.time:
                time.sleep(mj_data.time - run_time)
