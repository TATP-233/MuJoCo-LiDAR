import time
import mujoco
import numpy as np
import taichi as ti

from mujoco_lidar.core import MjLidarSensor

class MjLidarWrapper:
    def __init__(self, mj_model:mujoco.MjModel, mj_data:mujoco.MjData, site_name:str, args:dict={}):
        self.scene = mujoco.MjvScene(mj_model, maxgeom=100)
        mujoco.mj_forward(mj_model, mj_data)
        mujoco.mjv_updateScene(
            mj_model, mj_data, mujoco.MjvOption(), 
            None, mujoco.MjvCamera(), 
            mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)

        self.lidar_sensor = MjLidarSensor(self.scene, enable_profiling=args.get("enable_profiling", False), verbose=args.get("verbose", False))
        for i in range(self.lidar_sensor.n_geoms):
            if mj_model.geom(self.scene.geoms[i].objid).contype[0] == 0:
                self.lidar_sensor.geom_types[i] = -1

        self.site_name = site_name
        self.sensor_pose = np.eye(4, dtype=np.float32)
        self.update_sensor_pose(mj_data, site_name)

    @property
    def sensor_position(self):
        return self.sensor_pose[:3,3].copy()

    @property
    def sensor_rotation(self):
        return self.sensor_pose[:3,:3].copy()

    def update_scene(self, mj_model:mujoco.MjModel, mj_data:mujoco.MjData):
        self.start_total = time.time() if self.lidar_sensor.enable_profiling else 0
        mujoco.mj_forward(mj_model, mj_data)
        mujoco.mjv_updateScene(
            mj_model, mj_data, mujoco.MjvOption(), 
            None, mujoco.MjvCamera(), 
            mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)

        # 更新几何体位置
        self.lidar_sensor.update_geom_positions(self.scene)

    def update_sensor_pose(self, mj_data:mujoco.MjData, site_name:str):
        self.sensor_pose[:3,:3] = mj_data.site(site_name).xmat.reshape(3,3)
        self.sensor_pose[:3,3] = mj_data.site(site_name).xpos

    def get_lidar_points(self, rays_phi, rays_theta, mj_data:mujoco.MjData, site_name:str=None):
        if site_name is None:
            self.update_sensor_pose(mj_data, self.site_name)
        else:
            self.update_sensor_pose(mj_data, site_name)
        
        assert rays_phi.shape == rays_theta.shape, "rays_phi和rays_theta的形状必须相同"
        n_rays = rays_phi.shape[0]

        if self.lidar_sensor.cached_n_rays != n_rays:
            self.lidar_sensor.rays_phi_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
            self.lidar_sensor.rays_theta_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
            self.lidar_sensor.hit_points = ti.Vector.field(3, dtype=ti.f32, shape=n_rays)
            # 同时创建临时字段
            self.lidar_sensor.hit_points_world = ti.Vector.field(3, dtype=ti.f32, shape=n_rays)
            self.lidar_sensor.hit_mask = ti.field(dtype=ti.i32, shape=n_rays)
            self.lidar_sensor.cached_n_rays = n_rays

        self.lidar_sensor.sensor_pose_ti.from_numpy(self.sensor_pose)
        self.lidar_sensor.rays_phi_ti.from_numpy(rays_phi.astype(np.float32))
        self.lidar_sensor.rays_theta_ti.from_numpy(rays_theta.astype(np.float32))

        # 准备阶段结束，记录时间
        prepare_end = time.time() if self.lidar_sensor.enable_profiling else 0
        self.lidar_sensor.prepare_time = (prepare_end - self.start_total) * 1000 if self.lidar_sensor.enable_profiling else 0

        kernel_start = time.time() if self.lidar_sensor.enable_profiling else 0
        self.lidar_sensor.trace_rays(
            self.lidar_sensor.sensor_pose_ti,
            self.lidar_sensor.rays_phi_ti,
            self.lidar_sensor.rays_theta_ti,
            n_rays,
            self.lidar_sensor.hit_points
        )
        ti.sync()
        kernel_end = time.time() if self.lidar_sensor.enable_profiling else 0
        self.lidar_sensor.kernel_time = (kernel_end - kernel_start) * 1000 if self.lidar_sensor.enable_profiling else 0

        points_local = self.lidar_sensor.hit_points.to_numpy()
        return points_local

