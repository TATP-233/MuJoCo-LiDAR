import argparse
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as rt
from etils import epath

from mujoco_lidar import MjLidarWrapper, scan_gen

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"
_MJCF_PATH = _HERE.parent / "models" / "scene_g1.xml"

_JOINT_NUM = 29


class OnnxController:
    """ONNX controller for the G-1 robot."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        policy_path: str,
        default_angles: np.ndarray,
        ctrl_dt: float,
        n_substeps: int,
        action_scale: float = 0.5,
        lidar_type: str = "mid360",
        backend: str = "jax",
    ):

        self._output_names = ["continuous_actions"]
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

        self._action_scale = action_scale
        self._default_angles = default_angles
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)

        self._counter = 0
        self._n_substeps = n_substeps

        self._phase = np.array([0.0, np.pi])
        self._gait_freq = 1.5
        self._phase_dt = 2 * np.pi * self._gait_freq * ctrl_dt

        self._output_names = ["continuous_actions"]
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

        # lidar
        self.dynamic_lidar = False
        if lidar_type == "airy":
            self.rays_theta, self.rays_phi = scan_gen.generate_airy96()
        elif lidar_type == "mid360":
            self.livox_generator = scan_gen.LivoxGenerator(lidar_type)
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
            self.dynamic_lidar = True

        self.rays_theta = np.ascontiguousarray(self.rays_theta).astype(np.float32)
        self.rays_phi = np.ascontiguousarray(self.rays_phi).astype(np.float32)

        geomgroup = np.ones((mujoco.mjNGROUP,), dtype=np.ubyte)
        geomgroup[3:] = 0  # 排除group 1中的几何体
        self.lidar = MjLidarWrapper(
            mj_model,
            site_name="lidar",
            backend=backend,
            args={"bodyexclude": mj_model.body("torso_link").id, "geomgroup": geomgroup},
        )

    def get_obs(self, model, data) -> np.ndarray:
        linvel = data.sensor("local_linvel_pelvis").data
        gyro = data.sensor("gyro_pelvis").data
        imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_angles = data.qpos[7 : 7 + _JOINT_NUM] - self._default_angles
        joint_velocities = data.qvel[6 : 6 + _JOINT_NUM]
        phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)])
        command = np.zeros(3, dtype=np.float32)
        obs = np.hstack(
            [
                linvel,
                gyro,
                gravity,
                command,
                joint_angles,
                joint_velocities,
                self._last_action,
                phase,
            ]
        )
        return obs.astype(np.float32)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            onnx_input = {"obs": obs.reshape(1, -1)}
            onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
            self._last_action = onnx_pred.copy()
            data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles
            phase_tp1 = self._phase + self._phase_dt
            self._phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo LiDAR可视化与Unitree G1 ROS2集成")
    parser.add_argument(
        "--backend",
        type=str,
        default="jax",
        help="LiDAR后端 (cpu, taichi, jax, warp)",
        choices=["cpu", "taichi", "jax", "warp"],
    )
    parser.add_argument(
        "--lidar",
        type=str,
        default="mid360",
        help="LiDAR型号 (airy, mid360)",
        choices=["airy", "mid360"],
    )
    args = parser.parse_args()

    mj_model = mujoco.MjModel.from_xml_path(_MJCF_PATH.as_posix())
    mj_data = mujoco.MjData(mj_model)

    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

    ctrl_dt = 0.02
    lidar_dt = 1.0 / 10.0
    mj_model.opt.timestep = 0.004

    policy = OnnxController(
        mj_model,
        policy_path=(_ONNX_DIR / "g1_policy.onnx").as_posix(),
        default_angles=np.array(mj_model.keyframe("home").qpos[7 : 7 + _JOINT_NUM]),
        ctrl_dt=ctrl_dt,
        n_substeps=int(round(ctrl_dt / mj_model.opt.timestep)),
        action_scale=0.5,
        lidar_type=args.lidar,
        backend=args.backend,
    )

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.user_scn.ngeom = policy.rays_theta.shape[0]
        for i in range(viewer.user_scn.ngeom):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0, 0],
                pos=[0, 0, 0],
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 0.8]),
            )
        print("Starting simulation...")
        print("Number of rays:", policy.rays_theta.shape[0])

        # 创建颜色映射
        cmap = plt.get_cmap("hsv")  # 或使用 'jet', 'viridis', 'plasma' 等

        _last_time = 1e6
        n_substeps = int(round(lidar_dt / mj_model.opt.timestep))
        while viewer.is_running():
            if mj_data.time < _last_time:
                _counter = 0
                _start_time = time.time()
            _last_time = mj_data.time

            mujoco.mj_step(mj_model, mj_data)
            policy.get_control(mj_model, mj_data)

            _counter += 1
            if _counter % n_substeps == 0:
                if policy.dynamic_lidar:
                    policy.rays_theta, policy.rays_phi = policy.livox_generator.sample_ray_angles()
                policy.lidar.trace_rays(mj_data, policy.rays_theta, policy.rays_phi)
                points = policy.lidar.get_hit_points()
                world_points = (
                    points @ policy.lidar.sensor_rotation.T + policy.lidar.sensor_position
                )

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
