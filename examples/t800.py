import argparse
import time

import matplotlib.pyplot as plt
import MNN
import mujoco
import mujoco.viewer
import numpy as np
from etils import epath

from mujoco_lidar import MjLidarWrapper, scan_gen

_HERE = epath.Path(__file__).parent.absolute()
_MNN_DIR = _HERE / "mnn"
_MJCF_PATH = _HERE.parent / "models" / "scene_t800.xml"

_JOINT_NUM = 25
_ACTION_NUM = 22
_OBS_NUM = 72
_OBS_STEPS = 15

# Joints controlled by the RL policy (J00-J11 legs, J13-J22 arms)
_ACTIVE_JOINT_IDX = np.array([*range(12), *range(13, 23)])

# From engineai_robotics_native_sdk assets/config/t800/rl_walking_example/default.yaml
_DEFAULT_Q = np.array(
    [
        -0.06, 0.0, 0.0, 0.12, -0.06, 0.0,  # left leg
        -0.06, 0.0, 0.0, 0.12, -0.06, 0.0,  # right leg
        0.0,  # torso
        0.0, 0.15, 0.0, -0.25, 0.0,  # left arm
        0.0, -0.15, 0.0, -0.25, 0.0,  # right arm
        0.0, 0.0,  # head
    ]
)
_ACTION_SCALE = np.array(
    [
        0.5, 0.2, 0.2, 0.5, 0.5, 0.2,  # left leg
        0.5, 0.2, 0.2, 0.5, 0.5, 0.2,  # right leg
        0.2, 0.2, 0.05, 0.2, 0.05,  # left arm
        0.2, 0.2, 0.05, 0.2, 0.05,  # right arm
    ]
)
_COMMAND_OBS_SCALE = np.array([2.0, 2.0, 1.0], dtype=np.float32)


class MnnController:
    """MNN walking controller for the T800 robot."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        policy_path: str,
        n_substeps: int,
        lidar_type: str = "mid360",
        backend: str = "warp",
    ):
        self._interpreter = MNN.Interpreter(policy_path)
        self._session = self._interpreter.createSession({"numThread": 1})
        self._input_tensor = self._interpreter.getSessionInput(self._session, "input")

        self._last_action = np.zeros(_ACTION_NUM, dtype=np.float32)
        self._obs_history = np.zeros((_OBS_NUM, _OBS_STEPS), dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)  # vx, vy, yaw_rate

        self._counter = 0
        self._n_substeps = n_substeps
        self._is_first = True

        # observation scales: dof_pos=1.0, dof_vel=0.05, last_action=1.0,
        # angular_vel=1.0, gravity=1.0
        self._obs_scale = np.ones(_OBS_NUM, dtype=np.float32)
        self._obs_scale[_ACTION_NUM : 2 * _ACTION_NUM] = 0.05

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
        geomgroup[3:] = 0  # 排除group 3\4\5 中的碰撞几何体
        self.lidar = MjLidarWrapper(
            mj_model,
            site_name="lidar",
            backend=backend,
            args={"bodyexclude": mj_model.body("LINK_HEAD_YAW").id, "geomgroup": geomgroup},
        )

    def get_obs(self, model, data) -> np.ndarray:
        gyro = data.sensor("imu_angular_velocity").data
        imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_angles = (data.qpos[7 : 7 + _JOINT_NUM] - _DEFAULT_Q)[_ACTIVE_JOINT_IDX]
        joint_velocities = data.qvel[6 : 6 + _JOINT_NUM][_ACTIVE_JOINT_IDX]
        obs = np.hstack(
            [
                joint_angles,
                joint_velocities,
                self._last_action,
                gyro,
                gravity,
            ]
        )
        obs *= self._obs_scale
        return np.clip(obs, -100.0, 100.0).astype(np.float32)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            if self._is_first:
                self._is_first = False
                self._obs_history[:] = obs[:, None]
            self._obs_history[:, :-1] = self._obs_history[:, 1:]
            self._obs_history[:, -1] = obs
            # policy input: [obs_step_0 ... obs_step_N, command(3)]
            mnn_input = np.concatenate(
                [self._obs_history.T.flatten(), self.command * _COMMAND_OBS_SCALE]
            ).reshape(1, -1)
            tmp_in = MNN.Tensor(
                (1, _OBS_NUM * _OBS_STEPS + 3),
                MNN.Halide_Type_Float,
                mnn_input,
                MNN.Tensor_DimensionType_Caffe,
            )
            self._input_tensor.copyFrom(tmp_in)
            self._interpreter.runSession(self._session)
            out = self._interpreter.getSessionOutput(self._session, "output")
            host = MNN.Tensor(
                (1, _ACTION_NUM),
                MNN.Halide_Type_Float,
                np.zeros((1, _ACTION_NUM), dtype=np.float32),
                MNN.Tensor_DimensionType_Caffe,
            )
            out.copyToHostTensor(host)
            action = host.getNumpyData()[0]
            self._last_action = action.copy()
            q_des = _DEFAULT_Q.copy()
            q_des[_ACTIVE_JOINT_IDX] += action * _ACTION_SCALE
            data.ctrl[:] = q_des


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo LiDAR可视化与T800 ROS2集成")
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu",
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
    parser.add_argument(
        "--walk",
        action="store_true",
        help="行走模式：前进3秒、后退3秒循环",
    )
    args = parser.parse_args()

    mj_model = mujoco.MjModel.from_xml_path(_MJCF_PATH.as_posix())
    mj_data = mujoco.MjData(mj_model)

    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

    ctrl_dt = 0.01
    lidar_dt = 1.0 / 10.0

    policy = MnnController(
        mj_model,
        policy_path=(_MNN_DIR / "t800_policy.mnn").as_posix(),
        n_substeps=int(round(ctrl_dt / mj_model.opt.timestep)),
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
            if args.walk:
                if (mj_data.time % 10.0) < 5.0:
                    policy.command[0] = 0.5
                    policy.command[2] = 0
                else:
                    policy.command[0] = 0
                    policy.command[2] = np.pi / 4
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
