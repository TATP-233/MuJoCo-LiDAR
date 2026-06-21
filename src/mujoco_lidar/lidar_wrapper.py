from typing import Any

import mujoco
import numpy as np


class MjLidarWrapper:
    """
    MuJoCo LiDAR wrapper supporting CPU, Taichi, JAX, and Warp backends.

    Args:
        mj_model: MuJoCo model object
        site_name: Name of the LiDAR site in the model
        backend: 'cpu', 'taichi', 'jax', or 'warp' (default: 'taichi')
        cutoff_dist: Maximum ray distance in meters (default: 100.0)
        args: Backend-specific arguments (see docs/en/API.md)
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        site_name: str,
        backend: str = "taichi",
        cutoff_dist: float = 100.0,
        args: dict[str, Any] = None,
    ):
        if args is None:
            args = {}
        if backend == "gpu":
            backend = "taichi"
        self.backend = backend
        self.mj_model = mj_model
        self.cutoff_dist = cutoff_dist
        self.args = args

        if backend == "taichi":
            self._init_taichi_backend()
        elif backend == "warp":
            self._init_warp_backend()
        elif backend == "jax":
            self._init_jax_backend()
        elif backend == "cpu":
            self._init_cpu_backend()
        else:
            raise ValueError(
                f"Unsupported backend: {backend}, choose from 'cpu', 'taichi', 'jax', or 'warp'"
            )

        self.site_name = site_name
        self._sensor_pose = np.eye(4, dtype=np.float32)
        self._local_rays: np.ndarray | None = None
        self._distances: np.ndarray | None = None

    def _init_taichi_backend(self) -> None:
        """Initialize Taichi backend"""
        try:
            # Lazy import: only import when Taichi backend is selected
            import taichi as ti

            from mujoco_lidar.core_ti.mjlidar_ti import MjLidarTi

            # Initialize Taichi if not already done
            if not hasattr(ti, "_is_initialized") or not ti._is_initialized:
                ti.init(arch=ti.gpu, **self.args.get("ti_init_args", {}))

            # Create Taichi backend instance
            geomgroup = self.args.get("geomgroup", None)
            bodyexclude = self.args.get("bodyexclude", -1)
            max_candidates = self.args.get("max_candidates", 64)
            self._backend_instance = MjLidarTi(
                self.mj_model,
                cutoff_dist=self.cutoff_dist,
                geomgroup=geomgroup,
                bodyexclude=bodyexclude,
                max_candidates=max_candidates,
            )

        except ImportError as e:
            raise ImportError(
                f"Failed to import Taichi backend dependencies. "
                f'Please install taichi: uv add "mujoco-lidar[taichi]"\n'
                f"Error: {e}"
            ) from e

    def _init_warp_backend(self) -> None:
        """Initialize Warp backend"""
        try:
            from mujoco_lidar.core_warp.mjlidar_warp import MjLidarWarp

            geomgroup = self.args.get("geomgroup", None)
            bodyexclude = self.args.get("bodyexclude", -1)
            device = self.args.get("device", None)
            use_bvh = self.args.get("use_bvh", True)
            self._backend_instance = MjLidarWarp(
                self.mj_model,
                cutoff_dist=self.cutoff_dist,
                geomgroup=geomgroup,
                bodyexclude=bodyexclude,
                device=device,
                use_bvh=use_bvh,
            )

        except ImportError as e:
            raise ImportError(
                f"Failed to import Warp backend dependencies. "
                f'Please install warp: uv add "mujoco-lidar[warp]"\n'
                f"Error: {e}"
            ) from e

    def _init_jax_backend(self) -> None:
        """Initialize JAX backend"""
        try:
            from mujoco_lidar.core_jax.mjlidar_jax import MjLidarJax

            geomgroup = self.args.get("geomgroup", None)
            bodyexclude = self.args.get("bodyexclude", -1)

            # Pass mj_model directly. MjLidarJax will extract what it needs.
            self._backend_instance = MjLidarJax(
                self.mj_model,
                geom_ids=self.args.get("geom_ids"),
                geomgroup=geomgroup,
                bodyexclude=bodyexclude,
            )

        except ImportError as e:
            raise ImportError(f"Failed to import JAX backend dependencies.\nError: {e}") from e

    def _init_cpu_backend(self) -> None:
        """Initialize CPU backend"""
        try:
            from mujoco_lidar.core_cpu.mjlidar_cpu import MjLidarCPU

            geomgroup = self.args.get("geomgroup", None)
            bodyexclude = self.args.get("bodyexclude", -1)
            self._backend_instance = MjLidarCPU(
                self.mj_model,
                cutoff_dist=self.cutoff_dist,
                geomgroup=geomgroup,
                bodyexclude=bodyexclude,
            )

        except ImportError as e:
            raise ImportError(f"Failed to import CPU backend dependencies.\nError: {e}") from e

    @property
    def sensor_position(self) -> np.ndarray:
        return self._sensor_pose[:3, 3].copy()

    @property
    def sensor_rotation(self) -> np.ndarray:
        return self._sensor_pose[:3, :3].copy()

    def update_sensor_pose(self, mj_data: mujoco.MjData, site_name: str) -> None:
        # For CPU/Taichi/JAX/Warp backend, mj_data is mujoco.MjData
        if self.backend in ["cpu", "taichi", "jax", "warp"]:
            self._sensor_pose[:3, :3] = mj_data.site(site_name).xmat.reshape(3, 3)
            self._sensor_pose[:3, 3] = mj_data.site(site_name).xpos

    def trace_rays(
        self,
        mj_data: mujoco.MjData,
        ray_theta: np.ndarray,
        ray_phi: np.ndarray,
        site_name: str | None = None,
    ) -> np.ndarray:
        """
        Trace rays.
        For JAX backend, mj_data can be mujoco.MjData.
        """
        target_site = self.site_name if site_name is None else site_name

        if self.backend == "jax":
            # Update sensor pose for consistency
            self.update_sensor_pose(mj_data, target_site)

            # Use JITed trace_rays from backend instance
            # This handles ray generation, transformation and rendering in one JIT call
            self._distances, self._local_rays = self._backend_instance.trace_rays(
                mj_data.geom_xpos,
                mj_data.geom_xmat,
                mj_data.site(target_site).xpos,
                mj_data.site(target_site).xmat.reshape(3, 3),
                ray_theta,
                ray_phi,
            )

            return self._distances

        elif self.backend == "taichi":
            # Taichi Backend
            self.update_sensor_pose(mj_data, target_site)
            self._backend_instance.update(mj_data)

            import taichi as ti

            # Convert numpy arrays to Taichi ndarrays if necessary
            if isinstance(ray_theta, np.ndarray):
                theta_ti = ti.ndarray(dtype=ti.f32, shape=ray_theta.shape[0])
                theta_ti.from_numpy(ray_theta.astype(np.float32))
            else:
                theta_ti = ray_theta

            if isinstance(ray_phi, np.ndarray):
                phi_ti = ti.ndarray(dtype=ti.f32, shape=ray_phi.shape[0])
                phi_ti.from_numpy(ray_phi.astype(np.float32))
            else:
                phi_ti = ray_phi

            self._backend_instance.trace_rays(self._sensor_pose, theta_ti, phi_ti)
            return self._backend_instance.get_distances()

        elif self.backend == "warp":
            self.update_sensor_pose(mj_data, target_site)
            self._backend_instance.update(mj_data)
            self._backend_instance.trace_rays(self._sensor_pose, ray_theta, ray_phi)
            return self._backend_instance.get_distances()

        else:
            # CPU Backend
            self.update_sensor_pose(mj_data, target_site)
            self._backend_instance.update(mj_data)
            self._backend_instance.trace_rays(self._sensor_pose, ray_theta, ray_phi)
            return self._backend_instance.get_distances()

    def trace_rays_batch(
        self,
        geom_xpos: np.ndarray,
        geom_xmat: np.ndarray,
        sensor_pos: np.ndarray,
        sensor_rot: np.ndarray,
        ray_theta: np.ndarray,
        ray_phi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.backend in ["jax", "warp"]:
            distances, local_rays = self._backend_instance.trace_rays_batch(
                geom_xpos,
                geom_xmat,
                sensor_pos,
                sensor_rot,
                ray_theta,
                ray_phi,
            )
            return np.asarray(distances), np.asarray(local_rays)

        if self.backend == "taichi":
            distances, hit_points = self._backend_instance.trace_rays_batch(
                sensor_pos,
                sensor_rot,
                ray_theta,
                ray_phi,
            )
            return distances.to_numpy(), hit_points.to_numpy()

        raise NotImplementedError(f"Batch ray tracing is not supported by {self.backend} backend")

    def get_hit_points(self) -> np.ndarray:
        if self.backend == "jax":
            if self._distances is None or self._local_rays is None:
                return np.zeros((0, 3), dtype=np.float32)
            return np.asarray(np.maximum(self._distances, 0)[:, np.newaxis] * self._local_rays)
        return self._backend_instance.get_hit_points()

    def get_distances(self) -> np.ndarray:
        if self.backend == "jax":
            if self._distances is None:
                return np.zeros(0, dtype=np.float32)
            return np.asarray(self._distances)
        return self._backend_instance.get_distances()
