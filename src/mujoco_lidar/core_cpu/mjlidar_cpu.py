import mujoco
import numpy as np


class MjLidarCPU:
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        cutoff_dist: float = 100.0,
        geomgroup: np.ndarray | None = None,
        bodyexclude: int = -1,
    ) -> None:

        self.mj_model = mj_model
        self.cutoff_dist = cutoff_dist
        self.geomgroup = geomgroup
        self.bodyexclude = bodyexclude

        self._dist: np.ndarray | None = None
        self._hit_points: np.ndarray | None = None

    def update(self, mj_data: mujoco.MjData) -> None:
        self.mj_data = mj_data

    def trace_rays(self, pose_4x4: np.ndarray, ray_theta: np.ndarray, ray_phi: np.ndarray) -> None:

        if ray_phi.shape[0] != ray_theta.shape[0]:
            raise ValueError("ray_phi and ray_theta must have the same shape")

        _nray = ray_phi.shape[0]

        # Initialize
        self._dist = np.full(_nray, self.cutoff_dist, dtype=np.float64)
        _geomid = np.full(_nray, 0, dtype=np.int32)

        # Uniformly generate vec from site's pose and lidar settings
        # Note that all the vec are in the local frame.
        site_pos, site_mat = pose_4x4[:3, 3], pose_4x4[:3, :3]
        pnt = np.array([site_pos]).T
        x = np.cos(ray_phi) * np.cos(ray_theta)
        y = np.cos(ray_phi) * np.sin(ray_theta)
        z = np.sin(ray_phi)
        local_vecs = np.stack((x, y, z), axis=-1)
        world_vecs = local_vecs @ site_mat.T
        world_vecs /= np.linalg.norm(world_vecs, axis=1, keepdims=True)
        world_vecs_flat = world_vecs.flatten()

        # Get the ray casting results
        mujoco.mj_multiRay(
            m=self.mj_model,
            d=self.mj_data,
            pnt=pnt,
            vec=world_vecs_flat,
            geomgroup=self.geomgroup,
            flg_static=1,
            bodyexclude=self.bodyexclude,
            geomid=_geomid,
            dist=self._dist,
            normal=None,
            nray=_nray,
            cutoff=self.cutoff_dist,
        )
        # Calculate the point's position in local frame from vec + dist
        self._dist[_geomid == -1] = -1

        # Update the pcl frame with local frame data
        self._hit_points = local_vecs * np.maximum(self._dist, 0)[:, np.newaxis]

    def get_hit_points(self) -> np.ndarray | None:
        return self._hit_points

    def get_distances(self) -> np.ndarray | None:
        return self._dist
