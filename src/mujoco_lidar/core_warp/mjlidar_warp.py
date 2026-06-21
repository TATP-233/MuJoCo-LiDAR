import mujoco
import numpy as np
import warp as wp

from .kernels import (
    trace_rays_batch_bvh_kernel,
    trace_rays_batch_kernel,
    trace_rays_bvh_kernel,
    trace_rays_kernel,
    update_aabbs_batch_kernel,
    update_aabbs_kernel,
)


class MjLidarWarp:
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        cutoff_dist: float = 100.0,
        geomgroup: np.ndarray | None = None,
        bodyexclude: int = -1,
        device: str | None = None,
        use_bvh: bool = True,
    ) -> None:
        wp.init()
        self.mj_model = mj_model
        self.cutoff_dist = cutoff_dist
        self.device = device
        self.use_bvh = use_bvh

        geom_types = mj_model.geom_type.astype(np.int32).copy()
        geom_sizes = mj_model.geom_size.astype(np.float32).copy()
        geom_aabb_center = mj_model.geom_aabb[:, :3].astype(np.float32).copy()
        geom_aabb_size = mj_model.geom_aabb[:, 3:].astype(np.float32).copy()
        cylinder_capsule_args = np.where((geom_types == 3) | (geom_types == 5))[0]
        geom_sizes[cylinder_capsule_args, 2] = geom_sizes[cylinder_capsule_args, 1]
        geom_sizes[cylinder_capsule_args, 1] = geom_sizes[cylinder_capsule_args, 0]
        plane_args = np.where(geom_types == 0)[0]
        geom_aabb_center[plane_args, :] = 0.0
        geom_aabb_size[plane_args, :] = geom_sizes[plane_args, :]

        if bodyexclude >= 0:
            geom_types[mj_model.geom_bodyid == bodyexclude] = -1

        if geomgroup is not None:
            geomgroup = np.asarray(geomgroup)
            for group_id in range(mujoco.mjNGROUP):
                if not geomgroup[group_id]:
                    geom_types[mj_model.geom_group == group_id] = -1

        self.geom_types = wp.array(geom_types, dtype=wp.int32, device=self.device)
        self.geom_sizes = wp.array(geom_sizes, dtype=wp.vec3, device=self.device)
        self.geom_aabb_center = wp.array(geom_aabb_center, dtype=wp.vec3, device=self.device)
        self.geom_aabb_size = wp.array(geom_aabb_size, dtype=wp.vec3, device=self.device)
        self._meshes, geom_mesh_ids = self._build_meshes(mj_model, geom_types)
        self.geom_mesh_ids = wp.array(geom_mesh_ids, dtype=wp.int32, device=self.device)
        self.mesh_ids = wp.array(
            np.array([mesh.id for mesh in self._meshes], dtype=np.uint64),
            dtype=wp.uint64,
            device=self.device,
        )

        self._geom_xpos = None
        self._geom_xmat = None
        self._aabb_lowers = wp.zeros(mj_model.ngeom, dtype=wp.vec3, device=self.device)
        self._aabb_uppers = wp.zeros(mj_model.ngeom, dtype=wp.vec3, device=self.device)
        self._bvh = wp.Bvh(self._aabb_lowers, self._aabb_uppers) if self.use_bvh else None
        self._bvh_built = False
        self._pose = None
        self._theta = None
        self._phi = None
        self._distances = None
        self._hit_points = None
        self._batch_shape = None
        self._batch_aabb_lowers = None
        self._batch_aabb_uppers = None
        self._batch_bvh = None
        self._batch_bvh_built = False

    def update(self, mj_data: mujoco.MjData) -> None:
        self._geom_xpos = wp.array(
            mj_data.geom_xpos.astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self._geom_xmat = wp.array(
            mj_data.geom_xmat.astype(np.float32), dtype=wp.float32, device=self.device
        )
        if self._bvh is not None:
            wp.launch(
                update_aabbs_kernel,
                dim=self.mj_model.ngeom,
                inputs=[
                    self.geom_types,
                    self.geom_sizes,
                    self.geom_aabb_center,
                    self.geom_aabb_size,
                    self._geom_xpos,
                    self._geom_xmat,
                    self._aabb_lowers,
                    self._aabb_uppers,
                ],
                device=self.device,
            )
            if self._bvh_built:
                self._bvh.refit()
            else:
                self._bvh.rebuild()
                self._bvh_built = True

    def trace_rays(self, pose_4x4: np.ndarray, ray_theta: np.ndarray, ray_phi: np.ndarray) -> None:
        if ray_phi.shape[0] != ray_theta.shape[0]:
            raise ValueError("ray_phi and ray_theta must have the same shape")

        n_rays = ray_phi.shape[0]
        self._ensure_capacity(n_rays)
        self._pose = wp.array(pose_4x4.astype(np.float32), dtype=wp.float32, device=self.device)
        self._theta = wp.array(ray_theta.astype(np.float32), dtype=wp.float32, device=self.device)
        self._phi = wp.array(ray_phi.astype(np.float32), dtype=wp.float32, device=self.device)

        if self._bvh is None:
            wp.launch(
                trace_rays_kernel,
                dim=n_rays,
                inputs=[
                    self.geom_types,
                    self.geom_sizes,
                    self.geom_mesh_ids,
                    self.mesh_ids,
                    self._geom_xpos,
                    self._geom_xmat,
                    self._pose,
                    self._theta,
                    self._phi,
                    float(self.cutoff_dist),
                    self._distances,
                    self._hit_points,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                trace_rays_bvh_kernel,
                dim=n_rays,
                inputs=[
                    self._bvh.id,
                    self.geom_types,
                    self.geom_sizes,
                    self.geom_mesh_ids,
                    self.mesh_ids,
                    self._geom_xpos,
                    self._geom_xmat,
                    self._pose,
                    self._theta,
                    self._phi,
                    float(self.cutoff_dist),
                    self._distances,
                    self._hit_points,
                ],
                device=self.device,
            )
        wp.synchronize_device(self.device)

    def trace_rays_batch(
        self,
        geom_xpos: np.ndarray,
        geom_xmat: np.ndarray,
        sensor_pos: np.ndarray,
        sensor_rot: np.ndarray,
        ray_theta: np.ndarray,
        ray_phi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if ray_phi.shape[0] != ray_theta.shape[0]:
            raise ValueError("ray_phi and ray_theta must have the same shape")
        if geom_xmat.ndim == 4:
            geom_xmat = geom_xmat.reshape(geom_xmat.shape[0], geom_xmat.shape[1], 9)

        n_envs = geom_xpos.shape[0]
        n_rays = ray_theta.shape[0]
        geom_xpos_wp = wp.array(geom_xpos.astype(np.float32), dtype=wp.vec3, device=self.device)
        geom_xmat_wp = wp.array(geom_xmat.astype(np.float32), dtype=wp.float32, device=self.device)
        sensor_pos_wp = wp.array(sensor_pos.astype(np.float32), dtype=wp.vec3, device=self.device)
        sensor_rot_wp = wp.array(
            sensor_rot.astype(np.float32), dtype=wp.float32, device=self.device
        )
        theta_wp = wp.array(ray_theta.astype(np.float32), dtype=wp.float32, device=self.device)
        phi_wp = wp.array(ray_phi.astype(np.float32), dtype=wp.float32, device=self.device)
        distances = wp.zeros((n_envs, n_rays), dtype=wp.float32, device=self.device)
        hit_points = wp.zeros((n_envs, n_rays), dtype=wp.vec3, device=self.device)

        if self.use_bvh:
            self._update_batch_bvh(n_envs, geom_xpos_wp, geom_xmat_wp)
            wp.launch(
                trace_rays_batch_bvh_kernel,
                dim=(n_envs, n_rays),
                inputs=[
                    self._batch_bvh.id,
                    self.geom_types,
                    self.geom_sizes,
                    self.geom_mesh_ids,
                    self.mesh_ids,
                    geom_xpos_wp,
                    geom_xmat_wp,
                    sensor_pos_wp,
                    sensor_rot_wp,
                    theta_wp,
                    phi_wp,
                    float(self.cutoff_dist),
                    distances,
                    hit_points,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                trace_rays_batch_kernel,
                dim=(n_envs, n_rays),
                inputs=[
                    self.geom_types,
                    self.geom_sizes,
                    self.geom_mesh_ids,
                    self.mesh_ids,
                    geom_xpos_wp,
                    geom_xmat_wp,
                    sensor_pos_wp,
                    sensor_rot_wp,
                    theta_wp,
                    phi_wp,
                    float(self.cutoff_dist),
                    distances,
                    hit_points,
                ],
                device=self.device,
            )
        wp.synchronize_device(self.device)
        return distances.numpy(), hit_points.numpy()

    def get_hit_points(self) -> np.ndarray | None:
        if self._hit_points is None:
            return None
        return self._hit_points.numpy()

    def get_distances(self) -> np.ndarray | None:
        if self._distances is None:
            return None
        return self._distances.numpy()

    def _ensure_capacity(self, n_rays: int) -> None:
        if self._distances is None or self._distances.shape[0] != n_rays:
            self._distances = wp.zeros(n_rays, dtype=wp.float32, device=self.device)
            self._hit_points = wp.zeros(n_rays, dtype=wp.vec3, device=self.device)

    def _update_batch_bvh(self, n_envs: int, geom_xpos_wp, geom_xmat_wp) -> None:
        shape = (n_envs, self.mj_model.ngeom)
        if self._batch_shape != shape:
            n_bounds = n_envs * self.mj_model.ngeom
            self._batch_aabb_lowers = wp.zeros(n_bounds, dtype=wp.vec3, device=self.device)
            self._batch_aabb_uppers = wp.zeros(n_bounds, dtype=wp.vec3, device=self.device)
            groups = np.repeat(np.arange(n_envs, dtype=np.int32), self.mj_model.ngeom)
            groups_wp = wp.array(groups, dtype=wp.int32, device=self.device)
            self._batch_bvh = wp.Bvh(
                self._batch_aabb_lowers,
                self._batch_aabb_uppers,
                groups=groups_wp,
            )
            self._batch_shape = shape
            self._batch_bvh_built = False

        wp.launch(
            update_aabbs_batch_kernel,
            dim=shape,
            inputs=[
                self.geom_types,
                self.geom_sizes,
                self.geom_aabb_center,
                self.geom_aabb_size,
                geom_xpos_wp,
                geom_xmat_wp,
                self._batch_aabb_lowers,
                self._batch_aabb_uppers,
            ],
            device=self.device,
        )
        if self._batch_bvh_built:
            self._batch_bvh.refit()
        else:
            self._batch_bvh.rebuild()
            self._batch_bvh_built = True

    def _build_meshes(
        self, mj_model: mujoco.MjModel, geom_types: np.ndarray
    ) -> tuple[list[wp.Mesh], np.ndarray]:
        geom_mesh_ids = np.full(mj_model.ngeom, -1, dtype=np.int32)
        mesh_index_by_dataid: dict[tuple[int, int], int] = {}
        meshes: list[wp.Mesh] = []

        for geom_id, data_id in enumerate(mj_model.geom_dataid):
            if geom_types[geom_id] not in (1, 7) or data_id < 0:
                continue
            mesh_key = (int(geom_types[geom_id]), int(data_id))
            if mesh_key not in mesh_index_by_dataid:
                if geom_types[geom_id] == 1:
                    points, faces = self._build_hfield_mesh(mj_model, data_id)
                else:
                    vert_adr = mj_model.mesh_vertadr[data_id]
                    vert_num = mj_model.mesh_vertnum[data_id]
                    face_adr = mj_model.mesh_faceadr[data_id]
                    face_num = mj_model.mesh_facenum[data_id]

                    points = mj_model.mesh_vert[vert_adr : vert_adr + vert_num].astype(np.float32)
                    faces = mj_model.mesh_face[face_adr : face_adr + face_num].astype(np.int32)
                mesh = wp.Mesh(
                    points=wp.array(points, dtype=wp.vec3, device=self.device),
                    indices=wp.array(faces.reshape(-1), dtype=wp.int32, device=self.device),
                )
                mesh_index_by_dataid[mesh_key] = len(meshes)
                meshes.append(mesh)

            geom_mesh_ids[geom_id] = mesh_index_by_dataid[mesh_key]

        return meshes, geom_mesh_ids

    def _build_hfield_mesh(
        self, mj_model: mujoco.MjModel, hfield_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        nrow = mj_model.hfield_nrow[hfield_id]
        ncol = mj_model.hfield_ncol[hfield_id]
        adr = mj_model.hfield_adr[hfield_id]
        data = mj_model.hfield_data[adr : adr + nrow * ncol].reshape(nrow, ncol)
        size = mj_model.hfield_size[hfield_id]
        rx, ry, ez = size[0], size[1], size[2]

        x = np.linspace(-rx, rx, ncol, dtype=np.float32)
        y = np.linspace(-ry, ry, nrow, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        points = np.stack((xx, yy, data * ez), axis=-1).reshape(-1, 3).astype(np.float32)

        cells = np.arange((nrow - 1) * (ncol - 1), dtype=np.int32).reshape(nrow - 1, ncol - 1)
        row = cells // (ncol - 1)
        col = cells % (ncol - 1)
        v00 = row * ncol + col
        v10 = v00 + 1
        v01 = v00 + ncol
        v11 = v01 + 1
        faces = np.stack(
            (
                np.stack((v00, v10, v11), axis=-1),
                np.stack((v00, v11, v01), axis=-1),
            ),
            axis=2,
        ).reshape(-1, 3)
        return points, faces.astype(np.int32)
