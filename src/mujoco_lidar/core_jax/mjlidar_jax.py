from functools import partial

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjtGeom

from .geometry import (
    ray_box_intersection,
    ray_capsule_intersection,
    ray_cylinder_intersection,
    ray_ellipsoid_intersection,
    ray_hfield_intersection,
    ray_plane_intersection,
    ray_sphere_intersection,
)


class MjLidarJax:
    def __init__(
        self,
        model: mujoco.MjModel,
        geom_ids: np.ndarray | list | None = None,
        geomgroup: np.ndarray | list | None = None,
        bodyexclude: int = -1,
    ):
        self.model = model

        # If geom_ids is None, use all geoms
        if geom_ids is None:
            self.geom_ids = np.arange(model.ngeom)
        else:
            self.geom_ids = np.array(geom_ids)

        # Filter by geomgroup if provided
        if geomgroup is not None:
            geomgroup = np.asarray(geomgroup)
            # model.geom_group is (ngeom,)
            # geomgroup is (mjNGROUP,) where 1 means include
            mask = geomgroup[model.geom_group[self.geom_ids]] == 1
            self.geom_ids = self.geom_ids[mask]

        # Filter by bodyexclude if provided
        if bodyexclude >= 0:
            # model.geom_bodyid is (ngeom,)
            mask = model.geom_bodyid[self.geom_ids] != bodyexclude
            self.geom_ids = self.geom_ids[mask]

        # Extract static properties
        all_types = np.array(model.geom_type)

        # Filter by geom_ids
        self.selected_types = all_types[self.geom_ids]

        # Group indices by type
        # - PLANE (0): 平面
        # - HFIELD (1): 高度场
        # - SPHERE (2): 球体
        # - CAPSULE (3): 胶囊体
        # - ELLIPSOID (4): 椭球体
        # - CYLINDER (5): 圆柱体
        # - BOX (6): 长方体
        self.plane_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_PLANE]
        self.hfield_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_HFIELD]
        self.sphere_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_SPHERE]
        self.capsule_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_CAPSULE]
        self.ellipsoid_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_ELLIPSOID]
        self.cylinder_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_CYLINDER]
        self.box_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_BOX]

        # Convert to jnp arrays for JIT
        self.plane_ids = jnp.array(self.plane_ids)
        self.hfield_ids = jnp.array(self.hfield_ids)
        self.sphere_ids = jnp.array(self.sphere_ids)
        self.capsule_ids = jnp.array(self.capsule_ids)
        self.ellipsoid_ids = jnp.array(self.ellipsoid_ids)
        self.cylinder_ids = jnp.array(self.cylinder_ids)
        self.box_ids = jnp.array(self.box_ids)

        # Store sizes (static)
        self.geom_sizes = jnp.array(model.geom_size)

        # Extract hfield data
        if self.hfield_ids.shape[0] > 0:
            # We need to get the data for each hfield
            # mj_model.hfield_data is flat
            # mj_model.hfield_adr is index
            # mj_model.hfield_nrow/ncol

            # We need to map geom_id -> hfield_id
            # m.geom_dataid[geom_id] gives hfield_id

            hfield_geom_ids = np.array(self.hfield_ids)  # These are geom indices
            hfield_asset_ids = model.geom_dataid[hfield_geom_ids]

            # Get max dimensions
            max_nrow = np.max(model.hfield_nrow[hfield_asset_ids])
            max_ncol = np.max(model.hfield_ncol[hfield_asset_ids])

            n_hfields = len(hfield_geom_ids)
            self.hfield_data = np.zeros((n_hfields, max_nrow, max_ncol), dtype=np.float32)
            self.hfield_nrow = np.zeros(n_hfields, dtype=np.int32)
            self.hfield_ncol = np.zeros(n_hfields, dtype=np.int32)
            self.hfield_sizes = np.zeros((n_hfields, 4), dtype=np.float32)

            for i, hid in enumerate(hfield_asset_ids):
                nrow = model.hfield_nrow[hid]
                ncol = model.hfield_ncol[hid]
                adr = model.hfield_adr[hid]
                data = model.hfield_data[adr : adr + nrow * ncol].reshape(nrow, ncol)

                # Try Transpose
                # self.hfield_data[i, :nrow, :ncol] = data
                self.hfield_data[i, :nrow, :ncol] = data
                self.hfield_nrow[i] = nrow
                self.hfield_ncol[i] = ncol
                self.hfield_sizes[i] = model.hfield_size[hid]

            self.hfield_data = jnp.array(self.hfield_data)
            self.hfield_nrow = jnp.array(self.hfield_nrow)
            self.hfield_ncol = jnp.array(self.hfield_ncol)
            self.hfield_sizes = jnp.array(self.hfield_sizes)
        else:
            self.hfield_data = jnp.zeros((0, 0, 0))
            self.hfield_nrow = jnp.zeros(0, dtype=jnp.int32)
            self.hfield_ncol = jnp.zeros(0, dtype=jnp.int32)
            self.hfield_sizes = jnp.zeros((0, 4), dtype=np.float32)

    @partial(jax.jit, static_argnums=(0,))
    def render(
        self,
        geom_xpos: jax.Array,
        geom_xmat: jax.Array,
        rays_origin: jax.Array,
        rays_direction: jax.Array,
    ) -> jax.Array:
        """
        Render LiDAR scan for a single environment.

        Args:
            geom_xpos: (Ngeom, 3) Geometry positions
            geom_xmat: (Ngeom, 9) or (Ngeom, 3, 3) Geometry rotation matrices
            rays_origin: (3,) World position of sensor
            rays_direction: (Nrays, 3) World direction of rays

        Returns:
            distances: (Nrays,)
        """
        # Handle rotation matrix shape
        if geom_xmat.ndim == 2 and geom_xmat.shape[-1] == 9:
            geom_xmat = geom_xmat.reshape(-1, 3, 3)

        # Initialize with inf
        min_dist = jnp.full(rays_direction.shape[0], jnp.inf)

        # 1. Spheres
        if self.sphere_ids.shape[0] > 0:
            pos = geom_xpos[self.sphere_ids]
            rad = self.geom_sizes[self.sphere_ids, 0]

            def dist_all_rays_all_spheres(ro, rd, pos, rad):
                def scan_fn(carry, x):
                    p, r = x
                    dists = jax.vmap(lambda d: ray_sphere_intersection(ro, d, p, r))(rd)
                    return jnp.minimum(carry, dists), None

                final_dist, _ = jax.lax.scan(scan_fn, jnp.full(rd.shape[0], jnp.inf), (pos, rad))
                return final_dist

            d_spheres = dist_all_rays_all_spheres(rays_origin, rays_direction, pos, rad)
            min_dist = jnp.minimum(min_dist, d_spheres)

        # 2. Boxes
        if self.box_ids.shape[0] > 0:
            pos = geom_xpos[self.box_ids]
            rot = geom_xmat[self.box_ids]
            size = self.geom_sizes[self.box_ids]

            def dist_all_rays_all_boxes(ro, rd, pos, rot, size):
                def scan_fn(carry, x):
                    p, R, s = x
                    dists = jax.vmap(lambda d: ray_box_intersection(ro, d, p, R, s))(rd)
                    return jnp.minimum(carry, dists), None

                final_dist, _ = jax.lax.scan(
                    scan_fn, jnp.full(rd.shape[0], jnp.inf), (pos, rot, size)
                )
                return final_dist

            d_boxes = dist_all_rays_all_boxes(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_boxes)

        # 3. Capsules
        if self.capsule_ids.shape[0] > 0:
            pos = geom_xpos[self.capsule_ids]
            rot = geom_xmat[self.capsule_ids]
            size = self.geom_sizes[self.capsule_ids]

            def dist_all_rays_all_capsules(ro, rd, pos, rot, size):
                def scan_fn(carry, x):
                    p, R, s = x
                    dists = jax.vmap(lambda d: ray_capsule_intersection(ro, d, p, R, s))(rd)
                    return jnp.minimum(carry, dists), None

                final_dist, _ = jax.lax.scan(
                    scan_fn, jnp.full(rd.shape[0], jnp.inf), (pos, rot, size)
                )
                return final_dist

            d_capsules = dist_all_rays_all_capsules(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_capsules)

        # 4. Cylinders
        if self.cylinder_ids.shape[0] > 0:
            pos = geom_xpos[self.cylinder_ids]
            rot = geom_xmat[self.cylinder_ids]
            size = self.geom_sizes[self.cylinder_ids]

            def dist_all_rays_all_cylinders(ro, rd, pos, rot, size):
                def scan_fn(carry, x):
                    p, R, s = x
                    dists = jax.vmap(lambda d: ray_cylinder_intersection(ro, d, p, R, s))(rd)
                    return jnp.minimum(carry, dists), None

                final_dist, _ = jax.lax.scan(
                    scan_fn, jnp.full(rd.shape[0], jnp.inf), (pos, rot, size)
                )
                return final_dist

            d_cylinders = dist_all_rays_all_cylinders(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_cylinders)

        # 5. Planes
        if self.plane_ids.shape[0] > 0:
            pos = geom_xpos[self.plane_ids]
            rot = geom_xmat[self.plane_ids]
            size = self.geom_sizes[self.plane_ids]

            def dist_all_rays_all_planes(ro, rd, pos, rot, size):
                def scan_fn(carry, x):
                    p, R, s = x
                    dists = jax.vmap(lambda d: ray_plane_intersection(ro, d, p, R, s))(rd)
                    return jnp.minimum(carry, dists), None

                final_dist, _ = jax.lax.scan(
                    scan_fn, jnp.full(rd.shape[0], jnp.inf), (pos, rot, size)
                )
                return final_dist

            d_planes = dist_all_rays_all_planes(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_planes)

        # 6. Ellipsoids
        if self.ellipsoid_ids.shape[0] > 0:
            pos = geom_xpos[self.ellipsoid_ids]
            rot = geom_xmat[self.ellipsoid_ids]
            size = self.geom_sizes[self.ellipsoid_ids]

            def dist_all_rays_all_ellipsoids(ro, rd, pos, rot, size):
                def scan_fn(carry, x):
                    p, R, s = x
                    dists = jax.vmap(lambda d: ray_ellipsoid_intersection(ro, d, p, R, s))(rd)
                    return jnp.minimum(carry, dists), None

                final_dist, _ = jax.lax.scan(
                    scan_fn, jnp.full(rd.shape[0], jnp.inf), (pos, rot, size)
                )
                return final_dist

            d_ellipsoids = dist_all_rays_all_ellipsoids(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_ellipsoids)

        # 7. Hfields
        if self.hfield_ids.shape[0] > 0:
            pos = geom_xpos[self.hfield_ids]
            rot = geom_xmat[self.hfield_ids]
            size = self.hfield_sizes
            data = self.hfield_data
            nrows = self.hfield_nrow
            ncols = self.hfield_ncol

            def dist_all_rays_all_hfields(ro, rd, pos, rot, size, data, nr, nc):
                def scan_fn(carry, x):
                    p, R, s, d, n_r, n_c = x
                    dists = jax.vmap(
                        lambda ray_d: ray_hfield_intersection(ro, ray_d, p, R, s, d, n_r, n_c)
                    )(rd)
                    return jnp.minimum(carry, dists), None

                final_dist, _ = jax.lax.scan(
                    scan_fn, jnp.full(rd.shape[0], jnp.inf), (pos, rot, size, data, nrows, ncols)
                )
                return final_dist

            d_hfields = dist_all_rays_all_hfields(
                rays_origin, rays_direction, pos, rot, size, data, nrows, ncols
            )
            min_dist = jnp.minimum(min_dist, d_hfields)

        distance = jnp.where(jnp.isinf(min_dist), -1.0, min_dist)

        return distance

    @partial(jax.jit, static_argnums=(0,))
    def trace_rays(
        self,
        geom_xpos: jax.Array,
        geom_xmat: jax.Array,
        sensor_pos: jax.Array,
        sensor_mat: jax.Array,
        ray_theta: jax.Array,
        ray_phi: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Full ray tracing pipeline: ray generation, transformation, and rendering.
        """
        # 1. Ray generation (local space)
        x = jnp.cos(ray_phi) * jnp.cos(ray_theta)
        y = jnp.cos(ray_phi) * jnp.sin(ray_theta)
        z = jnp.sin(ray_phi)
        local_rays = jnp.stack([x, y, z], axis=-1)

        # 2. Transform to world rays
        world_rays = local_rays @ sensor_mat.T

        # 3. Render
        distances = self.render(geom_xpos, geom_xmat, sensor_pos, world_rays)

        return distances, local_rays

    @partial(jax.jit, static_argnums=(0,))
    def render_batch(
        self,
        geom_xpos: jax.Array,
        geom_xmat: jax.Array,
        rays_origin: jax.Array,
        rays_direction: jax.Array,
    ) -> jax.Array:
        """
        Render LiDAR scan for a batch of environments.

        Args:
            geom_xpos: (B, Ngeom, 3) Geometry positions
            geom_xmat: (B, Ngeom, 9) or (B, Ngeom, 3, 3) Geometry rotation matrices
            rays_origin: (B, 3) World position of sensor per env
            rays_direction: (B, Nrays, 3) World direction of rays per env

        Returns:
            distances: (B, Nrays)
        """
        # Optimization: Reshape rotation matrices once if needed
        if geom_xmat.ndim == 3 and geom_xmat.shape[-1] == 9:
            geom_xmat = geom_xmat.reshape(geom_xmat.shape[0], -1, 3, 3)

        return jax.vmap(self.render)(geom_xpos, geom_xmat, rays_origin, rays_direction)

    @partial(jax.jit, static_argnums=(0,))
    def trace_rays_batch(
        self,
        geom_xpos: jax.Array,
        geom_xmat: jax.Array,
        sensor_pos: jax.Array,
        sensor_mat: jax.Array,
        ray_theta: jax.Array,
        ray_phi: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Full ray tracing pipeline for a batch of environments: ray generation, transformation, and rendering.

        Args:
            geom_xpos: (B, Ngeom, 3) Geometry positions
            geom_xmat: (B, Ngeom, 9) or (B, Ngeom, 3, 3) Geometry rotation matrices
            sensor_pos: (B, 3) World position of sensor per env
            sensor_mat: (B, 3, 3) World rotation matrix of sensor per env
            ray_theta: (Nrays,) Ray horizontal angles
            ray_phi: (Nrays,) Ray vertical angles
        """
        # Optimization: Reshape rotation matrices once if needed
        if geom_xmat.ndim == 3 and geom_xmat.shape[-1] == 9:
            geom_xmat = geom_xmat.reshape(geom_xmat.shape[0], -1, 3, 3)

        # 1. Ray generation (local space) - Compute once for all envs
        x = jnp.cos(ray_phi) * jnp.cos(ray_theta)
        y = jnp.cos(ray_phi) * jnp.sin(ray_theta)
        z = jnp.sin(ray_phi)
        local_rays = jnp.stack([x, y, z], axis=-1)

        def trace_single_env(geom_xpos, geom_xmat, sensor_pos, sensor_mat):
            # 2. Transform to world rays
            world_rays = local_rays @ sensor_mat.T

            # 3. Render
            distances = self.render(geom_xpos, geom_xmat, sensor_pos, world_rays)

            return distances

        distances = jax.vmap(trace_single_env)(geom_xpos, geom_xmat, sensor_pos, sensor_mat)

        # Broadcast local_rays to match batch size
        batch_size = geom_xpos.shape[0]
        local_rays_batch = jnp.broadcast_to(local_rays, (batch_size, *local_rays.shape))

        return distances, local_rays_batch
