import warp as wp

from .geometry import (
    ray_box_distance,
    ray_capsule_distance,
    ray_cylinder_distance,
    ray_ellipsoid_distance,
    ray_plane_distance,
    ray_sphere_distance,
)


@wp.func
def compute_oriented_box_aabb(center: wp.vec3, size: wp.vec3, rot: wp.mat33):
    extent = wp.vec3(
        wp.abs(rot[0, 0]) * size[0] + wp.abs(rot[0, 1]) * size[1] + wp.abs(rot[0, 2]) * size[2],
        wp.abs(rot[1, 0]) * size[0] + wp.abs(rot[1, 1]) * size[1] + wp.abs(rot[1, 2]) * size[2],
        wp.abs(rot[2, 0]) * size[0] + wp.abs(rot[2, 1]) * size[1] + wp.abs(rot[2, 2]) * size[2],
    )
    return center - extent, center + extent


@wp.kernel
def update_aabbs_kernel(
    geom_types: wp.array(dtype=wp.int32),
    geom_sizes: wp.array(dtype=wp.vec3),
    geom_aabb_center: wp.array(dtype=wp.vec3),
    geom_aabb_size: wp.array(dtype=wp.vec3),
    geom_xpos: wp.array(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.float32),
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
):
    geom_id = wp.tid()
    geom_type = geom_types[geom_id]
    rot = read_mat33(geom_xmat, geom_id)
    pos = geom_xpos[geom_id]

    lower = wp.vec3(1.0e10, 1.0e10, 1.0e10)
    upper = wp.vec3(-1.0e10, -1.0e10, -1.0e10)
    if geom_type < 0:
        lower = wp.vec3(0.0, 0.0, 0.0)
        upper = wp.vec3(0.0, 0.0, 0.0)
    elif geom_type == 0:
        lower, upper = compute_oriented_box_aabb(
            pos,
            wp.vec3(geom_sizes[geom_id][0], geom_sizes[geom_id][1], 1.0e-3),
            rot,
        )
    else:
        aabb_center = pos + rot * geom_aabb_center[geom_id]
        lower, upper = compute_oriented_box_aabb(aabb_center, geom_aabb_size[geom_id], rot)

    eps = wp.vec3(1.0e-4, 1.0e-4, 1.0e-4)
    lowers[geom_id] = lower - eps
    uppers[geom_id] = upper + eps


@wp.kernel
def update_aabbs_batch_kernel(
    geom_types: wp.array(dtype=wp.int32),
    geom_sizes: wp.array(dtype=wp.vec3),
    geom_aabb_center: wp.array(dtype=wp.vec3),
    geom_aabb_size: wp.array(dtype=wp.vec3),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array3d(dtype=wp.float32),
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
):
    env_id, geom_id = wp.tid()
    flat_id = env_id * geom_types.shape[0] + geom_id
    geom_type = geom_types[geom_id]
    rot = read_mat33_batch(geom_xmat, env_id, geom_id)
    pos = geom_xpos[env_id, geom_id]

    lower = wp.vec3(1.0e10, 1.0e10, 1.0e10)
    upper = wp.vec3(-1.0e10, -1.0e10, -1.0e10)
    if geom_type < 0:
        lower = wp.vec3(0.0, 0.0, 0.0)
        upper = wp.vec3(0.0, 0.0, 0.0)
    elif geom_type == 0:
        lower, upper = compute_oriented_box_aabb(
            pos,
            wp.vec3(geom_sizes[geom_id][0], geom_sizes[geom_id][1], 1.0e-3),
            rot,
        )
    else:
        aabb_center = pos + rot * geom_aabb_center[geom_id]
        lower, upper = compute_oriented_box_aabb(aabb_center, geom_aabb_size[geom_id], rot)

    eps = wp.vec3(1.0e-4, 1.0e-4, 1.0e-4)
    lowers[flat_id] = lower - eps
    uppers[flat_id] = upper + eps


@wp.func
def read_mat33(mats: wp.array2d(dtype=wp.float32), geom_id: int):
    return wp.mat33(
        mats[geom_id, 0],
        mats[geom_id, 1],
        mats[geom_id, 2],
        mats[geom_id, 3],
        mats[geom_id, 4],
        mats[geom_id, 5],
        mats[geom_id, 6],
        mats[geom_id, 7],
        mats[geom_id, 8],
    )


@wp.func
def read_mat33_batch(mats: wp.array3d(dtype=wp.float32), env_id: int, geom_id: int):
    return wp.mat33(
        mats[env_id, geom_id, 0],
        mats[env_id, geom_id, 1],
        mats[env_id, geom_id, 2],
        mats[env_id, geom_id, 3],
        mats[env_id, geom_id, 4],
        mats[env_id, geom_id, 5],
        mats[env_id, geom_id, 6],
        mats[env_id, geom_id, 7],
        mats[env_id, geom_id, 8],
    )


@wp.func
def read_pose_rot(pose: wp.array2d(dtype=wp.float32)):
    return wp.mat33(
        pose[0, 0],
        pose[0, 1],
        pose[0, 2],
        pose[1, 0],
        pose[1, 1],
        pose[1, 2],
        pose[2, 0],
        pose[2, 1],
        pose[2, 2],
    )


@wp.func
def trace_geom(
    geom_type: int,
    ray_origin: wp.vec3,
    ray_dir: wp.vec3,
    center: wp.vec3,
    size: wp.vec3,
    rot: wp.mat33,
):
    t = -1.0
    if geom_type == 0:
        t = ray_plane_distance(ray_origin, ray_dir, center, size, rot)
    elif geom_type == 2:
        t = ray_sphere_distance(ray_origin, ray_dir, center, size[0])
    elif geom_type == 3:
        t = ray_capsule_distance(ray_origin, ray_dir, center, size, rot)
    elif geom_type == 4:
        t = ray_ellipsoid_distance(ray_origin, ray_dir, center, size, rot)
    elif geom_type == 5:
        t = ray_cylinder_distance(ray_origin, ray_dir, center, size, rot)
    elif geom_type == 6:
        t = ray_box_distance(ray_origin, ray_dir, center, size, rot)
    return t


@wp.func
def ray_mesh_distance(
    mesh_id: wp.uint64,
    ray_origin: wp.vec3,
    ray_dir: wp.vec3,
    center: wp.vec3,
    rot: wp.mat33,
    max_t: float,
):
    rot_t = wp.transpose(rot)
    local_origin = rot_t * (ray_origin - center)
    local_dir = wp.normalize(rot_t * ray_dir)
    query = wp.mesh_query_ray(mesh_id, local_origin, local_dir, max_t)
    t = -1.0
    if query.result:
        t = query.t
    return t


@wp.kernel
def trace_rays_kernel(
    geom_types: wp.array(dtype=wp.int32),
    geom_sizes: wp.array(dtype=wp.vec3),
    geom_mesh_ids: wp.array(dtype=wp.int32),
    mesh_ids: wp.array(dtype=wp.uint64),
    geom_xpos: wp.array(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.float32),
    pose: wp.array2d(dtype=wp.float32),
    theta: wp.array(dtype=wp.float32),
    phi: wp.array(dtype=wp.float32),
    cutoff: float,
    distances: wp.array(dtype=wp.float32),
    hit_points: wp.array(dtype=wp.vec3),
):
    ray_id = wp.tid()
    t_angle = theta[ray_id]
    p_angle = phi[ray_id]
    cos_t = wp.cos(t_angle)
    sin_t = wp.sin(t_angle)
    cos_p = wp.cos(p_angle)
    sin_p = wp.sin(p_angle)
    local_dir = wp.vec3(cos_p * cos_t, cos_p * sin_t, sin_p)

    sensor_rot = read_pose_rot(pose)
    ray_dir = wp.normalize(sensor_rot * local_dir)
    origin = wp.vec3(pose[0, 3], pose[1, 3], pose[2, 3])

    best = cutoff
    for geom_id in range(geom_types.shape[0]):
        geom_type = geom_types[geom_id]
        if geom_type >= 0:
            t = -1.0
            rot = read_mat33(geom_xmat, geom_id)
            if geom_type == 1 or geom_type == 7:
                mesh_idx = geom_mesh_ids[geom_id]
                if mesh_idx >= 0:
                    t = ray_mesh_distance(
                        mesh_ids[mesh_idx],
                        origin,
                        ray_dir,
                        geom_xpos[geom_id],
                        rot,
                        best,
                    )
            else:
                t = trace_geom(
                    geom_type,
                    origin,
                    ray_dir,
                    geom_xpos[geom_id],
                    geom_sizes[geom_id],
                    rot,
                )
            if t >= 0.0 and t < best:
                best = t

    if best < cutoff:
        distances[ray_id] = best
        hit_points[ray_id] = best * local_dir
    else:
        distances[ray_id] = -1.0
        hit_points[ray_id] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def trace_rays_bvh_kernel(
    bvh_id: wp.uint64,
    geom_types: wp.array(dtype=wp.int32),
    geom_sizes: wp.array(dtype=wp.vec3),
    geom_mesh_ids: wp.array(dtype=wp.int32),
    mesh_ids: wp.array(dtype=wp.uint64),
    geom_xpos: wp.array(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.float32),
    pose: wp.array2d(dtype=wp.float32),
    theta: wp.array(dtype=wp.float32),
    phi: wp.array(dtype=wp.float32),
    cutoff: float,
    distances: wp.array(dtype=wp.float32),
    hit_points: wp.array(dtype=wp.vec3),
):
    ray_id = wp.tid()
    t_angle = theta[ray_id]
    p_angle = phi[ray_id]
    cos_t = wp.cos(t_angle)
    sin_t = wp.sin(t_angle)
    cos_p = wp.cos(p_angle)
    sin_p = wp.sin(p_angle)
    local_dir = wp.vec3(cos_p * cos_t, cos_p * sin_t, sin_p)

    sensor_rot = read_pose_rot(pose)
    ray_dir = wp.normalize(sensor_rot * local_dir)
    origin = wp.vec3(pose[0, 3], pose[1, 3], pose[2, 3])

    best = cutoff
    query = wp.bvh_query_ray(bvh_id, origin, ray_dir)
    geom_id = int(0)  # noqa: UP018
    while wp.bvh_query_next(query, geom_id):
        geom_type = geom_types[geom_id]
        if geom_type >= 0:
            t = -1.0
            rot = read_mat33(geom_xmat, geom_id)
            if geom_type == 1 or geom_type == 7:
                mesh_idx = geom_mesh_ids[geom_id]
                if mesh_idx >= 0:
                    t = ray_mesh_distance(
                        mesh_ids[mesh_idx],
                        origin,
                        ray_dir,
                        geom_xpos[geom_id],
                        rot,
                        best,
                    )
            else:
                t = trace_geom(
                    geom_type,
                    origin,
                    ray_dir,
                    geom_xpos[geom_id],
                    geom_sizes[geom_id],
                    rot,
                )
            if t >= 0.0 and t < best:
                best = t

    if best < cutoff:
        distances[ray_id] = best
        hit_points[ray_id] = best * local_dir
    else:
        distances[ray_id] = -1.0
        hit_points[ray_id] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def trace_rays_batch_kernel(
    geom_types: wp.array(dtype=wp.int32),
    geom_sizes: wp.array(dtype=wp.vec3),
    geom_mesh_ids: wp.array(dtype=wp.int32),
    mesh_ids: wp.array(dtype=wp.uint64),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array3d(dtype=wp.float32),
    sensor_pos: wp.array(dtype=wp.vec3),
    sensor_rot: wp.array3d(dtype=wp.float32),
    theta: wp.array(dtype=wp.float32),
    phi: wp.array(dtype=wp.float32),
    cutoff: float,
    distances: wp.array2d(dtype=wp.float32),
    hit_points: wp.array2d(dtype=wp.vec3),
):
    env_id, ray_id = wp.tid()
    t_angle = theta[ray_id]
    p_angle = phi[ray_id]
    cos_t = wp.cos(t_angle)
    sin_t = wp.sin(t_angle)
    cos_p = wp.cos(p_angle)
    sin_p = wp.sin(p_angle)
    local_dir = wp.vec3(cos_p * cos_t, cos_p * sin_t, sin_p)

    rot = wp.mat33(
        sensor_rot[env_id, 0, 0],
        sensor_rot[env_id, 0, 1],
        sensor_rot[env_id, 0, 2],
        sensor_rot[env_id, 1, 0],
        sensor_rot[env_id, 1, 1],
        sensor_rot[env_id, 1, 2],
        sensor_rot[env_id, 2, 0],
        sensor_rot[env_id, 2, 1],
        sensor_rot[env_id, 2, 2],
    )
    ray_dir = wp.normalize(rot * local_dir)
    origin = sensor_pos[env_id]

    best = cutoff
    for geom_id in range(geom_types.shape[0]):
        geom_type = geom_types[geom_id]
        if geom_type >= 0:
            t = -1.0
            geom_rot = read_mat33_batch(geom_xmat, env_id, geom_id)
            if geom_type == 1 or geom_type == 7:
                mesh_idx = geom_mesh_ids[geom_id]
                if mesh_idx >= 0:
                    t = ray_mesh_distance(
                        mesh_ids[mesh_idx],
                        origin,
                        ray_dir,
                        geom_xpos[env_id, geom_id],
                        geom_rot,
                        best,
                    )
            else:
                t = trace_geom(
                    geom_type,
                    origin,
                    ray_dir,
                    geom_xpos[env_id, geom_id],
                    geom_sizes[geom_id],
                    geom_rot,
                )
            if t >= 0.0 and t < best:
                best = t

    if best < cutoff:
        distances[env_id, ray_id] = best
        hit_points[env_id, ray_id] = best * local_dir
    else:
        distances[env_id, ray_id] = -1.0
        hit_points[env_id, ray_id] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def trace_rays_batch_bvh_kernel(
    bvh_id: wp.uint64,
    geom_types: wp.array(dtype=wp.int32),
    geom_sizes: wp.array(dtype=wp.vec3),
    geom_mesh_ids: wp.array(dtype=wp.int32),
    mesh_ids: wp.array(dtype=wp.uint64),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array3d(dtype=wp.float32),
    sensor_pos: wp.array(dtype=wp.vec3),
    sensor_rot: wp.array3d(dtype=wp.float32),
    theta: wp.array(dtype=wp.float32),
    phi: wp.array(dtype=wp.float32),
    cutoff: float,
    distances: wp.array2d(dtype=wp.float32),
    hit_points: wp.array2d(dtype=wp.vec3),
):
    env_id, ray_id = wp.tid()
    t_angle = theta[ray_id]
    p_angle = phi[ray_id]
    cos_t = wp.cos(t_angle)
    sin_t = wp.sin(t_angle)
    cos_p = wp.cos(p_angle)
    sin_p = wp.sin(p_angle)
    local_dir = wp.vec3(cos_p * cos_t, cos_p * sin_t, sin_p)

    rot = wp.mat33(
        sensor_rot[env_id, 0, 0],
        sensor_rot[env_id, 0, 1],
        sensor_rot[env_id, 0, 2],
        sensor_rot[env_id, 1, 0],
        sensor_rot[env_id, 1, 1],
        sensor_rot[env_id, 1, 2],
        sensor_rot[env_id, 2, 0],
        sensor_rot[env_id, 2, 1],
        sensor_rot[env_id, 2, 2],
    )
    ray_dir = wp.normalize(rot * local_dir)
    origin = sensor_pos[env_id]

    best = cutoff
    root = wp.bvh_get_group_root(bvh_id, env_id)
    query = wp.bvh_query_ray(bvh_id, origin, ray_dir, root)
    flat_id = int(0)  # noqa: UP018
    while wp.bvh_query_next(query, flat_id):
        geom_id = flat_id - env_id * geom_types.shape[0]
        if geom_id >= 0 and geom_id < geom_types.shape[0]:
            geom_type = geom_types[geom_id]
            t = -1.0
            geom_rot = read_mat33_batch(geom_xmat, env_id, geom_id)
            if geom_type < 0:
                pass
            elif geom_type == 1 or geom_type == 7:
                mesh_idx = geom_mesh_ids[geom_id]
                if mesh_idx >= 0:
                    t = ray_mesh_distance(
                        mesh_ids[mesh_idx],
                        origin,
                        ray_dir,
                        geom_xpos[env_id, geom_id],
                        geom_rot,
                        best,
                    )
            else:
                t = trace_geom(
                    geom_type,
                    origin,
                    ray_dir,
                    geom_xpos[env_id, geom_id],
                    geom_sizes[geom_id],
                    geom_rot,
                )
            if t >= 0.0 and t < best:
                best = t

    if best < cutoff:
        distances[env_id, ray_id] = best
        hit_points[env_id, ray_id] = best * local_dir
    else:
        distances[env_id, ray_id] = -1.0
        hit_points[env_id, ray_id] = wp.vec3(0.0, 0.0, 0.0)
