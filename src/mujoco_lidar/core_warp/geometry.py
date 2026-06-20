import warp as wp


@wp.func
def transform_ray_to_local(ray_origin: wp.vec3, ray_dir: wp.vec3, center: wp.vec3, rot: wp.mat33):
    rot_t = wp.transpose(rot)
    return rot_t * (ray_origin - center), rot_t * ray_dir


@wp.func
def ray_sphere_distance(ray_origin: wp.vec3, ray_dir: wp.vec3, center: wp.vec3, radius: float):
    oc = ray_origin - center
    b = wp.dot(oc, ray_dir)
    c = wp.dot(oc, oc) - radius * radius
    disc = b * b - c
    t = -1.0
    if disc >= 0.0:
        root = wp.sqrt(disc)
        t0 = -b - root
        t1 = -b + root
        if t0 >= 0.0:
            t = t0
        elif t1 >= 0.0:
            t = 0.0
    return t


@wp.func
def ray_plane_distance(
    ray_origin: wp.vec3,
    ray_dir: wp.vec3,
    center: wp.vec3,
    size: wp.vec3,
    rot: wp.mat33,
):
    ro, rd = transform_ray_to_local(ray_origin, ray_dir, center, rot)
    t = -1.0
    if wp.abs(rd[2]) > 1.0e-6:
        candidate = -ro[2] / rd[2]
        hit = ro + candidate * rd
        in_bounds = wp.abs(hit[0]) <= size[0] and wp.abs(hit[1]) <= size[1]
        infinite = size[0] == 0.0 and size[1] == 0.0
        if candidate >= 0.0 and (in_bounds or infinite):
            t = candidate
    return t


@wp.func
def ray_box_distance(
    ray_origin: wp.vec3,
    ray_dir: wp.vec3,
    center: wp.vec3,
    size: wp.vec3,
    rot: wp.mat33,
):
    ro, rd = transform_ray_to_local(ray_origin, ray_dir, center, rot)
    t_enter = -1.0e20
    t_exit = 1.0e20
    valid = True

    for axis in range(3):
        origin = ro[axis]
        direction = rd[axis]
        extent = size[axis]
        if wp.abs(direction) < 1.0e-8:
            if origin < -extent or origin > extent:
                valid = False
        else:
            inv_dir = 1.0 / direction
            t0 = (-extent - origin) * inv_dir
            t1 = (extent - origin) * inv_dir
            if t0 > t1:
                tmp = t0
                t0 = t1
                t1 = tmp
            t_enter = wp.max(t_enter, t0)
            t_exit = wp.min(t_exit, t1)

    t = -1.0
    if valid and t_enter <= t_exit and t_exit >= 0.0:
        t = t_enter if t_enter >= 0.0 else 0.0
    return t


@wp.func
def ray_cylinder_distance(
    ray_origin: wp.vec3,
    ray_dir: wp.vec3,
    center: wp.vec3,
    size: wp.vec3,
    rot: wp.mat33,
):
    ro, rd = transform_ray_to_local(ray_origin, ray_dir, center, rot)
    radius = size[0]
    half_height = size[1]
    best = 1.0e20

    a = rd[0] * rd[0] + rd[1] * rd[1]
    b = 2.0 * (ro[0] * rd[0] + ro[1] * rd[1])
    c = ro[0] * ro[0] + ro[1] * ro[1] - radius * radius
    disc = b * b - 4.0 * a * c
    if a > 1.0e-8 and disc >= 0.0:
        root = wp.sqrt(disc)
        for sign in range(2):
            s = -1.0
            if sign == 1:
                s = 1.0
            t = (-b + s * root) / (2.0 * a)
            z = ro[2] + t * rd[2]
            if t >= 0.0 and wp.abs(z) <= half_height and t < best:
                best = t

    if wp.abs(rd[2]) > 1.0e-8:
        for cap in range(2):
            z_cap = -half_height
            if cap == 1:
                z_cap = half_height
            t = (z_cap - ro[2]) / rd[2]
            hit = ro + t * rd
            radial = hit[0] * hit[0] + hit[1] * hit[1]
            if t >= 0.0 and radial <= radius * radius and t < best:
                best = t

    if best == 1.0e20:
        best = -1.0
    return best


@wp.func
def ray_ellipsoid_distance(
    ray_origin: wp.vec3,
    ray_dir: wp.vec3,
    center: wp.vec3,
    size: wp.vec3,
    rot: wp.mat33,
):
    ro, rd = transform_ray_to_local(ray_origin, ray_dir, center, rot)
    a = (
        rd[0] * rd[0] / (size[0] * size[0])
        + rd[1] * rd[1] / (size[1] * size[1])
        + rd[2] * rd[2] / (size[2] * size[2])
    )
    b = 2.0 * (
        ro[0] * rd[0] / (size[0] * size[0])
        + ro[1] * rd[1] / (size[1] * size[1])
        + ro[2] * rd[2] / (size[2] * size[2])
    )
    c = (
        ro[0] * ro[0] / (size[0] * size[0])
        + ro[1] * ro[1] / (size[1] * size[1])
        + ro[2] * ro[2] / (size[2] * size[2])
        - 1.0
    )
    disc = b * b - 4.0 * a * c
    t = -1.0
    if disc >= 0.0 and a > 1.0e-8:
        root = wp.sqrt(disc)
        t0 = (-b - root) / (2.0 * a)
        t1 = (-b + root) / (2.0 * a)
        if t0 >= 0.0:
            t = t0
        elif t1 >= 0.0:
            t = 0.0
    return t


@wp.func
def ray_capsule_distance(
    ray_origin: wp.vec3,
    ray_dir: wp.vec3,
    center: wp.vec3,
    size: wp.vec3,
    rot: wp.mat33,
):
    ro, rd = transform_ray_to_local(ray_origin, ray_dir, center, rot)
    radius = size[0]
    half_height = size[1]
    best = ray_cylinder_distance(ray_origin, ray_dir, center, size, rot)
    if best < 0.0:
        best = 1.0e20

    bottom = wp.vec3(0.0, 0.0, -half_height)
    top = wp.vec3(0.0, 0.0, half_height)
    t0 = ray_sphere_distance(ro, rd, bottom, radius)
    t1 = ray_sphere_distance(ro, rd, top, radius)
    if t0 >= 0.0 and t0 < best:
        best = t0
    if t1 >= 0.0 and t1 < best:
        best = t1
    if best == 1.0e20:
        best = -1.0
    return best
