import argparse
import importlib.util
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

from mujoco_lidar import MjLidarWrapper, scan_gen
from mujoco_lidar.core_warp.kernels import trace_rays_batch_bvh_kernel, trace_rays_batch_kernel

_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _ROOT / "models"
_TRACE_SCENES = {
    "go2": "scene_go2.xml",
    "go2_stairs": "scene_go2_stairs_terrain.xml",
}


@wp.kernel
def _make_mujoco_warp_rays(
    sensor_pos: wp.array(dtype=wp.vec3),
    sensor_rot: wp.array3d(dtype=wp.float32),
    theta: wp.array(dtype=wp.float32),
    phi: wp.array(dtype=wp.float32),
    origins: wp.array2d(dtype=wp.vec3),
    directions: wp.array2d(dtype=wp.vec3),
):
    world_id, ray_id = wp.tid()
    t_angle = theta[ray_id]
    p_angle = phi[ray_id]
    local_dir = wp.vec3(
        wp.cos(p_angle) * wp.cos(t_angle),
        wp.cos(p_angle) * wp.sin(t_angle),
        wp.sin(p_angle),
    )
    rot = wp.mat33(
        sensor_rot[world_id, 0, 0],
        sensor_rot[world_id, 0, 1],
        sensor_rot[world_id, 0, 2],
        sensor_rot[world_id, 1, 0],
        sensor_rot[world_id, 1, 1],
        sensor_rot[world_id, 1, 2],
        sensor_rot[world_id, 2, 0],
        sensor_rot[world_id, 2, 1],
        sensor_rot[world_id, 2, 2],
    )
    origins[world_id, ray_id] = sensor_pos[world_id]
    directions[world_id, ray_id] = wp.normalize(rot * local_dir)


def load_model(filename):
    return mujoco.MjModel.from_xml_path((_MODELS_DIR / filename).as_posix())


def make_scene_state(model, n_envs):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    geom_xpos = np.repeat(data.geom_xpos[np.newaxis, :, :], n_envs, axis=0).astype(np.float32)
    geom_xmat = np.repeat(data.geom_xmat[np.newaxis, :, :], n_envs, axis=0).astype(np.float32)
    sensor_pos = np.repeat(data.site("lidar").xpos[np.newaxis, :], n_envs, axis=0).astype(
        np.float32
    )
    sensor_rot = np.repeat(data.site("lidar").xmat.reshape(1, 3, 3), n_envs, axis=0).astype(
        np.float32
    )

    return data, geom_xpos, geom_xmat, sensor_pos, sensor_rot


def make_rays(n_rays):
    theta, phi = scan_gen.LivoxGenerator("mid360").sample_ray_angles()
    theta = np.ascontiguousarray(theta[:n_rays]).astype(np.float32)
    phi = np.ascontiguousarray(phi[:n_rays]).astype(np.float32)
    return theta, phi


def benchmark_mujoco_lidar_warp(
    model,
    geom_xpos,
    geom_xmat,
    sensor_pos,
    sensor_rot,
    theta,
    phi,
    n_runs,
):
    geomgroup = np.ones((mujoco.mjNGROUP,), dtype=np.ubyte)
    geomgroup[3:] = 0
    lidar = MjLidarWrapper(
        model,
        site_name="lidar",
        backend="warp",
        args={"bodyexclude": model.body("base").id, "geomgroup": geomgroup},
    )
    lidar.trace_rays_batch(geom_xpos, geom_xmat, sensor_pos, sensor_rot, theta, phi)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        lidar.trace_rays_batch(geom_xpos, geom_xmat, sensor_pos, sensor_rot, theta, phi)
        times.append(time.perf_counter() - start)

    result = summarize("MuJoCo-LiDAR warp", geom_xpos.shape[0], theta.shape[0], times)
    result["segments"] = profile_mujoco_lidar_warp(
        lidar._backend_instance,
        geom_xpos,
        geom_xmat,
        sensor_pos,
        sensor_rot,
        theta,
        phi,
        n_runs,
    )
    return result


def profile_mujoco_lidar_warp(
    lidar,
    geom_xpos,
    geom_xmat,
    sensor_pos,
    sensor_rot,
    theta,
    phi,
    n_runs,
):
    if geom_xmat.ndim == 4:
        geom_xmat = geom_xmat.reshape(geom_xmat.shape[0], geom_xmat.shape[1], 9)

    n_envs = geom_xpos.shape[0]
    n_rays = theta.shape[0]
    segment_times = {
        "host_to_device": [],
        "output_alloc": [],
        "bvh_update": [],
        "trace_kernel": [],
        "readback": [],
        "profile_total": [],
    }

    for _ in range(n_runs):
        total_start = time.perf_counter()

        start = time.perf_counter()
        geom_xpos_wp = wp.array(geom_xpos.astype(np.float32), dtype=wp.vec3, device=lidar.device)
        geom_xmat_wp = wp.array(geom_xmat.astype(np.float32), dtype=wp.float32, device=lidar.device)
        sensor_pos_wp = wp.array(sensor_pos.astype(np.float32), dtype=wp.vec3, device=lidar.device)
        sensor_rot_wp = wp.array(
            sensor_rot.astype(np.float32), dtype=wp.float32, device=lidar.device
        )
        theta_wp = wp.array(theta.astype(np.float32), dtype=wp.float32, device=lidar.device)
        phi_wp = wp.array(phi.astype(np.float32), dtype=wp.float32, device=lidar.device)
        wp.synchronize_device(lidar.device)
        segment_times["host_to_device"].append(time.perf_counter() - start)

        start = time.perf_counter()
        distances = wp.zeros((n_envs, n_rays), dtype=wp.float32, device=lidar.device)
        hit_points = wp.zeros((n_envs, n_rays), dtype=wp.vec3, device=lidar.device)
        wp.synchronize_device(lidar.device)
        segment_times["output_alloc"].append(time.perf_counter() - start)

        start = time.perf_counter()
        if lidar.use_bvh:
            lidar._update_batch_bvh(n_envs, geom_xpos_wp, geom_xmat_wp)
        wp.synchronize_device(lidar.device)
        segment_times["bvh_update"].append(time.perf_counter() - start)

        start = time.perf_counter()
        if lidar.use_bvh:
            wp.launch(
                trace_rays_batch_bvh_kernel,
                dim=(n_envs, n_rays),
                inputs=[
                    lidar._batch_bvh.id,
                    lidar.geom_types,
                    lidar.geom_sizes,
                    lidar.geom_mesh_ids,
                    lidar.mesh_ids,
                    geom_xpos_wp,
                    geom_xmat_wp,
                    sensor_pos_wp,
                    sensor_rot_wp,
                    theta_wp,
                    phi_wp,
                    float(lidar.cutoff_dist),
                    distances,
                    hit_points,
                ],
                device=lidar.device,
            )
        else:
            wp.launch(
                trace_rays_batch_kernel,
                dim=(n_envs, n_rays),
                inputs=[
                    lidar.geom_types,
                    lidar.geom_sizes,
                    lidar.geom_mesh_ids,
                    lidar.mesh_ids,
                    geom_xpos_wp,
                    geom_xmat_wp,
                    sensor_pos_wp,
                    sensor_rot_wp,
                    theta_wp,
                    phi_wp,
                    float(lidar.cutoff_dist),
                    distances,
                    hit_points,
                ],
                device=lidar.device,
            )
        wp.synchronize_device(lidar.device)
        segment_times["trace_kernel"].append(time.perf_counter() - start)

        start = time.perf_counter()
        distances.numpy()
        hit_points.numpy()
        segment_times["readback"].append(time.perf_counter() - start)
        segment_times["profile_total"].append(time.perf_counter() - total_start)

    return {
        name: {
            "mean_ms": float(np.mean(times)) * 1000,
            "std_ms": float(np.std(times)) * 1000,
        }
        for name, times in segment_times.items()
    }


def benchmark_mujoco_warp(model, data, sensor_pos, sensor_rot, theta, phi, n_runs):
    if importlib.util.find_spec("mujoco_warp") is None:
        return None

    import mujoco_warp as mjw
    from mujoco_warp._src.types import vec6

    n_envs = sensor_pos.shape[0]
    n_rays = theta.shape[0]
    mjw_model = mjw.put_model(model)
    mjw_data = mjw.put_data(model, data, nworld=n_envs)
    rc = mjw.create_render_context(
        model,
        nworld=n_envs,
        enabled_geom_groups=[0, 1, 2],
        use_textures=False,
        use_precomputed_rays=False,
    )

    origins = wp.empty((n_envs, n_rays), dtype=wp.vec3)
    directions = wp.empty((n_envs, n_rays), dtype=wp.vec3)
    sensor_pos_wp = wp.array(sensor_pos, dtype=wp.vec3)
    sensor_rot_wp = wp.array(sensor_rot, dtype=wp.float32)
    theta_wp = wp.array(theta, dtype=wp.float32)
    phi_wp = wp.array(phi, dtype=wp.float32)
    dist = wp.empty((n_envs, n_rays), dtype=float)
    geomid = wp.empty((n_envs, n_rays), dtype=int)
    normal = wp.empty((n_envs, n_rays), dtype=wp.vec3)
    bodyexclude = wp.empty(n_rays, dtype=int)
    bodyexclude.fill_(model.body("base").id)
    geomgroup = vec6(1, 1, 1, 0, 0, 0)

    def run():
        wp.launch(
            _make_mujoco_warp_rays,
            dim=(n_envs, n_rays),
            inputs=[
                sensor_pos_wp,
                sensor_rot_wp,
                theta_wp,
                phi_wp,
                origins,
                directions,
            ],
        )
        mjw.refit_bvh(mjw_model, mjw_data, rc)
        mjw.rays(
            mjw_model,
            mjw_data,
            origins,
            directions,
            geomgroup,
            True,
            bodyexclude,
            dist,
            geomid,
            normal,
            rc,
        )
        wp.synchronize()

    run()
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)

    return summarize("mujoco-warp", n_envs, n_rays, times)


def summarize(name, n_envs, n_rays, times):
    mean_s = float(np.mean(times))
    return {
        "name": name,
        "n_envs": n_envs,
        "n_rays": n_rays,
        "mean_ms": mean_s * 1000,
        "std_ms": float(np.std(times)) * 1000,
        "m_rays_per_sec": (n_envs * n_rays) / mean_s / 1e6,
    }


def print_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt.format(*row))


def format_row(scene_name, result):
    return [
        scene_name,
        result["name"],
        str(result["n_envs"]),
        str(result["n_rays"]),
        f"{result['mean_ms']:.2f}",
        f"{result['std_ms']:.2f}",
        f"{result['m_rays_per_sec']:.3f} M rays/s",
    ]


def print_segments(result):
    segments = result.get("segments")
    if not segments:
        return

    profile_total = segments["profile_total"]["mean_ms"]
    print("MuJoCo-LiDAR warp breakdown:")
    print_table(
        ["Segment", "Mean ms", "Std ms", "% profiled"],
        [
            [
                name,
                f"{value['mean_ms']:.3f}",
                f"{value['std_ms']:.3f}",
                f"{value['mean_ms'] / profile_total * 100:.1f}%",
            ]
            for name, value in segments.items()
        ],
    )
    print(f"End-to-end trace_rays_batch mean: {result['mean_ms']:.3f} ms")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare MuJoCo-LiDAR warp and mujoco-warp raycast throughput on the same MuJoCo scenes."
        )
    )
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-rays", type=int, default=1024)
    args = parser.parse_args()

    rows = []
    print("=== Batch Raycaster Benchmark ===")
    print(
        "MuJoCo-LiDAR warp includes numpy->Warp upload, batch BVH rebuild, synchronize, and numpy readback."
    )
    print(
        "mujoco-warp keeps ray buffers and outputs on device, refits RenderContext BVH, and only synchronizes for timing."
    )

    for scene_name, scene_file in _TRACE_SCENES.items():
        print(f"\n--- {scene_name} ({scene_file}) ---")
        model = load_model(scene_file)
        data, geom_xpos, geom_xmat, sensor_pos, sensor_rot = make_scene_state(model, args.n_envs)
        theta, phi = make_rays(args.n_rays)

        lidar_result = benchmark_mujoco_lidar_warp(
            model,
            geom_xpos,
            geom_xmat,
            sensor_pos,
            sensor_rot,
            theta,
            phi,
            args.n_runs,
        )
        print(
            f"{lidar_result['name']}: {lidar_result['mean_ms']:.2f}+/-{lidar_result['std_ms']:.2f}ms "
            f"({lidar_result['m_rays_per_sec']:.3f} M rays/s)"
        )
        print_segments(lidar_result)
        rows.append(format_row(scene_name, lidar_result))

        mjw_result = benchmark_mujoco_warp(
            model, data, sensor_pos, sensor_rot, theta, phi, args.n_runs
        )
        if mjw_result is None:
            print("mujoco-warp: skipped (package not installed)")
            continue
        print(
            f"{mjw_result['name']}: {mjw_result['mean_ms']:.2f}+/-{mjw_result['std_ms']:.2f}ms "
            f"({mjw_result['m_rays_per_sec']:.3f} M rays/s)"
        )
        rows.append(format_row(scene_name, mjw_result))

    print("\n=== Batch Raycaster Summary ===")
    print_table(["Scene", "Backend", "Envs", "Rays/env", "Mean ms", "Std ms", "Throughput"], rows)


if __name__ == "__main__":
    main()
