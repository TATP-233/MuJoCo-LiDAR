import time
from pathlib import Path

import mujoco
import numpy as np

from mujoco_lidar import MjLidarWrapper, scan_gen

_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _ROOT / "models"
_TRACE_SCENES = {
    "go2": "scene_go2.xml",
    "go2_stairs": "scene_go2_stairs_terrain.xml",
}


def benchmark_ray_generation(n_runs=10):
    """基准测试：射线生成速度"""
    results = {}

    patterns = {
        "HDL64": scan_gen.generate_HDL64,
        "VLP32": scan_gen.generate_vlp32,
        "Airy96": scan_gen.generate_airy96,
    }

    for name, gen_func in patterns.items():
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            theta, phi = gen_func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        results[name] = {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "n_rays": len(theta),
            "m_rays_per_sec": len(theta) / np.mean(times) / 1e6,
        }

    return results


def load_model(filename):
    return mujoco.MjModel.from_xml_path((_MODELS_DIR / filename).as_posix())


def benchmark_trace_rays(scene_name, scene_file, backend="cpu", n_runs=10):
    """基准测试：射线追踪速度"""
    model = load_model(scene_file)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    geomgroup = np.ones((mujoco.mjNGROUP,), dtype=np.ubyte)
    geomgroup[3:] = 0

    try:
        lidar = MjLidarWrapper(
            model,
            site_name="lidar",
            backend=backend,
            args={"bodyexclude": model.body("base").id, "geomgroup": geomgroup},
        )
    except ImportError:
        return None

    theta, phi = scan_gen.LivoxGenerator("mid360").sample_ray_angles()
    theta = np.ascontiguousarray(theta).astype(np.float32)
    phi = np.ascontiguousarray(phi).astype(np.float32)

    # Warmup
    ranges = lidar.trace_rays(data, theta, phi)
    if hasattr(ranges, "block_until_ready"):
        ranges.block_until_ready()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        ranges = lidar.trace_rays(data, theta, phi)
        if hasattr(ranges, "block_until_ready"):
            ranges.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "scene": scene_name,
        "backend": backend,
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "n_rays": len(theta),
        "m_rays_per_sec": len(theta) / np.mean(times) / 1e6,
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


if __name__ == "__main__":
    generation_rows = []
    tracing_rows = []

    print("=== Ray Generation Benchmark ===")
    gen_results = benchmark_ray_generation()
    for name, result in gen_results.items():
        print(
            f"{name}: {result['mean_ms']:.2f}±{result['std_ms']:.2f}ms "
            f"({result['m_rays_per_sec']:.3f} M rays/s)"
        )
        generation_rows.append(
            [
                name,
                str(result["n_rays"]),
                f"{result['mean_ms']:.2f}",
                f"{result['std_ms']:.2f}",
                f"{result['m_rays_per_sec']:.3f} M rays/s",
            ]
        )

    print("\n=== Ray Tracing Benchmark ===")
    for scene_name, scene_file in _TRACE_SCENES.items():
        print(f"\n--- {scene_name} ({scene_file}) ---")
        for backend in ["cpu", "taichi", "jax", "warp"]:
            result = benchmark_trace_rays(scene_name, scene_file, backend)
            if result:
                print(
                    f"{backend}: {result['mean_ms']:.2f}±{result['std_ms']:.2f}ms "
                    f"({result['m_rays_per_sec']:.3f} M rays/s)"
                )
                tracing_rows.append(
                    [
                        scene_name,
                        backend,
                        str(result["n_rays"]),
                        f"{result['mean_ms']:.2f}",
                        f"{result['std_ms']:.2f}",
                        f"{result['m_rays_per_sec']:.3f} M rays/s",
                    ]
                )

    print("\n=== Ray Generation Summary ===")
    print_table(["Name", "Rays", "Mean ms", "Std ms", "Throughput"], generation_rows)

    print("\n=== Ray Tracing Summary ===")
    print_table(["Scene", "Backend", "Rays", "Mean ms", "Std ms", "Throughput"], tracing_rows)
