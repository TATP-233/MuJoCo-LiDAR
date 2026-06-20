import time

import mujoco
import numpy as np

from mujoco_lidar import MjLidarWrapper, scan_gen


def create_benchmark_scene():
    """创建基准测试场景"""
    xml = """
    <mujoco>
      <worldbody>
        <body name="box1" pos="2 0 0.5">
          <geom type="box" size="0.5 0.5 0.5"/>
        </body>
        <body name="box2" pos="0 2 0.5">
          <geom type="box" size="0.5 0.5 0.5"/>
        </body>
        <body name="sphere" pos="-2 0 0.5">
          <geom type="sphere" size="0.5"/>
        </body>
        <site name="lidar_site" pos="0 0 1"/>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


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
        }

    return results


def benchmark_trace_rays(backend="cpu", n_runs=10):
    """基准测试：射线追踪速度"""
    model = create_benchmark_scene()
    data = mujoco.MjData(model)

    try:
        lidar = MjLidarWrapper(model, site_name="lidar_site", backend=backend)
    except ImportError:
        return None

    theta, phi = scan_gen.generate_HDL64()

    # Warmup
    lidar.trace_rays(data, theta, phi)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        lidar.trace_rays(data, theta, phi)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "backend": backend,
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "n_rays": len(theta),
        "rays_per_sec": len(theta) / np.mean(times),
    }


if __name__ == "__main__":
    print("=== Ray Generation Benchmark ===")
    gen_results = benchmark_ray_generation()
    for name, result in gen_results.items():
        print(f"{name}: {result['mean_ms']:.2f}±{result['std_ms']:.2f}ms ({result['n_rays']} rays)")

    print("\n=== Ray Tracing Benchmark ===")
    for backend in ["cpu", "taichi", "jax", "warp"]:
        result = benchmark_trace_rays(backend)
        if result:
            print(
                f"{backend}: {result['mean_ms']:.2f}±{result['std_ms']:.2f}ms ({result['rays_per_sec']:.0f} rays/s)"
            )
