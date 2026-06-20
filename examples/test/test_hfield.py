import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"  # 如果显存充足，可以调大一些

import argparse
import pathlib
import time

import matplotlib.cm as cm
import mujoco
import numpy as np

from mujoco_lidar import MjLidarWrapper, scan_gen

np.set_printoptions(precision=3, suppress=True, linewidth=500)

parser = argparse.ArgumentParser(description="Test hfield LiDAR scan")
parser.add_argument("--save", action="store_true", help="Save point clouds to .ply files")
args = parser.parse_args()

# Load model
model_path = pathlib.Path(__file__).parent.parent.parent / "models" / "stairs_terrain.xml"
print(f"Loading model from: {model_path}")
mj_model = mujoco.MjModel.from_xml_path(model_path.as_posix())
mj_data = mujoco.MjData(mj_model)
mujoco.mj_step(mj_model, mj_data)

# Generate scan pattern
# Use a denser scan for hfield to see details
theta, phi = scan_gen.generate_airy96()
phi *= -1.0
print(f"Number of rays: {len(theta)}")

# Prepare random indices for sampling
np.random.seed(0)
rnd_args = np.random.randint(0, len(theta), size=30)

backends = ["cpu", "taichi", "jax", "warp"]
results = {}

for backend in backends:
    print(f"\nInitializing {backend.upper()} LiDAR...")
    try:
        # The site name in stairs_terrain.xml is "lidar"
        lidar = MjLidarWrapper(mj_model, site_name="static_site", backend=backend)

        # Warm up
        print("Running scan (warmup)...")
        ranges = lidar.trace_rays(mj_data, theta, phi)
        if backend == "jax":
            ranges.block_until_ready()

        # Timing
        print("Running benchmark...")
        start = time.time()
        num_runs = 10 if backend != "cpu" else 2
        for _ in range(num_runs):
            ranges = lidar.trace_rays(mj_data, theta, phi)
            if backend == "jax":
                ranges.block_until_ready()
        end = time.time()

        print(f"Scan time: {1e3 * (end - start) / num_runs:.2f}ms")

        # Store results
        ranges_np = np.array(ranges)

        # Check for valid hits (not inf, not 0)
        valid_hits = (ranges_np > 0) & (ranges_np < np.inf)
        print(f"Valid hits: {np.sum(valid_hits)} / {len(ranges_np)}")

        ranges_sorted = np.sort(ranges_np)
        results[backend] = ranges_sorted[rnd_args]

        if args.save:
            # Compute point cloud (x, y, z)
            r = ranges_np
            x = r * np.cos(phi) * np.cos(theta)
            y = r * np.cos(phi) * np.sin(theta)
            z = r * np.sin(phi)

            points = np.stack([x, y, z], axis=-1)

            # Save to PLY (filter invalid points)
            valid_mask = (r > 0) & (r < np.inf)
            valid_points = points[valid_mask]

            # Color mapping along Z axis
            z_vals = valid_points[:, 2]
            if len(z_vals) > 0:
                z_min, z_max = z_vals.min(), z_vals.max()
                z_range = z_max - z_min

                z_norm = (
                    np.zeros_like(z_vals) if z_range < 1e-6 else (z_vals - z_min) / z_range
                )

                # Map to RGB using matplotlib
                if backend == "cpu":
                    colors = (cm.jet(1.0 - z_norm)[:, :3] * 255).astype(np.uint8)
                else:
                    colors = (cm.jet(z_norm)[:, :3] * 255).astype(np.uint8)

                # Combine points and colors
                vertex_data = np.hstack([valid_points, colors])

                ply_filename = f"hfield_points_{backend}.ply"
                with open(ply_filename, "w") as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {len(valid_points)}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")
                    f.write("end_header\n")
                    np.savetxt(f, vertex_data, fmt="%.6f %.6f %.6f %d %d %d")

                print(f"Saved {ply_filename}")
            else:
                print("No valid points to save.")

    except Exception as e:
        print(f"Failed to run {backend} backend: {e}")
        import traceback

        traceback.print_exc()

print("\n" + "=" * 120)
print("Summary of Sample Ranges:")
for backend in backends:
    if backend in results:
        print(f"{backend:<8}: {results[backend]}")
    else:
        print(f"{backend:<8}: Failed")
print("=" * 120)
