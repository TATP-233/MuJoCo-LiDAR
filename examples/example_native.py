import argparse
import platform
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from etils import epath

from mujoco_lidar import MjLidarWrapper, scan_gen

_MACOS_VIEWER_HELP = """\
On macOS, MuJoCo's viewer must run under mjpython.

  uv run mjpython examples/example_native.py

If mjpython fails with "Library not loaded: @rpath/libpython...", link libpython into .venv:

  uv run python -c "import sys, sysconfig, pathlib; lib=pathlib.Path(sysconfig.get_config_var('LIBDIR'))/f'libpython{sys.version_info.major}.{sys.version_info.minor}.dylib'; t=pathlib.Path('.venv')/lib.name; t.unlink(missing_ok=True); t.symlink_to(lib); print(t, '->', lib)"
"""


def _height_rgba(z_norm: np.ndarray) -> np.ndarray:
    rgba = np.empty((z_norm.shape[0], 4), dtype=np.float64)
    rgba[:, 0] = z_norm
    rgba[:, 1] = 1.0 - np.abs(z_norm - 0.5) * 2.0
    rgba[:, 2] = 1.0 - z_norm
    rgba[:, 3] = 0.8
    return rgba


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MID360 LiDAR demo (MuJoCo native viewer)")
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu",
        help="LiDAR后端 (cpu, taichi, jax, warp)",
        choices=["cpu", "taichi", "jax", "warp"],
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=6,
        help="Ray stride for mid360 (default 6 -> 4000 rays; use 1 for full 24000)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=12.0,
        help="LiDAR update rate in Hz (default 12)",
    )
    args = parser.parse_args()
    if args.downsample < 1:
        parser.error("--downsample must be >= 1")
    if args.rate <= 0:
        parser.error("--rate must be > 0")

    mjcf_file = epath.Path(__file__).parent.parent / "models" / "demo.xml"
    mj_model = mujoco.MjModel.from_xml_path(mjcf_file.as_posix())
    mj_data = mujoco.MjData(mj_model)

    n_substeps = int(round(1.0 / (mj_model.opt.timestep * args.rate)))
    print(f"n_substeps = {n_substeps} (physics steps per LiDAR frame)")

    lidar = MjLidarWrapper(
        mj_model,
        "lidar_site",
        backend=args.backend,
        args={"bodyexclude": mj_model.body("your_robot_name").id},
    )
    livox_generator = scan_gen.LivoxGenerator("mid360")
    rays_theta, rays_phi = livox_generator.sample_ray_angles(downsample=args.downsample)
    n_rays = rays_theta.shape[0]

    if platform.system() == "Darwin" and mujoco.viewer._MJPYTHON is None:
        sys.exit(_MACOS_VIEWER_HELP)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.user_scn.ngeom = n_rays
        for i in range(n_rays):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.03, 0, 0],
                pos=[0, 0, 0],
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 0.8]),
            )

        print("Starting simulation...")
        print(f"Rays per scan: {n_rays} (downsample={args.downsample})")

        geoms = viewer.user_scn.geoms
        _start_time = time.time()
        while viewer.is_running():
            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)

            rays_theta, rays_phi = livox_generator.sample_ray_angles(downsample=args.downsample)
            lidar.trace_rays(mj_data, rays_theta, rays_phi)
            points = lidar.get_hit_points()
            world_points = points @ lidar.sensor_rotation.T + lidar.sensor_position

            z_values = world_points[:, 2]
            z_min, z_max = z_values.min(), z_values.max()
            z_norm = (
                (z_values - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(z_values)
            )
            colors = _height_rgba(z_norm)

            for i in range(n_rays):
                geoms[i].pos[:] = world_points[i]
                geoms[i].rgba[:] = colors[i]

            viewer.sync()
            elapsed = time.time() - _start_time
            if elapsed < mj_data.time:
                time.sleep(mj_data.time - elapsed)
