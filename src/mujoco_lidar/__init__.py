# Lazy import to avoid loading dependencies when not needed
# Import the wrapper class directly
from typing import Any

from mujoco_lidar.lidar_wrapper import MjLidarWrapper

__version__ = "0.3.3"

__all__ = [
    "MjLidarWrapper",
    # Scan generation functions (imported lazily via __getattr__)
    "LivoxGeneratorTi",
    "LivoxGenerator",  # From scan_gen_livox (requires taichi)
    "generate_grid_scan_pattern",
    "create_lidar_single_line",
    "generate_HDL64",  # From scan_gen (no taichi needed)
    "generate_vlp32",
    "generate_os128",
    "generate_airy96",
]


def __getattr__(name: str) -> Any:
    """Lazy import for scan generation functions."""
    # LivoxGeneratorTi requires taichi - import from scan_gen_livox_ti
    if name == "LivoxGeneratorTi":
        from mujoco_lidar.scan_gen_livox_ti import LivoxGeneratorTi

        return LivoxGeneratorTi
    # Other scan functions don't require taichi - import from scan_gen
    elif name in [
        "LivoxGenerator",
        "generate_grid_scan_pattern",
        "create_lidar_single_line",
        "generate_HDL64",
        "generate_vlp32",
        "generate_os128",
        "generate_airy96",
    ]:
        from mujoco_lidar import scan_gen

        return getattr(scan_gen, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
