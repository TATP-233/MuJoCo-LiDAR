"""Warp backend for MuJoCo-LiDAR."""

from importlib import import_module


def __getattr__(name: str):
    if name == "MjLidarWarp":
        return import_module(".mjlidar_warp", __name__).MjLidarWarp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MjLidarWarp"]
