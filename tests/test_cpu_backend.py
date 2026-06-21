import mujoco
import numpy as np

from mujoco_lidar import MjLidarWrapper


def test_cpu_backend_basic(simple_model, simple_rays):
    """测试 CPU 后端基础功能"""
    lidar = MjLidarWrapper(simple_model, site_name="lidar_site", backend="cpu")
    data = mujoco.MjData(simple_model)
    mujoco.mj_forward(simple_model, data)
    theta, phi = simple_rays
    ranges = lidar.trace_rays(data, theta, phi)
    assert ranges.shape == theta.shape
    assert np.all(np.isfinite(ranges))
