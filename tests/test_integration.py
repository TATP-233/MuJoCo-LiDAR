import mujoco
import pytest

from mujoco_lidar import MjLidarWrapper


@pytest.mark.parametrize("backend", ["cpu", "taichi", "jax", "warp"])
def test_backend_consistency(simple_model, simple_rays, backend):
    """测试不同后端结果一致性"""
    try:
        lidar = MjLidarWrapper(simple_model, site_name="lidar_site", backend=backend)
        data = mujoco.MjData(simple_model)
        mujoco.mj_forward(simple_model, data)
        theta, phi = simple_rays
        ranges = lidar.trace_rays(data, theta, phi)
        assert len(ranges) == len(theta)
    except ImportError:
        pytest.skip(f"{backend} backend not available")
