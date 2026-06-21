import numpy as np

from mujoco_lidar import MjLidarWrapper


def test_trace_rays_returns_correct_shape(simple_model, simple_rays):
    """测试返回形状正确"""
    lidar = MjLidarWrapper(simple_model, site_name="lidar_site", backend="cpu")
    data = __import__("mujoco").MjData(simple_model)
    __import__("mujoco").mj_forward(simple_model, data)
    theta, phi = simple_rays
    ranges = lidar.trace_rays(data, theta, phi)
    assert ranges.shape == theta.shape


def test_trace_rays_with_cutoff(simple_model):
    """测试截断距离"""
    lidar = MjLidarWrapper(simple_model, site_name="lidar_site", backend="cpu", cutoff_dist=10.0)
    data = __import__("mujoco").MjData(simple_model)
    __import__("mujoco").mj_forward(simple_model, data)
    theta = np.array([0, np.pi / 2])
    phi = np.array([0, 0])
    ranges = lidar.trace_rays(data, theta, phi)
    assert np.all(ranges <= 10.0)
    assert ranges[1] == -1
    np.testing.assert_allclose(lidar.get_hit_points()[1], np.zeros(3))
