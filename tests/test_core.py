import mujoco
import numpy as np

from mujoco_lidar import MjLidarWrapper


def test_wrapper_initialization(simple_model):
    """测试 wrapper 初始化"""
    lidar = MjLidarWrapper(simple_model, site_name="lidar_site", backend="cpu")
    assert lidar is not None


def test_trace_rays_basic(simple_model, simple_rays):
    """测试基础射线追踪"""
    lidar = MjLidarWrapper(simple_model, site_name="lidar_site", backend="cpu")
    data = mujoco.MjData(simple_model)
    mujoco.mj_forward(simple_model, data)
    theta, phi = simple_rays
    ranges = lidar.trace_rays(data, theta, phi)
    assert len(ranges) == len(theta)
    assert np.all(ranges >= -1)
