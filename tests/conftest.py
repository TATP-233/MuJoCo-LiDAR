import mujoco
import numpy as np
import pytest


@pytest.fixture
def simple_model():
    """创建简单测试场景"""
    xml = """
    <mujoco>
      <worldbody>
        <body name="box" pos="1 0 0.5">
          <geom type="box" size="0.5 0.5 0.5"/>
        </body>
        <site name="lidar_site" pos="0 0 1"/>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


@pytest.fixture
def simple_rays():
    """简单射线模式"""
    theta = np.array([0, np.pi / 4, np.pi / 2], dtype=np.float32)
    phi = np.array([0, 0, 0], dtype=np.float32)
    return theta, phi
