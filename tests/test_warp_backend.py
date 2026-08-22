import importlib.util

import mujoco
import numpy as np
import pytest

from mujoco_lidar import MjLidarWrapper

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("warp") is None,
    reason="warp backend not available",
)


def test_warp_backend_matches_cpu_for_simple_box(simple_model):
    data = mujoco.MjData(simple_model)
    mujoco.mj_forward(simple_model, data)
    theta = np.array([0.0, np.pi / 2], dtype=np.float32)
    phi = np.array([0.0, 0.0], dtype=np.float32)

    cpu_lidar = MjLidarWrapper(simple_model, site_name="lidar_site", backend="cpu")
    warp_lidar = MjLidarWrapper(simple_model, site_name="lidar_site", backend="warp")

    cpu_ranges = cpu_lidar.trace_rays(data, theta, phi)
    warp_ranges = warp_lidar.trace_rays(data, theta, phi)

    np.testing.assert_allclose(warp_ranges, cpu_ranges, atol=1e-5)


def test_warp_backend_batch_trace(simple_model):
    data = mujoco.MjData(simple_model)
    mujoco.mj_forward(simple_model, data)
    lidar = MjLidarWrapper(simple_model, site_name="lidar_site", backend="warp")

    geom_xpos = np.stack([data.geom_xpos, data.geom_xpos]).astype(np.float32)
    geom_xmat = np.stack([data.geom_xmat, data.geom_xmat]).astype(np.float32)
    sensor_pos = np.stack([data.site("lidar_site").xpos, data.site("lidar_site").xpos]).astype(
        np.float32
    )
    sensor_rot = np.stack(
        [
            data.site("lidar_site").xmat.reshape(3, 3),
            data.site("lidar_site").xmat.reshape(3, 3),
        ]
    ).astype(np.float32)
    theta = np.array([0.0], dtype=np.float32)
    phi = np.array([0.0], dtype=np.float32)

    distances, hit_points = lidar.trace_rays_batch(
        geom_xpos,
        geom_xmat,
        sensor_pos,
        sensor_rot,
        theta,
        phi,
    )

    np.testing.assert_allclose(distances, np.array([[0.5], [0.5]], dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(
        hit_points,
        np.array([[[0.5, 0.0, 0.0]], [[0.5, 0.0, 0.0]]], dtype=np.float32),
        atol=1e-5,
    )


def test_warp_backend_batch_trace_separates_scenes(simple_model):
    data = mujoco.MjData(simple_model)
    mujoco.mj_forward(simple_model, data)
    lidar = MjLidarWrapper(simple_model, site_name="lidar_site", backend="warp")

    geom_xpos = np.stack([data.geom_xpos, data.geom_xpos]).astype(np.float32)
    geom_xpos[1, 0, 0] = 2.0
    geom_xmat = np.stack([data.geom_xmat, data.geom_xmat]).astype(np.float32)
    sensor_pos = np.stack([data.site("lidar_site").xpos, data.site("lidar_site").xpos]).astype(
        np.float32
    )
    sensor_rot = np.stack(
        [
            data.site("lidar_site").xmat.reshape(3, 3),
            data.site("lidar_site").xmat.reshape(3, 3),
        ]
    ).astype(np.float32)
    theta = np.array([0.0], dtype=np.float32)
    phi = np.array([0.0], dtype=np.float32)

    distances, _ = lidar.trace_rays_batch(
        geom_xpos,
        geom_xmat,
        sensor_pos,
        sensor_rot,
        theta,
        phi,
    )

    np.testing.assert_allclose(distances, np.array([[0.5], [1.5]], dtype=np.float32), atol=1e-5)


def test_warp_backend_returns_negative_one_on_miss(simple_model):
    data = mujoco.MjData(simple_model)
    mujoco.mj_forward(simple_model, data)
    theta = np.array([np.pi / 2], dtype=np.float32)
    phi = np.array([0.0], dtype=np.float32)

    cpu_lidar = MjLidarWrapper(simple_model, site_name="lidar_site", backend="cpu")
    warp_lidar = MjLidarWrapper(simple_model, site_name="lidar_site", backend="warp")

    np.testing.assert_allclose(cpu_lidar.trace_rays(data, theta, phi), np.array([-1.0]))
    np.testing.assert_allclose(warp_lidar.trace_rays(data, theta, phi), np.array([-1.0]))


def test_warp_backend_mesh_matches_cpu():
    xml = """
    <mujoco>
      <asset>
        <mesh
          name="tri"
          vertex="1 -1 -1 1 1 -1 1 0 1 2 0 0"
          face="0 1 2 0 3 1 1 3 2 2 3 0"
        />
      </asset>
      <worldbody>
        <geom type="mesh" mesh="tri"/>
        <site name="lidar_site" pos="0 0 0"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    theta = np.array([0.0], dtype=np.float32)
    phi = np.array([0.0], dtype=np.float32)

    cpu_lidar = MjLidarWrapper(model, site_name="lidar_site", backend="cpu")
    warp_lidar = MjLidarWrapper(model, site_name="lidar_site", backend="warp")

    cpu_ranges = cpu_lidar.trace_rays(data, theta, phi)
    warp_ranges = warp_lidar.trace_rays(data, theta, phi)

    np.testing.assert_allclose(warp_ranges, cpu_ranges, atol=1e-5)


def test_warp_backend_mesh_uses_dynamic_geom_pose():
    xml = """
    <mujoco>
      <asset>
        <mesh
          name="tri"
          vertex="0 -1 -1 0 1 -1 0 0 1 1 0 0"
          face="0 1 2 0 3 1 1 3 2 2 3 0"
        />
      </asset>
      <worldbody>
        <body name="mesh_body" pos="1 0 0">
          <joint name="slide_x" type="slide" axis="1 0 0"/>
          <geom type="mesh" mesh="tri"/>
        </body>
        <site name="lidar_site" pos="0 0 0"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    lidar = MjLidarWrapper(model, site_name="lidar_site", backend="warp")
    theta = np.array([0.0], dtype=np.float32)
    phi = np.array([0.0], dtype=np.float32)

    mujoco.mj_forward(model, data)
    near = lidar.trace_rays(data, theta, phi)

    data.qpos[0] = 1.0
    mujoco.mj_forward(model, data)
    far = lidar.trace_rays(data, theta, phi)

    np.testing.assert_allclose(near, np.array([1.0], dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(far, np.array([2.0], dtype=np.float32), atol=1e-5)


def test_warp_backend_hfield_matches_cpu():
    xml = """
    <mujoco>
      <asset>
        <hfield name="terrain" nrow="2" ncol="2" size="1 1 1 0.1" elevation="0 0 0 0"/>
      </asset>
      <worldbody>
        <geom type="hfield" hfield="terrain"/>
        <site name="lidar_site" pos="0 0 1"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    theta = np.array([0.0], dtype=np.float32)
    phi = np.array([-np.pi / 2], dtype=np.float32)

    cpu_lidar = MjLidarWrapper(model, site_name="lidar_site", backend="cpu")
    warp_lidar = MjLidarWrapper(model, site_name="lidar_site", backend="warp")

    cpu_ranges = cpu_lidar.trace_rays(data, theta, phi)
    warp_ranges = warp_lidar.trace_rays(data, theta, phi)

    np.testing.assert_allclose(warp_ranges, cpu_ranges, atol=1e-5)


def test_warp_backend_capsule_cylinder_match_cpu():
    xml = """
    <mujoco>
      <worldbody>
        <geom type="capsule" pos="2 0 0.5" quat="0.7071068 0 0.7071068 0" size="0.3 0.8"/>
        <geom type="capsule" pos="0 3 -0.2" size="0.2 0.5"/>
        <geom type="cylinder" pos="-2 0 0" size="0.4 0.9"/>
        <site name="lidar_site" pos="0 0 0"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    rng = np.random.default_rng(0)
    theta = rng.uniform(0, np.pi, 2000).astype(np.float32)
    phi = rng.uniform(-np.pi / 2, np.pi / 2, 2000).astype(np.float32)

    cpu_lidar = MjLidarWrapper(model, site_name="lidar_site", backend="cpu")
    warp_lidar = MjLidarWrapper(model, site_name="lidar_site", backend="warp")

    cpu_ranges = cpu_lidar.trace_rays(data, theta, phi)
    warp_ranges = warp_lidar.trace_rays(data, theta, phi)

    np.testing.assert_allclose(warp_ranges, cpu_ranges, atol=1e-5)
