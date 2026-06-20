# API Reference

## MjLidarWrapper

Main interface for LiDAR simulation.

### Constructor

```python
MjLidarWrapper(
    mj_model: mujoco.MjModel,
    site_name: str,
    backend: str = "taichi",
    cutoff_dist: float = 100.0,
    args: dict = {}
)
```

**Parameters:**
- `mj_model`: MuJoCo model object
- `site_name`: Name of LiDAR site in the model
- `backend`: `"cpu"`, `"taichi"`, `"jax"`, or `"warp"` (default: `"taichi"`)
- `cutoff_dist`: Maximum ray distance in meters (default: 100.0)
- `args`: Backend-specific arguments (see below)

**Attributes:**
- `backend`: Selected backend name
- `cutoff_dist`: Maximum ray distance
- `mj_model`: MuJoCo model reference

### Methods

#### trace_rays

```python
trace_rays(data: mujoco.MjData, theta: np.ndarray, phi: np.ndarray) -> np.ndarray
```

Trace rays and return ranges.

**Parameters:**
- `data`: MuJoCo data object
- `theta`: Azimuth angles in radians
- `phi`: Elevation angles in radians

**Returns:**
- `ranges`: Distance array (same shape as theta/phi)

### Backend Arguments

#### CPU Backend

```python
args = {
    'geomgroup': np.ndarray | None,  # Geometry group filter (0-5)
    'bodyexclude': int               # Body ID to exclude (-1 = none)
}
```

#### Taichi Backend

```python
args = {
    'max_candidates': int,           # BVH candidates (default: 64)
    'ti_init_args': {
        'device_memory_GB': float,   # GPU memory limit
        'debug': bool,               # Debug mode
        'log_level': str            # 'trace', 'debug', 'info', 'warn', 'error'
    }
}
```

#### JAX Backend

```python
args = {
    'geom_ids': list | None         # Geometry IDs to include (None = all)
}
```

#### Warp Backend

```python
args = {
    'geomgroup': np.ndarray | None,  # Geometry group filter (0-5)
    'bodyexclude': int,              # Body ID to exclude (-1 = none)
    'device': str | None,            # Warp device, e.g. "cuda:0"
    'use_bvh': bool                  # Use Warp BVH broad phase (default: True)
}
```

## Scan Pattern Generators

All functions return `(theta, phi)` tuple of numpy arrays.

### generate_HDL64

```python
scan_gen.generate_HDL64() -> tuple[np.ndarray, np.ndarray]
```

Velodyne HDL-64E pattern (~110K rays).

### generate_vlp32

```python
scan_gen.generate_vlp32() -> tuple[np.ndarray, np.ndarray]
```

Velodyne VLP-32C pattern (~120K rays).

### generate_os128

```python
scan_gen.generate_os128() -> tuple[np.ndarray, np.ndarray]
```

Ouster OS-128 pattern.

### generate_airy96

```python
scan_gen.generate_airy96() -> tuple[np.ndarray, np.ndarray]
```

RoboSense Airy-96 pattern (~86K rays).

### generate_grid_scan_pattern

```python
scan_gen.generate_grid_scan_pattern(
    num_ray_cols: int,
    num_ray_rows: int,
    theta_range: tuple = (-π, π),
    phi_range: tuple = (-π/3, π/3)
) -> tuple[np.ndarray, np.ndarray]
```

Custom grid pattern.

### create_lidar_single_line

```python
scan_gen.create_lidar_single_line(
    horizontal_resolution: int = 360,
    horizontal_fov: float = 2π
) -> tuple[np.ndarray, np.ndarray]
```

Single horizontal scan line.

## LivoxGenerator

```python
from mujoco_lidar.scan_gen_livox_ti import LivoxGenerator

livox = LivoxGenerator(
    pattern: str = "mid360",  # "mid360", "mid70", "mid40", "tele", "avia"
    samples: int = 100000
)

theta, phi = livox.sample_ray_angles()
```

Non-repetitive Livox scanning patterns.
