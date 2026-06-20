# API 参考

## MjLidarWrapper

LiDAR 仿真的主接口，统一封装 CPU、Taichi、JAX 和 Warp 后端。

### 构造函数

```python
MjLidarWrapper(
    mj_model: mujoco.MjModel,
    site_name: str,
    backend: str = "taichi",
    cutoff_dist: float = 100.0,
    args: dict = {}
)
```

**参数：**
- `mj_model`：MuJoCo 模型对象
- `site_name`：模型中 LiDAR site 的名称
- `backend`：计算后端，`"cpu"`、`"taichi"`、`"jax"` 或 `"warp"`（默认：`"taichi"`）
- `cutoff_dist`：最大射线距离，单位米（默认：100.0）
- `args`：后端专用参数（见下文）

**属性：**
- `backend`：当前后端名称
- `cutoff_dist`：最大射线距离
- `mj_model`：MuJoCo 模型引用

### 方法

#### trace_rays

```python
trace_rays(data: mujoco.MjData, theta: np.ndarray, phi: np.ndarray) -> np.ndarray
```

执行射线追踪，返回距离数组。

**参数：**
- `data`：MuJoCo 数据对象
- `theta`：方位角数组（弧度）
- `phi`：仰角数组（弧度）

**返回值：**
- `ranges`：距离数组（与 theta/phi 形状相同）

#### get_hit_points

```python
get_hit_points() -> np.ndarray
```

返回局部坐标系下的点云坐标，形状为 `(N, 3)`。

#### get_distances

```python
get_distances() -> np.ndarray
```

返回最近一次 `trace_rays` 的距离数组。

### 后端参数

#### CPU 后端

```python
args = {
    'geomgroup': np.ndarray | None,  # 几何组过滤器（0-5），None 表示全部
    'bodyexclude': int               # 排除的 body ID（-1 表示不排除）
}
```

#### Taichi 后端

```python
args = {
    'max_candidates': int,           # BVH 候选节点数（默认：64）
    'ti_init_args': {
        'device_memory_GB': float,   # GPU 显存限制（GB）
        'debug': bool,               # 调试模式
        'log_level': str             # 'trace', 'debug', 'info', 'warn', 'error'
    }
}
```

#### JAX 后端

```python
args = {
    'geom_ids': list | None,         # 包含的几何体 ID 列表（None 表示全部）
    'geomgroup': np.ndarray | None,  # 几何组过滤器（0-5），None 表示全部
    'bodyexclude': int               # 排除的 body ID（-1 表示不排除）
}
```

#### Warp 后端

```python
args = {
    'geomgroup': np.ndarray | None,  # 几何组过滤器（0-5），None 表示全部
    'bodyexclude': int,              # 排除的 body ID（-1 表示不排除）
    'device': str | None,            # Warp 设备，例如 "cuda:0"
    'use_bvh': bool                  # 是否使用 Warp BVH broad phase（默认：True）
}
```

---

## 扫描模式生成器

所有函数返回 `(theta, phi)` 元组，均为 numpy 数组。

### generate_HDL64

```python
scan_gen.generate_HDL64() -> tuple[np.ndarray, np.ndarray]
```

Velodyne HDL-64E 扫描模式（约 11 万射线）。

### generate_vlp32

```python
scan_gen.generate_vlp32() -> tuple[np.ndarray, np.ndarray]
```

Velodyne VLP-32C 扫描模式（约 12 万射线）。

### generate_os128

```python
scan_gen.generate_os128() -> tuple[np.ndarray, np.ndarray]
```

Ouster OS-128 扫描模式。

### generate_airy96

```python
scan_gen.generate_airy96() -> tuple[np.ndarray, np.ndarray]
```

RoboSense Airy-96 扫描模式（约 8.6 万射线）。

### generate_grid_scan_pattern

```python
scan_gen.generate_grid_scan_pattern(
    num_ray_cols: int,
    num_ray_rows: int,
    theta_range: tuple = (-π, π),
    phi_range: tuple = (-π/3, π/3)
) -> tuple[np.ndarray, np.ndarray]
```

自定义网格扫描模式。

### create_lidar_single_line

```python
scan_gen.create_lidar_single_line(
    horizontal_resolution: int = 360,
    horizontal_fov: float = 2π
) -> tuple[np.ndarray, np.ndarray]
```

单线水平扫描。

## LivoxGenerator

```python
from mujoco_lidar.scan_gen_livox_ti import LivoxGenerator

livox = LivoxGenerator(
    pattern: str = "mid360",  # "mid360", "mid70", "mid40", "tele", "avia"
    samples: int = 100000
)

theta, phi = livox.sample_ray_angles()
```

非重复性 Livox 扫描模式。
