# 安装指南

## 系统要求

**基础依赖：**
- Python >= 3.10（支持 3.10 – 3.13）
- MuJoCo >= 3.2.0
- NumPy >= 1.20.0

**可选后端依赖：**
- **Taichi**：`taichi >= 1.6.0`，`tibvh >= 0.1.2`
- **JAX**：`jax[cuda12]`
- **Warp**：`warp-lang >= 1.11.0`

## 快速安装

### 从 PyPI 安装

```bash
# 基础安装（仅 CPU 后端）
uv add mujoco-lidar

# 验证安装
uv run python -c "import mujoco_lidar; print(mujoco_lidar.__version__)"

# 安装 Taichi 后端
uv add "mujoco-lidar[taichi]"

# 安装 JAX 后端
uv add "mujoco-lidar[jax]"

# 安装 Warp 后端
uv add "mujoco-lidar[warp]"
```

### 从源码安装

```bash
git clone https://github.com/TATP-233/MuJoCo-LiDAR.git
cd MuJoCo-LiDAR

# 安装开发依赖和非 ROS 示例依赖
uv sync --extra dev --extra examples

# 可选 GPU 后端
uv sync --extra dev --extra examples --extra taichi
uv sync --extra dev --extra examples --extra warp
uv sync --extra dev --extra examples --extra taichi --extra jax --extra warp

# 运行非 ROS 示例
uv run --extra dev --extra examples python examples/example_native.py --backend cpu
uv run --extra dev --extra examples --extra warp python examples/example_native.py --backend warp

# 运行测试
uv run --extra dev pytest tests/

# 已安装 Warp 后端时运行 Warp 测试
uv run --extra dev --extra warp pytest tests/test_warp_backend.py
```

## 后端说明

- **CPU**：无需 GPU，开箱即用
- **Taichi**：需要 NVIDIA GPU 和 CUDA
- **JAX**：支持批量仿真，暂不支持 Mesh 几何体
- **Warp**：需要 NVIDIA GPU 和 CUDA，支持动态 Mesh 场景和 batch-scene
