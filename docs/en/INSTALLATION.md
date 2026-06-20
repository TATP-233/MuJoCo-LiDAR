# Installation Guide

## System Requirements

**Basic Dependencies:**
- Python >= 3.10 (3.10 – 3.13 supported)
- MuJoCo >= 3.2.0
- NumPy >= 1.20.0

**Optional Backend Dependencies:**
- **Taichi**: `taichi >= 1.6.0`, `tibvh >= 0.1.2`
- **JAX**: `jax[cuda12]`
- **Warp**: `warp-lang >= 1.11.0`

## Quick Installation

### From PyPI

```bash
# Basic (CPU backend)
uv add mujoco-lidar

# Verify
uv run python -c "import mujoco_lidar; print(mujoco_lidar.__version__)"

# With Taichi backend
uv add "mujoco-lidar[taichi]"

# With JAX backend
uv add "mujoco-lidar[jax]"

# With Warp backend
uv add "mujoco-lidar[warp]"
```

### From Source

```bash
git clone https://github.com/TATP-233/MuJoCo-LiDAR.git
cd MuJoCo-LiDAR

# Install with dev and non-ROS example dependencies
uv sync --extra dev --extra examples

# Optional GPU backends
uv sync --extra dev --extra examples --extra taichi
uv sync --extra dev --extra examples --extra warp
uv sync --extra dev --extra examples --extra taichi --extra jax --extra warp

# Run a non-ROS example
uv run --extra dev --extra examples python examples/example_native.py --backend cpu
uv run --extra dev --extra examples --extra warp python examples/example_native.py --backend warp

# Run tests
uv run --extra dev pytest tests/

# Run Warp tests, if the Warp backend is installed
uv run --extra dev --extra warp pytest tests/test_warp_backend.py
```

## Backend Notes

- **CPU**: No GPU required, works out-of-the-box
- **Taichi**: Requires NVIDIA GPU with CUDA
- **JAX**: Supports batch environments, no Mesh support currently
- **Warp**: Requires NVIDIA GPU with CUDA; supports dynamic Mesh scenes and batch scenes
