# MuJoCo-LiDAR: High-Performance LiDAR Simulation

[![PyPI](https://img.shields.io/pypi/v/mujoco-lidar)](https://pypi.org/project/mujoco-lidar/)
[![Python](https://img.shields.io/pypi/pyversions/mujoco-lidar)](https://pypi.org/project/mujoco-lidar/)

High-performance LiDAR simulation for MuJoCo with CPU, Taichi, JAX, and Warp backends.

<p align="center">
  <img src="./assets/go2.png" width="49%" />
  <img src="./assets/g1.png" width="49%" />
</p>
<p align="center">
  <img src="./assets/g1_native.png" width="32%" />
  <img src="./assets/go2_native.png" width="32%" />
  <img src="./assets/lidar_rviz.png" width="33%" />
</p>

[中文文档](README_zh.md) | [Installation](docs/en/INSTALLATION.md) | [Usage Guide](docs/en/USAGE.md) | [Development](docs/en/DEVELOPMENT.md) | [Contributing](CONTRIBUTING.md)

## Features

- **Multi-Backend Support**:
  - **CPU**: MuJoCo native `mj_multiRay`, no GPU required
  - **Taichi**: GPU parallel computing, supports Mesh and Hfield
  - **JAX**: GPU + MJX integration, batch simulation support
  - **Warp**: NVIDIA Warp ray casting, supports dynamic mesh scenes and batched scenes
- **High Performance**: 1M+ rays/sec on GPU, real-time BVH construction
- **Multiple LiDAR Models**: Velodyne (HDL-64E, VLP-32C), Livox (mid360, avia), Ouster (OS-128), custom patterns
- **ROS Integration**: Ready-to-use ROS1/ROS2 examples

## Quick Start

### Installation

**From PyPI:**

```bash
# Basic (CPU only)
uv add mujoco-lidar

# With Taichi backend (GPU)
uv add "mujoco-lidar[taichi]"

# With JAX backend (GPU + batch)
uv add "mujoco-lidar[jax]"

# With Warp backend (GPU + dynamic mesh + batch)
uv add "mujoco-lidar[warp]"
```

**From Source:**

```bash
git clone https://github.com/TATP-233/MuJoCo-LiDAR.git
cd MuJoCo-LiDAR

uv sync --extra dev --extra examples

# Optional GPU backends
uv sync --extra dev --extra examples --extra taichi
uv sync --extra dev --extra examples --extra warp
uv sync --extra dev --extra examples --extra taichi --extra jax --extra warp
```

Run a non-ROS example after installing from source:

```bash
# Native MuJoCo viewer, CPU backend
uv run --extra dev --extra examples python examples/example_native.py --backend cpu

# Same example with the Warp backend, if installed
uv run --extra dev --extra examples --extra warp python examples/example_native.py --backend warp

# Unitree Go2, no ROS required
uv run --extra dev --extra examples --extra warp python examples/unitree_go2.py --backend warp --stand
```

See [Installation Guide](docs/en/INSTALLATION.md) for details.

### Basic Usage

```python
import mujoco
from mujoco_lidar import MjLidarWrapper, scan_gen

# Load model
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# Create LiDAR
lidar = MjLidarWrapper(
    model,
    site_name="lidar_site",
    backend="cpu",  # or "taichi", "jax", "warp"
    cutoff_dist=50.0
)

# Generate scan pattern
theta, phi = scan_gen.generate_HDL64()

# Trace rays
ranges = lidar.trace_rays(data, theta, phi)
```

See [Usage Guide](docs/en/USAGE.md) for more examples.

## Performance

| Backend | Rays/sec | Hardware | Batch Support |
|---------|----------|----------|---------------|
| CPU     | ~9M      | Native   | No            |
| Taichi  | ~62M     | GPU      | Yes           |
| JAX     | ~231M    | GPU      | Yes           |
| Warp    | ~100M    | GPU      | Yes           |

Run benchmarks: `make benchmark`

## Documentation

- [Installation Guide](docs/en/INSTALLATION.md) - Detailed installation instructions
- [Usage Guide](docs/en/USAGE.md) - Examples and tutorials
- [API Reference](docs/en/API.md) - Complete API documentation
- [Development Guide](docs/en/DEVELOPMENT.md) - Contributing and testing
- [Project Structure](docs/en/PROJECT_STRUCTURE.md) - Codebase organization

## Examples

- [examples/example_native.py](examples/example_native.py) - MuJoCo viewer; supports `--backend cpu|taichi|jax|warp`
- [examples/example_mjcf.py](examples/example_mjcf.py) - MJCF scene with Matplotlib point-cloud view
- [examples/example_string.py](examples/example_string.py) - Self-contained XML string scene
- [examples/unitree_go2.py](examples/unitree_go2.py) - Unitree Go2 without ROS; supports `--backend`
- [examples/unitree_g1.py](examples/unitree_g1.py) - Unitree G1 without ROS; supports `--backend`

## Development

```bash
git clone https://github.com/TATP-233/MuJoCo-LiDAR.git
cd MuJoCo-LiDAR
uv sync --extra dev

make test      # Run tests
make lint      # Check code quality
make benchmark # Run performance tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@article{jia2025discoverse,
      title={DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments},
      author={Yufei Jia and Guangyu Wang and Yuhang Dong and Junzhe Wu and Yupei Zeng and Haonan Lin and Zifan Wang and Haizhou Ge and Weibin Gu and Chuxuan Li and Ziming Wang and Yunjie Cheng and Wei Sui and Ruqi Huang and Guyue Zhou},
      journal={arXiv preprint arXiv:2507.21981},
      year={2025},
      url={https://arxiv.org/abs/2507.21981}
}
```
