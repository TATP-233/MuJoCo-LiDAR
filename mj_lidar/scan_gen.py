import os
import numpy as np
from functools import lru_cache

class LivoxGenerator:
    """
    生成 Livox 激光雷达的扫描模式
    """
    livox_lidar_params = {
        "avia": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 70.4,
            "vertical_fov": 77.2,
            "samples": 24000
        },
        "HAP": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "samples": 45300,
            "downsample": 1
        },
        "horizon" : {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 81.7,
            "vertical_fov": 25.1,
            "samples": 24000,
        },
        "mid40" : {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 81.7,
            "vertical_fov": 25.1,
            "samples": 24000,
        },
        "mid70" : {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 70.4,
            "vertical_fov": 70.4,
            "samples": 10000,
        },
        "mid360" : {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "samples": 24000,
        },
        "tele" : {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 14.5,
            "vertical_fov": 16.1,
            "samples": 24000,
        }
    }
    def __init__(self, name):
        if name in self.livox_lidar_params:
            self.laser_min_range = self.livox_lidar_params[name]["laser_min_range"]
            self.laser_max_range = self.livox_lidar_params[name]["laser_max_range"]
            self.samples = self.livox_lidar_params[name]["samples"]
            try:
                pattern_npy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_mode", f"{name}.npy")
                self.ray_angles = np.load(pattern_npy_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Scan mode file not found for {name}, file should be saved in {pattern_npy_path}")
            self.n_rays = len(self.ray_angles)
        else:
            raise ValueError(f"Invalid LiDAR name: {name}")
        self.currStartIndex = 0

    def sample_ray_angles(self, downsample=1):
        if self.currStartIndex + self.samples > self.n_rays:
            self.ray_part1 = self.ray_angles[self.currStartIndex:]
            self.ray_part2 = self.ray_angles[:self.samples - len(self.ray_part1)]
            self.currStartIndex = self.samples - len(self.ray_part1)
            self.ray_out = np.concatenate([self.ray_part1, self.ray_part2], axis=0)
        else:
            self.ray_part1 = self.ray_angles[self.currStartIndex:self.currStartIndex + self.samples]
            self.currStartIndex += self.samples
            self.ray_out = self.ray_part1
        if downsample > 1:
            self.ray_out = self.ray_out[::downsample]
        return self.ray_out[:, 0], self.ray_out[:, 1]

# =======================================================================
# 1. Velodyne HDL-64 (任意 360° 旋转式激光雷达)
# https://www.mapix.com/wp-content/uploads/2018/07/63-9194_Rev-J_HDL-64E_S3_Spec-Sheet-Web.pdf
# =======================================================================
def generate_HDL64(     # |参数            | Velodyne HDL-64
    f_rot=10.0,            # |转速 (Hz)       |  5-20Hz
    sample_rate=1.1e6,     # |采样率 (Hz)     | 2.2MHz(双返回模式)
    n_channels=64,         # |垂直通道数       | 64 (Vertical Angular Resolution : 0.4°)
    phi_fov=(-24.9, 2.)    # |垂直视场角 (度)  | (-24.9°, 2.°)
):
    # 转换为弧度
    phi_min, phi_max = np.deg2rad(phi_fov)
    
    # 时间序列（列向量）
    t = np.arange(0, 1./f_rot, n_channels/sample_rate)[:, None]  # shape: (n_times, 1)
    
    # 水平角计算（广播机制）
    theta = (2 * np.pi * f_rot * t) % (2 * np.pi)      # shape: (n_times, 1)
    
    # 垂直角（行向量）
    phi = np.linspace(phi_min, phi_max, n_channels)     # shape: (1, n_channels)
    
    # 生成网格（无需显式使用meshgrid）
    theta_grid = theta + np.zeros((1, n_channels))      # 广播至 (n_times, n_channels)
    phi_grid = np.zeros_like(theta) + phi               # 广播至 (n_times, n_channels)
    
    return theta_grid.flatten(), phi_grid.flatten()

# =======================================================================
# 2. Velodyne VLP-32 模式
# https://www.mapix.com/lidar-scanner-sensors/velodyne/velodyne-vlp-32c/
# =======================================================================
@lru_cache(maxsize=8)
def _get_vlp32_angles():
    """使用缓存获取VLP-32的角度分布，避免重复计算，返回弧度值"""
    vlp32_angles = np.array([
        -25.0, -22.5, -20.0, -15.0, -13.0, -10.0, -5.0, -3.0, 
        -2.333, -1.0, -0.667, -0.333, 0.0, 0.0, 0.333, 0.667, 
        1.0, 1.333, 1.667, 2.0, 2.333, 2.667, 3.0, 3.333, 
        3.667, 4.0, 5.0, 7.0, 10.0, 15.0, 17.0, 20.0
    ])
    # 转换为弧度并裁剪
    vlp32_angles = np.deg2rad(vlp32_angles)
    return vlp32_angles

def generate_vlp32(
    f_rot=10.0,       # 转速 (Hz)
    sample_rate=1.2e6 # 采样率 (Hz)
):
    # 垂直角参数
    phi = _get_vlp32_angles()       # shape: (n_channels,)
    
    # 时间序列（列向量）
    t = np.arange(0, 1/f_rot, 32/sample_rate)[:, None]  # shape: (n_times, 1)
    
    # 水平角计算
    theta = (2 * np.pi * f_rot * t) % (2 * np.pi)      # shape: (n_times, 1)
    
    # 广播生成网格
    theta_grid = theta + np.zeros_like(phi)            # shape: (n_times, n_channels)
    phi_grid = np.zeros_like(theta) + phi              # shape: (n_times, n_channels)
    
    return theta_grid.flatten(), phi_grid.flatten()

# =======================================================================
# 3. Ouster OS-128 模式
# https://www.general-laser.at/en/shop-en/ouster-os0-128-lidar-sensor-en
# =======================================================================
def generate_os128(
    f_rot=20.0,            # 转速 (Hz)
    sample_rate=5.2e6,     # 采样率 (Hz)
):
    # 垂直角参数（均匀分布）
    n_channels = 128
    phi = np.deg2rad(np.linspace(-22.5, 22.5, n_channels))  # shape: (n_channels,)
    
    # 时间序列（列向量）
    t = np.arange(0, 1/f_rot, n_channels/sample_rate)[:, None]     # shape: (n_times, 1)
    
    # 水平角计算
    theta = (2 * np.pi * f_rot * t) % (2 * np.pi)         # shape: (n_times, 1)
    
    # 广播生成网格
    theta_grid = theta + np.zeros_like(phi)               # shape: (n_times, n_channels)
    phi_grid = np.zeros_like(theta) + phi                 # shape: (n_times, n_channels)
    
    return theta_grid.flatten(), phi_grid.flatten()

# =======================================================================
# 4. 生成网格状扫描模式
# =======================================================================
def generate_grid_scan_pattern(num_ray_cols, num_ray_rows, theta_range=(-np.pi, np.pi), phi_range=(-np.pi/3, np.pi/3)):
    """
    生成网格状扫描模式
    
    参数:
        num_ray_cols: 水平方向射线数
        num_ray_rows: 垂直方向射线数
        
    返回:
        (ray_theta, ray_phi): 水平角和垂直角数组
    """
   # 创建网格扫描模式
    theta_grid, phi_grid = np.meshgrid(
        np.linspace(theta_range[0], theta_range[1], num_ray_cols),  # 水平角
        np.linspace(phi_range[0], phi_range[1], num_ray_rows)  # 垂直角
    )
    
    # 展平网格为一维数组
    ray_phi = phi_grid.flatten()
    ray_theta = theta_grid.flatten()
    
    # 打印扫描范围信息
    print(f"扫描模式：phi范围[{ray_phi.min():.2f}, {ray_phi.max():.2f}], theta范围[{ray_theta.min():.2f}, {ray_theta.max():.2f}]")
    return ray_theta, ray_phi

# =======================================================================
# 5. 创建激光雷达扫描线的角度数组，仅包含水平方向
# =======================================================================
def create_lidar_single_line(horizontal_resolution=360, horizontal_fov=2*np.pi):
    """创建激光雷达扫描线的角度数组，仅包含水平方向"""
    h_angles = np.linspace(-horizontal_fov/2, horizontal_fov/2, horizontal_resolution)
    v_angles = np.zeros_like(h_angles)
    return h_angles, v_angles
