import time
import mujoco
import numpy as np
import taichi as ti

ti.init(
    arch=ti.gpu, 
    kernel_profiler=True,
    advanced_optimization=True,  # 启用高级优化
    offline_cache=True,          # 启用离线缓存
    default_fp=ti.f32,           # 设置默认浮点类型
    default_ip=ti.i32,           # 设置默认整数类型
    device_memory_GB=4.0,        # 限制设备内存使用
)

@ti.data_oriented
class MjLidarSensor:

    def __init__(self, mj_scene: mujoco.MjvScene, enable_profiling=False, verbose=False):
        """
        初始化LiDAR传感器
        
        参数:
            mj_scene: MuJoCo场景对象
            enable_profiling: 是否启用性能分析（默认False）
            verbose: 是否打印详细信息（默认False）
        """
        self.n_geoms = mj_scene.ngeom
        self.enable_profiling = enable_profiling
        self.verbose = verbose
        
        if self.verbose:
            print(f"n_geoms: {self.n_geoms}")

        # 预分配所有Taichi字段，避免重复创建
        self.geom_types = ti.field(dtype=ti.i32, shape=(self.n_geoms))
        self.geom_sizes = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_geoms))
        self.geom_positions = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_geoms))
        self.geom_rotations = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.n_geoms))  # 修改为矩阵字段
        
        # 初始化几何体静态数据
        for i in range(self.n_geoms):
            geom = mj_scene.geoms[i]
            if geom.objtype != 5: # 5 is mjOBJ_GEOM
                self.geom_types[i] = -1
            else:
                self.geom_types[i] = geom.type
            self.geom_sizes[i] = ti.math.vec3(geom.size[0], geom.size[1], geom.size[2])
            self.geom_positions[i] = ti.math.vec3(geom.pos[0], geom.pos[1], geom.pos[2])
            # 保存旋转矩阵
            rot_mat = geom.mat.reshape(3, 3)
            for r in range(3):
                for c in range(3):
                    self.geom_rotations[i][r, c] = rot_mat[r, c]
        
        # 预先分配传感器位姿数组
        self.sensor_pose_ti = ti.ndarray(dtype=ti.f32, shape=(4, 4))
        
        # 缓存射线数据
        self.cached_n_rays = 0
        self.rays_phi_ti = None
        self.rays_theta_ti = None
        self.hit_points = None
        
        # 预分配临时数组（用于内核计算）
        self.hit_points_world = None  # 世界坐标系下的命中点
        self.hit_mask = None  # 射线命中标志
        
        # 性能统计
        self.kernel_time = 0
        self.prepare_time = 0
        self.total_time = 0
        
        # 详细性能统计
        self.update_geom_time = 0
        self.convert_sensor_pose_time = 0
        self.update_rays_time = 0
        self.memory_allocation_time = 0
        self.sync_time = 0

    def update_geom_positions(self, mj_scene:mujoco.MjvScene):
        """更新几何体位置和旋转数据"""
        start_time = time.time() if self.enable_profiling else 0
        
        # 预先分配NumPy数组以收集所有数据
        pos_data = np.zeros((self.n_geoms, 3), dtype=np.float32)
        rot_data = np.zeros((self.n_geoms, 3, 3), dtype=np.float32)
        
        # 在CPU上收集数据
        for i in range(self.n_geoms):
            geom = mj_scene.geoms[i]
            pos_data[i] = geom.pos
            rot_data[i] = geom.mat
        
        # 使用Taichi内核并行更新
        self.update_geom_positions_parallel(pos_data, rot_data)
        
        end_time = time.time() if self.enable_profiling else 0
        self.update_geom_time = (end_time - start_time) * 1000 if self.enable_profiling else 0

    @ti.kernel
    def update_geom_positions_parallel(self, 
                                       pos_data: ti.types.ndarray(dtype=ti.f32, ndim=2), 
                                       rot_data: ti.types.ndarray(dtype=ti.f32, ndim=3)):
        """并行更新几何体位置和旋转数据"""
        # 并行遍历所有几何体
        for i in range(self.n_geoms):
            # 更新位置
            self.geom_positions[i] = ti.math.vec3(pos_data[i, 0], pos_data[i, 1], pos_data[i, 2])
            
            # 更新旋转矩阵
            for r in range(3):
                for c in range(3):
                    self.geom_rotations[i][r, c] = rot_data[i, r, c]

    @ti.func
    def transform_ray_to_local(self, ray_start, ray_direction, center, rotation):
        """将射线从世界坐标系转换到物体的局部坐标系"""
        # 先平移射线起点
        local_start = ray_start - center
        
        # 旋转矩阵的转置是其逆（假设正交矩阵）
        rot_transpose = ti.Matrix.zero(ti.f32, 3, 3)
        for i in range(3):
            for j in range(3):
                rot_transpose[i, j] = rotation[j, i]
        
        # 应用旋转
        local_start = rot_transpose @ local_start
        local_direction = rot_transpose @ ray_direction
        
        return local_start, local_direction
    
    @ti.func
    def transform_point_to_world(self, local_point, center, rotation):
        """将点从局部坐标系转换回世界坐标系"""
        # 应用旋转
        world_point = rotation @ local_point
        # 应用平移
        world_point = world_point + center
        return world_point
    
    @ti.func
    def ray_plane_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
        """计算射线与平面的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        
        # 转换射线到平面的局部坐标系
        local_start, local_direction = self.transform_ray_to_local(ray_start, ray_direction, center, rotation)
        
        # 在局部坐标系中，平面的法向量是z轴
        normal = ti.math.vec3(0.0, 0.0, 1.0)
        half_width = size[0]
        half_height = size[1]
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        denom = local_direction.dot(normal)
        
        # 避免除以零，检查光线是否与平面平行
        if ti.abs(denom) >= 1e-6:
            # 局部坐标系中平面在原点，所以只需要计算到原点的距离
            t = -local_start.z / denom
            
            # 如果t为正，表示有有效交点
            if t >= 0:
                local_hit = local_start + t * local_direction
                
                # 检查交点是否在平面范围内
                if ti.abs(local_hit.x) <= half_width and ti.abs(local_hit.y) <= half_height:
                    # 将交点转换回世界坐标系
                    world_hit = self.transform_point_to_world(local_hit, center, rotation)
                    hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
        
        return hit_result
    
    @ti.func
    def ray_sphere_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
        """计算射线与球体的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        # 注意：球体旋转不会改变其形状，所以可以简化计算
        radius = size[0]
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        oc = ray_start - center
        a = ray_direction.dot(ray_direction)
        b = 2.0 * oc.dot(ray_direction)
        c = oc.dot(oc) - radius * radius
        
        discriminant = b * b - 4 * a * c
        
        # 计算交点
        if discriminant >= 0:
            t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
            
            # 如果t为负，则使用较大的t值
            if t < 0:
                t = (-b + ti.sqrt(discriminant)) / (2.0 * a)
            
            # 如果t为正，表示有有效交点
            if t >= 0:
                hit_pos = ray_start + t * ray_direction
                hit_result = ti.math.vec4(hit_pos.x, hit_pos.y, hit_pos.z, t)
        
        return hit_result
    
    @ti.func
    def ray_box_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
        """计算射线与盒子的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        
        # 转换射线到盒子的局部坐标系
        local_start, local_direction = self.transform_ray_to_local(ray_start, ray_direction, center, rotation)
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        
        # 处理局部坐标系中的射线方向为零的情况
        inv_dir = ti.math.vec3(
            1.0 / (local_direction.x if ti.abs(local_direction.x) > 1e-6 else 1e10),
            1.0 / (local_direction.y if ti.abs(local_direction.y) > 1e-6 else 1e10),
            1.0 / (local_direction.z if ti.abs(local_direction.z) > 1e-6 else 1e10)
        )
        
        t_min = -1e10  # 使用大数而不是无穷
        t_max = 1e10
        
        # 检查x轴
        t1 = (-size.x - local_start.x) * inv_dir.x
        t2 = (size.x - local_start.x) * inv_dir.x
        t_min = ti.max(t_min, ti.min(t1, t2))
        t_max = ti.min(t_max, ti.max(t1, t2))
        
        # 检查y轴
        t1 = (-size.y - local_start.y) * inv_dir.y
        t2 = (size.y - local_start.y) * inv_dir.y
        t_min = ti.max(t_min, ti.min(t1, t2))
        t_max = ti.min(t_max, ti.max(t1, t2))
        
        # 检查z轴
        t1 = (-size.z - local_start.z) * inv_dir.z
        t2 = (size.z - local_start.z) * inv_dir.z
        t_min = ti.max(t_min, ti.min(t1, t2))
        t_max = ti.min(t_max, ti.max(t1, t2))
        
        # 如果有有效的交点
        if t_max >= t_min and t_max >= 0:
            t = t_min if t_min >= 0 else t_max
            if t >= 0:
                # 计算局部坐标系中的交点
                local_hit = local_start + t * local_direction
                # 转换回世界坐标系
                world_hit = self.transform_point_to_world(local_hit, center, rotation)
                hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
        
        return hit_result
    
    @ti.func
    def ray_cylinder_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
        """计算射线与圆柱体的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        # size[0]是半径，size[1]是半高
        
        # 转换射线到圆柱体的局部坐标系
        local_start, local_direction = self.transform_ray_to_local(ray_start, ray_direction, center, rotation)
        
        radius = size[0]
        half_height = size[2]
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        
        # 在局部坐标系中，圆柱体的中心轴与z轴平行
        # 仅考虑xy平面上的方向分量
        ray_dir_xy = ti.math.vec2(local_direction.x, local_direction.y)
        oc_xy = ti.math.vec2(local_start.x, local_start.y)
        
        # 解二次方程 at² + bt + c = 0
        a = ray_dir_xy.dot(ray_dir_xy)
        
        # 如果a很小，射线几乎与z轴平行
        if a < 1e-6:
            # 检查射线是否在圆柱体内部
            if oc_xy.norm() <= radius:
                # 计算与顶部或底部平面的交点
                t1 = (half_height - local_start.z) / local_direction.z
                t2 = (-half_height - local_start.z) / local_direction.z
                
                # 选择最小的正t值
                t = -1.0  # 默认为无效值
                if t1 >= 0 and (t2 < 0 or t1 < t2):
                    t = t1
                elif t2 >= 0:
                    t = t2
                
                if t >= 0:
                    local_hit = local_start + t * local_direction
                    world_hit = self.transform_point_to_world(local_hit, center, rotation)
                    hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
        else:
            # 标准的圆柱体-射线相交测试
            b = 2.0 * oc_xy.dot(ray_dir_xy)
            c = oc_xy.dot(oc_xy) - radius * radius
            
            discriminant = b * b - 4 * a * c
            
            if discriminant >= 0:
                # 计算圆柱侧面的两个可能交点
                sqrt_disc = ti.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2.0 * a)
                t2 = (-b + sqrt_disc) / (2.0 * a)
                
                # 选择最小的正t值
                t = -1.0  # 默认为无效值
                if t1 >= 0:
                    t = t1
                elif t2 >= 0:
                    t = t2
                
                # 检查交点是否在圆柱体高度范围内
                if t >= 0:
                    local_hit = local_start + t * local_direction
                    
                    if ti.abs(local_hit.z) <= half_height:
                        # 交点在圆柱体侧面上
                        world_hit = self.transform_point_to_world(local_hit, center, rotation)
                        hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
                    else:
                        # 侧面交点不在圆柱体高度范围内，检查与顶部或底部平面的交点
                        cap_t = -1.0
                        
                        # 射线从上方射向底平面
                        if local_direction.z < 0 and local_start.z > half_height:
                            cap_t = (half_height - local_start.z) / local_direction.z
                        # 射线从下方射向顶平面
                        elif local_direction.z > 0 and local_start.z < -half_height:
                            cap_t = (-half_height - local_start.z) / local_direction.z
                        
                        if cap_t >= 0:
                            local_hit = local_start + cap_t * local_direction
                            cap_xy = ti.math.vec2(local_hit.x, local_hit.y)
                            
                            # 检查交点是否在圆盘内
                            if cap_xy.norm() <= radius:
                                world_hit = self.transform_point_to_world(local_hit, center, rotation)
                                hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, cap_t)
        
        return hit_result
    
    @ti.func
    def ray_ellipsoid_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
        """计算射线与椭球体的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        
        # 转换射线到椭球体的局部坐标系
        local_start, local_direction = self.transform_ray_to_local(ray_start, ray_direction, center, rotation)
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        
        # 将问题转换为单位球相交，通过缩放空间
        inv_size = ti.math.vec3(1.0/size.x, 1.0/size.y, 1.0/size.z)
        
        # 缩放局部坐标系中的射线（不要归一化方向向量，这会改变t的意义）
        scaled_start = ti.math.vec3(
            local_start.x * inv_size.x,
            local_start.y * inv_size.y,
            local_start.z * inv_size.z
        )
        scaled_dir = ti.math.vec3(
            local_direction.x * inv_size.x,
            local_direction.y * inv_size.y,
            local_direction.z * inv_size.z
        )
        
        # 解二次方程 at² + bt + c = 0
        a = scaled_dir.dot(scaled_dir)
        b = 2.0 * scaled_start.dot(scaled_dir)
        c = scaled_start.dot(scaled_start) - 1.0  # 单位球半径为1
        
        discriminant = b * b - 4 * a * c
        
        if discriminant >= 0:
            # 计算两个可能的t值，取最小的正值
            t1 = (-b - ti.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + ti.sqrt(discriminant)) / (2.0 * a)
            
            t = t1 if t1 >= 0 else t2
            
            # 如果t为正，表示有有效交点
            if t >= 0:
                # 使用原始射线方程计算交点
                local_hit = local_start + t * local_direction
                
                # 转换回世界坐标系
                world_hit = self.transform_point_to_world(local_hit, center, rotation)
                hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
        
        return hit_result
    
    @ti.func
    def ray_capsule_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
        """计算射线与胶囊体的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        # 在MuJoCo中: size[0]是半径，size[1]是圆柱部分的半高
        
        # 转换射线到胶囊体的局部坐标系
        local_start, local_direction = self.transform_ray_to_local(ray_start, ray_direction, center, rotation)
        
        radius = size[0]
        half_height = size[2]
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        
        # 计算胶囊体两个半球的中心（在局部坐标系中）
        # 半球中心点在距离胶囊体中心half_height处
        sphere1_center = ti.math.vec3(0.0, 0.0, half_height)
        sphere2_center = ti.math.vec3(0.0, 0.0, -half_height)
        
        # 为圆柱部分创建新的size
        cylinder_size = ti.math.vec3(radius, radius, half_height)
        identity_mat = ti.Matrix.identity(ti.f32, 3)  # 局部坐标系中用单位矩阵
        
        # 首先检查与圆柱体部分的交点（在局部坐标系中）
        cylinder_hit = self.ray_cylinder_intersection(local_start, local_direction, ti.math.vec3(0.0, 0.0, 0.0), cylinder_size, identity_mat)
        
        # 初始化最小距离为无穷大
        min_t = 1e10
        has_hit = False
        
        # 如果有圆柱体交点
        if cylinder_hit.w > 0 and cylinder_hit.w < min_t:
            min_t = cylinder_hit.w
            
            # 计算世界坐标系中的交点
            local_hit = local_start + cylinder_hit.w * local_direction
            world_hit = self.transform_point_to_world(local_hit, center, rotation)
            hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, min_t)
            has_hit = True
        
        # 然后检查与两个半球的交点
        sphere_size = ti.math.vec3(radius, radius, radius)
        
        # 上半球
        sphere1_hit = self.ray_sphere_intersection(local_start, local_direction, sphere1_center, sphere_size, identity_mat)
        if sphere1_hit.w > 0 and sphere1_hit.w < min_t:
            # 确保交点在半球内，而不是在完整球体的下半部分
            local_hit = local_start + sphere1_hit.w * local_direction
            local_z = local_hit.z - sphere1_center.z
            if local_z >= 0:  # 只取上半部分
                min_t = sphere1_hit.w
                world_hit = self.transform_point_to_world(local_hit, center, rotation)
                hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, min_t)
                has_hit = True
        
        # 下半球
        sphere2_hit = self.ray_sphere_intersection(local_start, local_direction, sphere2_center, sphere_size, identity_mat)
        if sphere2_hit.w > 0 and sphere2_hit.w < min_t:
            # 确保交点在半球内，而不是在完整球体的上半部分
            local_hit = local_start + sphere2_hit.w * local_direction
            local_z = local_hit.z - sphere2_center.z
            if local_z <= 0:  # 只取下半部分
                min_t = sphere2_hit.w
                world_hit = self.transform_point_to_world(local_hit, center, rotation)
                hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, min_t)
                has_hit = True
        
        # 如果没有任何交点，返回无效结果
        if not has_hit:
            hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        
        return hit_result
    
    # 优化的Taichi核函数
    @ti.kernel
    def trace_rays(self, 
        sensor_origins: ti.types.ndarray(dtype=ti.f32, ndim=2), 
        rays_phi: ti.types.ndarray(dtype=ti.f32, ndim=1),
        rays_theta: ti.types.ndarray(dtype=ti.f32, ndim=1),
        n_rays: ti.i32,
        hit_points: ti.template(),
    ):
        # 设置LiDAR传感器位置和姿态（只做一次）
        sensor_pose = ti.Matrix.identity(ti.f32, 4)
        for i in range(4):
            for j in range(4):
                sensor_pose[i, j] = sensor_origins[i, j]
        ray_start = sensor_pose[0:3, 3]  # 射线起点

        # 计算传感器位姿的逆矩阵（只计算一次）
        sensor_pose_inv = ti.Matrix.identity(ti.f32, 4)
        # 旋转部分的转置（因为正交矩阵的逆等于其转置）
        for i in range(3):
            for j in range(3):
                sensor_pose_inv[i, j] = sensor_pose[j, i]
        # 平移部分的变换
        for i in range(3):
            sensor_pose_inv[i, 3] = 0.0
            for j in range(3):
                sensor_pose_inv[i, 3] -= sensor_pose_inv[i, j] * sensor_pose[j, 3]

        # 使用预分配的字段，而不是在内核中创建新字段
        self.hit_points_world.fill(ti.Vector([0.0, 0.0, 0.0]))
        hit_points.fill(ti.Vector([0.0, 0.0, 0.0]))
        self.hit_mask.fill(0)
                
        # 为每条射线并行计算
        ti.loop_config(block_dim=512)
        for i in range(n_rays):
            min_distance = 1e10

            # 计算射线方向（球坐标系转笛卡尔坐标系）
            phi = rays_phi[i]      # 垂直角度
            theta = rays_theta[i]  # 水平角度
            
            # 预计算三角函数值
            cos_theta = ti.cos(theta)
            sin_theta = ti.sin(theta)
            cos_phi = ti.cos(phi)
            sin_phi = ti.sin(phi)

            dir_local = ti.Vector([
                cos_phi * cos_theta,  # x分量
                cos_phi * sin_theta,  # y分量
                sin_phi               # z分量
            ]).normalized()  # 单位化方向向量

            ray_direction = (sensor_pose @ ti.Vector([dir_local.x, dir_local.y, dir_local.z, 0.0])).xyz.normalized()

            # 检查与每个几何体的交点
            for j in range(self.n_geoms):
                hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
                
                # 获取几何体数据
                geom_type = self.geom_types[j]
                center = self.geom_positions[j]
                size = self.geom_sizes[j]
                rotation = self.geom_rotations[j]
                
                # 根据几何体类型调用相应的交点计算函数
                if geom_type == 0:  # PLANE
                    hit_result = self.ray_plane_intersection(ray_start, ray_direction, center, size, rotation)
                elif geom_type == 2:  # SPHERE
                    hit_result = self.ray_sphere_intersection(ray_start, ray_direction, center, size, rotation)
                elif geom_type == 3:  # CAPSULE
                    hit_result = self.ray_capsule_intersection(ray_start, ray_direction, center, size, rotation)
                elif geom_type == 4:  # ELLIPSOID
                    hit_result = self.ray_ellipsoid_intersection(ray_start, ray_direction, center, size, rotation)
                elif geom_type == 5:  # CYLINDER
                    hit_result = self.ray_cylinder_intersection(ray_start, ray_direction, center, size, rotation)
                elif geom_type == 6:  # BOX
                    hit_result = self.ray_box_intersection(ray_start, ray_direction, center, size, rotation)
                # 暂不支持HFIELD(1)

                # 检查是否有有效交点，并且是否是最近的
                if hit_result.w > 0 and hit_result.w < min_distance:
                    # 记录世界坐标系中的最近交点
                    self.hit_points_world[i] = ti.math.vec3(hit_result.x, hit_result.y, hit_result.z)
                    min_distance = hit_result.w
                    self.hit_mask[i] = 1  # 标记此射线有命中

            # 在遍历完当前射线的所有几何体后，进行坐标转换
            if self.hit_mask[i] == 1:  # 如果有命中
                world_hit = self.hit_points_world[i]
                # 转换为齐次坐标并应用传感器位姿的逆矩阵
                local_hit = sensor_pose_inv @ ti.Vector([world_hit.x, world_hit.y, world_hit.z, 1.0])
                hit_points[i] = local_hit.xyz

    def ray_cast_taichi(self, rays_phi, rays_theta, sensor_pose, mj_scene):
        """
        使用Taichi进行真正的并行光线追踪
        Params:
            rays_phi: 垂直角度数组
            rays_theta: 水平角度数组
            sensor_pose: 4x4 matrix, the pose of the sensor in the world frame
            mj_scene: mujoco.MjvScene object, the scene to cast rays into
        Return:
            Nx3 matrix, each row is the intersection point of the corresponding ray in the sensor frame
        """
        assert rays_phi.shape == rays_theta.shape, "rays_phi和rays_theta的形状必须相同"
        n_rays = rays_phi.shape[0]
        
        # 性能计时
        start_total = time.time()
        
        # 确保sensor_pose是float32类型
        convert_start = time.time() if self.enable_profiling else 0
        sensor_pose = sensor_pose.astype(np.float32)
        
        # 创建Taichi ndarray并从NumPy数组填充
        self.sensor_pose_ti.from_numpy(sensor_pose)
        convert_end = time.time() if self.enable_profiling else 0
        self.convert_sensor_pose_time = (convert_end - convert_start) * 1000 if self.enable_profiling else 0
        
        # 如果光线数量变化，重新分配内存
        memory_start = time.time() if self.enable_profiling else 0
        if self.cached_n_rays != n_rays:
            self.rays_phi_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
            self.rays_theta_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
            self.hit_points = ti.Vector.field(3, dtype=ti.f32, shape=n_rays)
            # 同时创建临时字段
            self.hit_points_world = ti.Vector.field(3, dtype=ti.f32, shape=n_rays)
            self.hit_mask = ti.field(dtype=ti.i32, shape=n_rays)
            self.cached_n_rays = n_rays
        memory_end = time.time() if self.enable_profiling else 0
        self.memory_allocation_time = (memory_end - memory_start) * 1000 if self.enable_profiling else 0

        # 更新光线数据
        rays_start = time.time() if self.enable_profiling else 0
        self.rays_phi_ti.from_numpy(rays_phi.astype(np.float32))
        self.rays_theta_ti.from_numpy(rays_theta.astype(np.float32))
        rays_end = time.time() if self.enable_profiling else 0
        self.update_rays_time = (rays_end - rays_start) * 1000 if self.enable_profiling else 0
        
        # 更新几何体位置
        self.update_geom_positions(mj_scene)
        
        # 准备阶段结束，记录时间
        prepare_end = time.time() if self.enable_profiling else 0
        self.prepare_time = (prepare_end - start_total) * 1000 if self.enable_profiling else 0
        
        # 开始Taichi内核计算
        sync_start = time.time() if self.enable_profiling else 0
        ti.sync()  # 确保之前的操作完成
        sync_end = time.time() if self.enable_profiling else 0
        self.sync_time = (sync_end - sync_start) * 1000 if self.enable_profiling else 0
        
        kernel_start = time.time() if self.enable_profiling else 0
        
        # 调用Taichi内核
        self.trace_rays(
            self.sensor_pose_ti,
            self.rays_phi_ti,
            self.rays_theta_ti,
            n_rays,
            self.hit_points
        )
        
        # 等待内核完成
        ti.sync()
        kernel_end = time.time() if self.enable_profiling else 0
        self.kernel_time = (kernel_end - kernel_start) * 1000 if self.enable_profiling else 0
        
        # 结果已经在内核中转换为局部坐标系
        result = self.hit_points.to_numpy()
        
        # 计算总时间
        end_total = time.time()
        self.total_time = (end_total - start_total) * 1000 if self.enable_profiling else 0
        
        # 打印详细性能信息
        if self.enable_profiling and self.verbose:
            print(f"准备阶段性能分析:")
            print(f"  - 传感器位姿转换时间: {self.convert_sensor_pose_time:.2f}ms")
            print(f"  - 内存分配时间: {self.memory_allocation_time:.2f}ms")
            print(f"  - 光线数据更新时间: {self.update_rays_time:.2f}ms")
            print(f"  - 几何体位置更新时间: {self.update_geom_time:.2f}ms")
            print(f"  - 同步操作时间: {self.sync_time:.2f}ms")
            print(f"总计: 准备时间: {self.prepare_time:.2f}ms, 内核时间: {self.kernel_time:.2f}ms, 总时间: {self.total_time:.2f}ms")
        
        return result
