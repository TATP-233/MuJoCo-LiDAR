import argparse

import mujoco
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt

from mujoco_lidar.core import MjLidarSensor
from mujoco_lidar.scan_gen import generate_grid_scan_pattern

from mujoco_lidar.mj_lidar_utils import create_demo_scene

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MuJoCo LiDAR传感器演示和性能测试')
    parser.add_argument('--profiling', action='store_true', help='启用性能分析')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出')
    parser.add_argument('--skip-test', action='store_true', help='跳过性能测试')
    parser.add_argument('--zh', action='store_true', help='使用中文标签')
    parser.add_argument('--save-fig', action='store_true', help='保存图表')
    args = parser.parse_args()

    use_english_labels = not args.zh
    if not use_english_labels:
        # 设置matplotlib支持中文显示
        plt.rcParams['font.sans-serif'] = 'simhei'  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 创建MuJoCo场景
    mj_model, mj_data = create_demo_scene()
    
    # 创建场景渲染对象
    scene = mujoco.MjvScene(mj_model, maxgeom=100)
    # 更新模拟
    mujoco.mj_forward(mj_model, mj_data)
    
    # 更新场景
    mujoco.mjv_updateScene(
        mj_model, mj_data, mujoco.MjvOption(), 
        None, mujoco.MjvCamera(), 
        mujoco.mjtCatBit.mjCAT_ALL.value, scene
    )

    # 创建激光雷达传感器
    lidar = MjLidarSensor(scene, enable_profiling=args.profiling, verbose=args.verbose)

    # 设置激光雷达传感器位姿
    lidar_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # 构建激光雷达位姿矩阵
    lidar_pose = np.eye(4, dtype=np.float32)
    lidar_pose[:3, 3] = lidar_position
    
    # 更新模拟
    mujoco.mj_step(mj_model, mj_data)
    
    # 更新场景
    mujoco.mjv_updateScene(
        mj_model, mj_data, mujoco.MjvOption(), 
        None, mujoco.MjvCamera(), 
        mujoco.mjtCatBit.mjCAT_ALL.value, scene
    )
    
    # 是否跳过性能测试
    if not args.skip_test and args.profiling:
        print("=" * 50)
        print("性能测试 - 不同射线数量")
        print("=" * 50)
        
        # 测试不同射线数量的性能
        ray_counts = [1000, 5000, 10000, 20000, 50000, 100000]
        prepare_times = []
        kernel_times = []
        update_geom_times = []
        # 添加准备阶段各操作的时间收集列表
        sensor_pose_times = []
        memory_alloc_times = []
        rays_update_times = []
        sync_times = []
        
        for count in ray_counts:
            print(f"\n测试射线数量: {count}")
            
            # 创建测试用的射线
            test_rays_phi, test_rays_theta = generate_grid_scan_pattern(
                num_ray_cols=int(np.sqrt(count) * 10), 
                num_ray_rows=int(np.sqrt(count) / 10)
            )

            # 优化内存布局
            test_rays_phi = np.ascontiguousarray(test_rays_phi[:count])
            test_rays_theta = np.ascontiguousarray(test_rays_theta[:count])
            
            # 执行多次测试取平均值
            n_tests = 5
            prepare_time_sum = 0
            kernel_time_sum = 0
            update_geom_time_sum = 0
            # 新增准备阶段各操作时间累加变量
            sensor_pose_time_sum = 0
            memory_alloc_time_sum = 0
            rays_update_time_sum = 0
            sync_time_sum = 0
            
            points = lidar.ray_cast_taichi(test_rays_phi, test_rays_theta, lidar_pose, scene)
            ti.sync()
            for i in range(n_tests):
                # 执行光线追踪
                points = lidar.ray_cast_taichi(test_rays_phi, test_rays_theta, lidar_pose, scene)
                ti.sync()
               
                # 累加时间
                prepare_time_sum += lidar.prepare_time
                kernel_time_sum += lidar.kernel_time
                update_geom_time_sum += lidar.update_geom_time
                # 累加准备阶段各操作时间
                sensor_pose_time_sum += lidar.convert_sensor_pose_time
                memory_alloc_time_sum += lidar.memory_allocation_time
                rays_update_time_sum += lidar.update_rays_time
                sync_time_sum += lidar.sync_time
            
            # 计算平均时间
            avg_prepare_time = prepare_time_sum / n_tests
            avg_kernel_time = kernel_time_sum / n_tests
            avg_update_geom_time = update_geom_time_sum / n_tests
            # 计算准备阶段各操作的平均时间
            avg_sensor_pose_time = sensor_pose_time_sum / n_tests
            avg_memory_alloc_time = memory_alloc_time_sum / n_tests
            avg_rays_update_time = rays_update_time_sum / n_tests
            avg_sync_time = sync_time_sum / n_tests
            
            prepare_times.append(avg_prepare_time)
            kernel_times.append(avg_kernel_time)
            update_geom_times.append(avg_update_geom_time)
            # 保存准备阶段各操作的平均时间
            sensor_pose_times.append(avg_sensor_pose_time)
            memory_alloc_times.append(avg_memory_alloc_time)
            rays_update_times.append(avg_rays_update_time)
            sync_times.append(avg_sync_time)
            
            print(f"平均准备时间: {avg_prepare_time:.2f}ms")
            print(f"平均内核时间: {avg_kernel_time:.2f}ms")
            print(f"平均几何体更新时间: {avg_update_geom_time:.2f}ms")
        
        print("=" * 50)
        print("性能测试 - 不同几何体数量")
        print("=" * 50)
        
        # 测试不同几何体数量的性能
        # 创建包含更多几何体的场景
        num_geoms = [10, 20, 50, 100, 200, 500]
        geom_prepare_times = []
        geom_kernel_times = []
        geom_update_times = []
        # 添加各操作时间的收集列表
        geom_sensor_pose_times = []
        geom_memory_alloc_times = []
        geom_rays_update_times = []
        geom_sync_times = []
        
        # 使用固定数量的射线
        test_rays_phi, test_rays_theta = generate_grid_scan_pattern(num_ray_cols=1800, num_ray_rows=64)
        # 优化内存布局
        test_rays_phi = np.ascontiguousarray(test_rays_phi)
        test_rays_theta = np.ascontiguousarray(test_rays_theta)

        test_rays_count = len(test_rays_phi)  # 记录测试用的射线数量
        
        for num_geom in num_geoms:
            print(f"\n测试几何体数量: {num_geom}")
            
            # 创建一个包含指定数量几何体的场景
            xml_header = """
            <mujoco>
              <worldbody>
                <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
                <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.9 0.9 0.9 1"/>
            """
            
            xml_footer = """
              </worldbody>
            </mujoco>
            """
            
            xml_content = xml_header
            
            # 添加指定数量的几何体
            for i in range(num_geom):
                geom_type = i % 5  # 0=box, 1=sphere, 2=capsule, 3=ellipsoid, 4=cylinder
                x = (i % 10) - 5
                y = (i // 10) - 5
                z = 0.5
                
                if geom_type == 0:  # box
                    xml_content += f'<geom name="box{i}" type="box" size="0.3 0.3 0.3" pos="{x} {y} {z}" rgba="1 0 0 1"/>\n'
                elif geom_type == 1:  # sphere
                    xml_content += f'<geom name="sphere{i}" type="sphere" size="0.3" pos="{x} {y} {z}" rgba="0 1 0 1"/>\n'
                elif geom_type == 2:  # capsule
                    xml_content += f'<geom name="capsule{i}" type="capsule" size="0.2 0.4" pos="{x} {y} {z}" rgba="1 0 1 1"/>\n'
                elif geom_type == 3:  # ellipsoid
                    xml_content += f'<geom name="ellipsoid{i}" type="ellipsoid" size="0.3 0.2 0.4" pos="{x} {y} {z}" rgba="1 1 0 1"/>\n'
                elif geom_type == 4:  # cylinder
                    xml_content += f'<geom name="cylinder{i}" type="cylinder" size="0.2 0.3" pos="{x} {y} {z}" rgba="0 0 1 1"/>\n'
            
            xml_content += xml_footer
            
            # 创建MuJoCo模型和场景
            test_model = mujoco.MjModel.from_xml_string(xml_content)
            test_data = mujoco.MjData(test_model)
            # mujoco.mj_saveLastXML(f"test_{num_geom}.xml", test_model)
            mujoco.mj_forward(test_model, test_data)
            
            test_scene = mujoco.MjvScene(test_model, maxgeom=max(100, num_geom + 10))
            mujoco.mjv_updateScene(
                test_model, test_data, mujoco.MjvOption(), 
                None, mujoco.MjvCamera(), 
                mujoco.mjtCatBit.mjCAT_ALL.value, test_scene
            )
            
            # 创建新的激光雷达传感器
            test_lidar = MjLidarSensor(test_scene, enable_profiling=args.profiling, verbose=args.verbose)
            
            # 执行多次测试取平均值
            n_tests = 5
            prepare_time_sum = 0
            kernel_time_sum = 0
            update_geom_time_sum = 0
            # 添加各操作时间的累加变量
            sensor_pose_time_sum = 0
            memory_alloc_time_sum = 0
            rays_update_time_sum = 0
            sync_time_sum = 0

            points = test_lidar.ray_cast_taichi(test_rays_phi, test_rays_theta, lidar_pose, test_scene)
            ti.sync()
            for i in range(n_tests):
                # 执行光线追踪
                points = test_lidar.ray_cast_taichi(test_rays_phi, test_rays_theta, lidar_pose, test_scene)
                ti.sync()
               
                # 累加时间
                prepare_time_sum += test_lidar.prepare_time
                kernel_time_sum += test_lidar.kernel_time
                update_geom_time_sum += test_lidar.update_geom_time
                # 累加各操作时间
                sensor_pose_time_sum += test_lidar.convert_sensor_pose_time
                memory_alloc_time_sum += test_lidar.memory_allocation_time
                rays_update_time_sum += test_lidar.update_rays_time
                sync_time_sum += test_lidar.sync_time
            
            # 计算平均时间
            avg_prepare_time = prepare_time_sum / n_tests
            avg_kernel_time = kernel_time_sum / n_tests
            avg_update_geom_time = update_geom_time_sum / n_tests
            # 计算各操作的平均时间
            avg_sensor_pose_time = sensor_pose_time_sum / n_tests
            avg_memory_alloc_time = memory_alloc_time_sum / n_tests
            avg_rays_update_time = rays_update_time_sum / n_tests
            avg_sync_time = sync_time_sum / n_tests
            
            geom_prepare_times.append(avg_prepare_time)
            geom_kernel_times.append(avg_kernel_time)
            geom_update_times.append(avg_update_geom_time)
            # 保存各操作的平均时间
            geom_sensor_pose_times.append(avg_sensor_pose_time)
            geom_memory_alloc_times.append(avg_memory_alloc_time)
            geom_rays_update_times.append(avg_rays_update_time)
            geom_sync_times.append(avg_sync_time)
            
            print(f"平均准备时间: {avg_prepare_time:.2f}ms")
            print(f"平均内核时间: {avg_kernel_time:.2f}ms")
            print(f"平均几何体更新时间: {avg_update_geom_time:.2f}ms")

        
        # 绘制性能结果图表
        plt.figure(figsize=(16, 10))
        
        # 全局字体设置
        title_font = {'fontsize': 14, 'fontweight': 'bold'}
        label_font = {'fontsize': 12}
        tick_font = {'fontsize': 10}
        
        # 根据字体可用性选择标签语言
        if use_english_labels:
            # 使用英文标签
            ray_count_title = 'Effect of Ray Count on Performance'
            prep_time_title = 'Breakdown of Preparation Time'
            geom_count_title = 'Effect of Geometry Count on Performance'
            geom_update_title = 'Effect of Geometry Count on update_geom_positions'
            
            ray_label = 'Ray Count'
            geom_label = 'Geometry Count'
            time_label = 'Time (ms)'
            
            prep_legend = 'Preparation Time'
            kernel_legend = 'Kernel Time'
            update_legend = 'Geometry Update Time'
            total_legend = 'Total Preparation Time'
            total_time_legend = 'Total Time'
            # 添加新的图例标签
            sensor_pose_legend = 'Sensor Pose Conversion'
            memory_alloc_legend = 'Memory Allocation'
            rays_update_legend = 'Rays Data Update'
            sync_legend = 'Synchronization'

            rays_num_word = 'rays'
        else:
            # 使用中文标签
            ray_count_title = '射线数量对性能的影响'
            prep_time_title = '准备时间的细分'
            geom_count_title = '几何体数量对性能的影响'
            geom_update_title = '几何体数量对update_geom_positions的影响'
            
            ray_label = '射线数量'
            geom_label = '几何体数量'
            time_label = '时间 (ms)'
            
            prep_legend = '准备时间'
            kernel_legend = '内核时间'
            update_legend = '几何体更新时间'
            total_legend = '总准备时间'
            total_time_legend = '总时间'
            # 添加新的图例标签
            sensor_pose_legend = '传感器位姿转换'
            memory_alloc_legend = '内存分配'
            rays_update_legend = '光线数据更新'
            sync_legend = '同步操作'

            rays_num_word = '射线'
        
        # 射线数量对性能的影响
        plt.subplot(2, 2, 1)
        plt.plot(ray_counts, prepare_times, 'o-', label=prep_legend)
        plt.plot(ray_counts, kernel_times, 's-', label=kernel_legend)
        # 添加总时间曲线
        total_times = [p + k for p, k in zip(prepare_times, kernel_times)]
        plt.plot(ray_counts, total_times, '^-', label=total_time_legend)
        plt.xlabel(ray_label, **label_font)
        plt.ylabel(time_label, **label_font)
        plt.title(ray_count_title, **title_font)
        plt.legend(prop={'size': 10})
        plt.grid(True)
        plt.tick_params(labelsize=10)
        
        # 准备时间的细分
        plt.subplot(2, 2, 2)
        # 绘制各个操作的时间曲线，使用不同的标记和颜色
        plt.plot(ray_counts, sensor_pose_times, 'o-', label=sensor_pose_legend, color='purple')
        plt.plot(ray_counts, memory_alloc_times, 's-', label=memory_alloc_legend, color='orange')
        plt.plot(ray_counts, rays_update_times, '^-', label=rays_update_legend, color='green')
        plt.plot(ray_counts, update_geom_times, 'D-', label=update_legend, color='red')
        plt.plot(ray_counts, sync_times, 'x-', label=sync_legend, color='brown')
        plt.plot(ray_counts, prepare_times, '*-', label=total_legend, color='blue', linewidth=2)
        plt.xlabel(ray_label, **label_font)
        plt.ylabel(time_label, **label_font)
        plt.title(prep_time_title, **title_font)
        plt.legend(prop={'size': 9}, loc='upper left')
        plt.grid(True)
        plt.tick_params(labelsize=10)
        
        # 几何体数量对性能的影响
        plt.subplot(2, 2, 3)
        plt.plot(num_geoms, geom_prepare_times, 'o-', label=prep_legend)
        plt.plot(num_geoms, geom_kernel_times, 's-', label=kernel_legend)
        # 添加总时间曲线
        geom_total_times = [p + k for p, k in zip(geom_prepare_times, geom_kernel_times)]
        plt.plot(num_geoms, geom_total_times, '^-', label=total_time_legend)
        plt.xlabel(geom_label, **label_font)
        plt.ylabel(time_label, **label_font)
        # 修改标题，添加测试用的雷达点数
        plt.title(f"{geom_count_title} ({test_rays_count} {rays_num_word})", **title_font)
        plt.legend(prop={'size': 10})
        plt.grid(True)
        plt.tick_params(labelsize=10)
        
        # 几何体数量对update_geom_positions的影响
        plt.subplot(2, 2, 4)
        # 绘制各个操作的时间曲线，使用不同的标记和颜色
        plt.plot(num_geoms, geom_sensor_pose_times, 'o-', label=sensor_pose_legend, color='purple')
        plt.plot(num_geoms, geom_memory_alloc_times, 's-', label=memory_alloc_legend, color='orange')
        plt.plot(num_geoms, geom_rays_update_times, '^-', label=rays_update_legend, color='green')
        plt.plot(num_geoms, geom_update_times, 'D-', label=update_legend, color='red')
        plt.plot(num_geoms, geom_sync_times, 'x-', label=sync_legend, color='brown')
        plt.plot(num_geoms, geom_prepare_times, '*-', label=total_legend, color='blue', linewidth=2)
        plt.xlabel(geom_label, **label_font)
        plt.ylabel(time_label, **label_font)
        # 修改标题，添加测试用的雷达点数
        plt.title(f"{geom_update_title} ({test_rays_count} {rays_num_word})", **title_font)
        plt.legend(prop={'size': 9}, loc='upper left')
        plt.grid(True)
        plt.tick_params(labelsize=10)
        
        plt.tight_layout()
        if args.save_fig:
            plt.savefig('lidar_performance_analysis.png', dpi=300)
        plt.show()
        
        # 将性能测试结果以表格形式打印
        print("\n" + "=" * 80)
        print("性能测试结果汇总")
        print("=" * 80)
        
        # 表格1：射线数量对性能的影响
        print("\n表格1: 射线数量对性能的影响")
        print("-" * 70)
        print(f"{'射线数量':^12} | {'准备时间 (ms)':^15} | {'内核时间 (ms)':^15} | {'总时间 (ms)':^15}")
        print("-" * 70)
        for i, count in enumerate(ray_counts):
            total_time = prepare_times[i] + kernel_times[i]
            print(f"{count:^12} | {prepare_times[i]:^15.2f} | {kernel_times[i]:^15.2f} | {total_time:^15.2f}")
        print("-" * 70)
        
        # 表格2：准备时间的细分
        print("\n表格2: 准备时间的细分")
        print("-" * 140)
        print(f"{'射线数量':^10} | {'传感器位姿转换':^15} | {'内存分配':^12} | {'光线数据更新':^15} | {'几何体更新':^15} | {'同步操作':^12} | {'总准备时间':^15}")
        print("-" * 140)
        for i, count in enumerate(ray_counts):
            print(f"{count:^10} | {sensor_pose_times[i]:^15.2f} | {memory_alloc_times[i]:^12.2f} | {rays_update_times[i]:^15.2f} | {update_geom_times[i]:^15.2f} | {sync_times[i]:^12.2f} | {prepare_times[i]:^15.2f}")
        print("-" * 140)
    
    # 执行标准光线追踪测试
    print("\n执行标准光线追踪测试:")
    old_enable_profiling = lidar.enable_profiling
    old_verbose = lidar.verbose
    
    # 临时开启性能分析和详细输出
    lidar.enable_profiling = True
    lidar.verbose = True

    rays_theta, rays_phi = generate_grid_scan_pattern(num_ray_cols=1800, num_ray_rows=64)

    # 优化内存布局
    rays_phi = np.ascontiguousarray(rays_phi)
    rays_theta = np.ascontiguousarray(rays_theta)

    for _ in range(3):
        start_time = time.time()
        points = lidar.ray_cast_taichi(rays_phi, rays_theta, lidar_pose, scene)
        ti.sync()
        end_time = time.time()
    
    # 恢复原始设置
    lidar.enable_profiling = old_enable_profiling
    lidar.verbose = old_verbose
    
    # 打印性能信息和当前位置
    print(f"耗时: {(end_time - start_time)*1000:.2f} ms, 射线数量: {len(rays_phi)}")
    
    # 三维点云可视化
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='viridis')

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)

    # 设置轴标签
    if use_english_labels:
        ax.set_xlabel('X-axis', fontsize=12)
        ax.set_ylabel('Y-axis', fontsize=12)
        ax.set_zlabel('Z-axis', fontsize=12)
        ax.set_title('Lidar 3D Point Cloud Visualization', fontsize=14, fontweight='bold')
        cbar.set_label('Height (Z-value)', fontsize=12)
    else:
        ax.set_xlabel('X轴', fontsize=12)
        ax.set_ylabel('Y轴', fontsize=12)
        ax.set_zlabel('Z轴', fontsize=12)
        ax.set_title('激光雷达三维点云可视化', fontsize=14, fontweight='bold')
        cbar.set_label('高度 (Z值)', fontsize=12)

    # 设置三轴等比例
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置网格线
    ax.grid(True)
    
    # 设置背景色为淡灰色，以更好地显示点云
    ax.set_facecolor((0.95, 0.95, 0.95))
    
    fig.tight_layout()
    if args.save_fig:
        plt.savefig('lidar_pointcloud.png', dpi=300)
    plt.show()