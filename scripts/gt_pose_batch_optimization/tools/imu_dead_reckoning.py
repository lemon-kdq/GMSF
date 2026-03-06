#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用轮速和IMU姿态进行航位推算，计算IMU位置

算法原理（基于完整的外参模型）：
1. 坐标系：世界系W、车体系V（轮速测量系）、IMU系I
2. 已知：IMU姿态R_I^W、陀螺仪ω_I^I、轮速v_wheel、外参R_I^V和p_I^V
3. 流程：
   - 计算车体姿态：R_V^W = R_I^W · (R_I^V)^T
   - 将IMU角速度转到车体系：ω_V^V = (R_I^V)^T · ω_I^I
   - 计算IMU在车体系下的速度（补偿杆臂效应）：v_I^V = [v_wheel, 0, 0] + ω_V^V × p_I^V
   - 转到世界坐标系：v_I^W = R_V^W · v_I^V
   - 积分得到位置：p_I^W(k) = p_I^W(k-1) + v_I^W · Δt
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import sys


def load_wheel_velocity(filepath):
    """加载轮速数据"""
    timestamps = []
    velocities = []  # [v_left, v_right]

    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过表头
            parts = line.strip().split(',')
            timestamps.append(float(parts[0]))
            velocities.append([float(parts[1]), float(parts[2])])

    return np.array(timestamps), np.array(velocities)


def load_imu_data(filepath):
    """加载完整IMU数据（包含四元数和陀螺仪）"""
    timestamps = []
    quaternions = []  # [qx, qy, qz, qw]
    angular_velocities = []  # [gx, gy, gz] (rad/s)

    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过表头
            parts = line.strip().split(',')
            timestamps.append(float(parts[0]))
            # 四元数：qx, qy, qz, qw (索引7-10)
            quaternions.append([float(parts[7]), float(parts[8]), float(parts[9]), float(parts[10])])
            # 角速度：gx, gy, gz (索引4-6)
            angular_velocities.append([float(parts[4]), float(parts[5]), float(parts[6])])

    return np.array(timestamps), np.array(quaternions), np.array(angular_velocities)


def interpolate_data(query_timestamps, ref_timestamps, ref_values):
    """线性插值"""
    from scipy.interpolate import interp1d
    if len(ref_timestamps) < 2:
        return np.full((len(query_timestamps), ref_values.shape[1]), ref_values[0] if len(ref_values) > 0 else 0)
    f = interp1d(ref_timestamps, ref_values, axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
    return f(query_timestamps)


def euler_angles_to_rotation_matrix(roll, pitch, yaw):
    """
    从欧拉角构造旋转矩阵（ZYX顺序，即绕Z-Y-X轴旋转）
    roll: 绕X轴旋转（滚转角）
    pitch: 绕Y轴旋转（俯仰角）
    yaw: 绕Z轴旋转（偏航角）
    """
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return r.as_matrix()


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """从四元数获取3D旋转矩阵"""
    r = R.from_quat([qx, qy, qz, qw])
    return r.as_matrix()


def dead_reckoning(wheel_timestamps, wheel_velocities, imu_timestamps, imu_quaternions, imu_gyro,
                   extrinsic_rot=None, extrinsic_pos=None, initial_pos=None):
    """
    航位推算（基于完整外参模型）

    Args:
        wheel_timestamps: 轮速时间戳
        wheel_velocities: 轮速 [v_left, v_right] (m/s，线速度)
        imu_timestamps: IMU时间戳
        imu_quaternions: IMU四元数 [qx, qy, qz, qw] (R_I^W)
        imu_gyro: IMU陀螺仪数据 [gx, gy, gz] (rad/s, ω_I^I)
        extrinsic_rot: IMU相对于车体的旋转外参（欧拉角）[roll, pitch, yaw] (rad)
        extrinsic_pos: IMU在车体坐标系下的位置 [x, y, z] (m，p_I^V)
        initial_pos: 初始位置 [x, y, z] (m)

    Returns:
        positions: IMU位置数组 [x, y, z]
        timestamps: 对应时间戳
        quaternions: IMU四元数数组 [qx, qy, qz, qw] (与世界坐标系对齐)
        velocities_world: IMU在世界坐标系下的速度数组 [vx, vy, vz]
    """
    if extrinsic_rot is None:
        extrinsic_rot = [0.0, 0.0, 0.0]  # 默认无安装偏角
    if extrinsic_pos is None:
        extrinsic_pos = [0.0, 0.0, 0.0]
    if initial_pos is None:
        initial_pos = [0.0, 0.0, 0.0]

    # 将轮速插值到IMU时间戳
    wheel_velocities_interp = interpolate_data(imu_timestamps, wheel_timestamps, wheel_velocities)

    # 构造外参旋转矩阵 R_I^V (IMU相对于车体的旋转)
    R_I_V = euler_angles_to_rotation_matrix(*extrinsic_rot)
    # 外参旋转矩阵的转置 (R_I^V)^T = R_V^I
    R_V_I = R_I_V.T

    # 初始化
    positions = np.zeros((len(imu_timestamps), 3))
    velocities_world = np.zeros((len(imu_timestamps), 3)) # 添加速度存储
    positions[0] = initial_pos

    dt = np.diff(imu_timestamps)

    for i in range(1, len(imu_timestamps)):
        # ===== 第一步：计算车体在世界系下的姿态 R_V^W =====
        # R_V^W = R_I^W · (R_I^V)^T
        R_I_W = quaternion_to_rotation_matrix(*imu_quaternions[i-1])
        R_V_W = R_I_W @ R_V_I

        # ===== 第二步：将IMU角速度转到车体坐标系 =====
        # ω_V^V = (R_I^V)^T · ω_I^I
        omega_I_I = imu_gyro[i-1]  # IMU测得的角速度
        omega_V_V = R_V_I @ omega_I_I  # 转到车体坐标系

        # ===== 第三步：计算IMU在车体系下的速度（补偿杆臂效应）=====
        # 车体后轴中心速度（轮速计测量的是这个）
        v_left = wheel_velocities_interp[i-1][0]
        v_right = wheel_velocities_interp[i-1][1]
        v_wheel = (v_left + v_right) / 2.0  # 前进速度

        v_V_V = np.array([v_wheel, 0.0, 0.0])  # 车体在自身坐标系下的速度

        # 杆臂效应补偿：IMU在车体系下的速度
        # v_I^V = v_V_V + ω_V^V × p_I^V
        p_I_V = np.array(extrinsic_pos)  # IMU在车体坐标系下的位置
        v_rotation = np.cross(omega_V_V, p_I_V)  # ω × p 产生的线速度
        v_I_V = v_V_V + v_rotation

        # ===== 第四步：将IMU速度转到世界坐标系 =====
        # v_I^W = R_V^W · v_I^V
        v_I_W = R_V_W @ v_I_V
        velocities_world[i] = v_I_W # 记录世界系下的IMU速度

        # ===== 第五步：位置积分 =====
        positions[i] = positions[i-1] + v_I_W * dt[i-1]

    return positions, imu_timestamps, imu_quaternions, velocities_world


def main():
    parser = argparse.ArgumentParser(description='使用轮速和IMU姿态进行航位推算（完整外参模型）')
    parser.add_argument('--wheel-vel', type=str, required=True, help='轮速数据文件路径')
    parser.add_argument('--imu-data', type=str, required=True, help='IMU完整数据文件路径（包含四元数和陀螺仪）')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--ext-rot-roll', type=float, default=0.0, help='外参旋转roll (rad)')
    parser.add_argument('--ext-rot-pitch', type=float, default=0.0, help='外参旋转pitch (rad)')
    parser.add_argument('--ext-rot-yaw', type=float, default=0.0, help='外参旋转yaw (rad)')
    parser.add_argument('--ext-pos-x', type=float, default=1.2, help='外参位置x (m)')
    parser.add_argument('--ext-pos-y', type=float, default=0.0, help='外参位置y (m)')
    parser.add_argument('--ext-pos-z', type=float, default=0.8, help='外参位置z (m)')
    parser.add_argument('--initial-x', type=float, default=0.0, help='初始位置x (m)')
    parser.add_argument('--initial-y', type=float, default=0.0, help='初始位置y (m)')
    parser.add_argument('--initial-z', type=float, default=0.0, help='初始位置z (m)')

    args = parser.parse_args()

    print(f'加载轮速数据: {args.wheel_vel}')
    wheel_timestamps, wheel_velocities = load_wheel_velocity(args.wheel_vel)
    print(f'  轮速数据点数: {len(wheel_timestamps)}')

    print(f'加载IMU完整数据: {args.imu_data}')
    imu_timestamps, imu_quaternions, imu_gyro = load_imu_data(args.imu_data)
    print(f'  IMU数据点数: {len(imu_timestamps)}')

    extrinsic_rot = [args.ext_rot_roll, args.ext_rot_pitch, args.ext_rot_yaw]
    extrinsic_pos = [args.ext_pos_x, args.ext_pos_y, args.ext_pos_z]
    initial_pos = [args.initial_x, args.initial_y, args.initial_z]

    print(f'参数设置:')
    print(f'  外参旋转（安装偏角）(rad): roll={extrinsic_rot[0]}, pitch={extrinsic_rot[1]}, yaw={extrinsic_rot[2]}')
    print(f'  外参位置（杆臂）(m): {extrinsic_pos}')
    print(f'  初始位置（世界坐标系）(m): {initial_pos}')

    print('开始航位推算...')
    positions, timestamps, quaternions, velocities_world = dead_reckoning(
        wheel_timestamps, wheel_velocities,
        imu_timestamps, imu_quaternions, imu_gyro,
        extrinsic_rot=extrinsic_rot,
        extrinsic_pos=extrinsic_pos,
        initial_pos=initial_pos
    )

    # 保存结果（TUM格式：timestamp tx ty tz qx qy qz qw）
    print(f'保存结果到: {args.output} (TUM格式)')
    with open(args.output, 'w') as f:
        for t, pos, quat in zip(timestamps, positions, quaternions):
            f.write(f'{t:.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n')

    # 保存速度结果 (CSV格式)
    vel_output = args.output.replace('.txt', '_vel.csv')
    print(f'保存速度数据到: {vel_output}')
    with open(vel_output, 'w') as f:
        f.write("timestamp,vx,vy,vz\n")
        for t, v in zip(timestamps, velocities_world):
            f.write(f'{t:.6f},{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}\n')

    print(f'完成! 共输出 {len(positions)} 个位置点')
    print(f'位置范围:')
    print(f'  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] m')
    print(f'  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] m')
    print(f'  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] m')


if __name__ == '__main__':
    main()