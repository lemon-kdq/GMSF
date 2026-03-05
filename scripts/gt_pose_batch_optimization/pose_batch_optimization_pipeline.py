#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
姿态批量优化流水线

处理流程：
1. 执行 extract_sensor_data.py 从 rosbag 提取传感器数据
2. 执行 vqf_att_offline_estimate.py 进行姿态估计
3. 执行 convert_tum_to_att.py 将VQF输出转换为ATT格式
4. 执行 replace_imu_quat.py 替换四元数
5. 执行 imu_dead_reckoning.py 进行姿态递推
"""

import os
import sys
import argparse
import subprocess


def run_step(description, cmd):
    """执行一个处理步骤"""
    print(f"\n{'='*60}")
    print(f"步骤: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("错误输出:", result.stderr)

    if result.returncode != 0:
        print(f"错误: 步骤 '{description}' 执行失败 (返回码: {result.returncode})")
        sys.exit(1)

    print(f"✓ 步骤 '{description}' 完成")


def get_scripts_dir():
    """获取脚本所在目录的tools子目录"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    tools_dir = os.path.join(current_dir, 'tools')
    return tools_dir


def main():
    parser = argparse.ArgumentParser(description='姿态批量优化流水线')
    parser.add_argument('--bag-path', type=str, required=True, help='rosbag文件路径')
    parser.add_argument('--lid-imu-path', type=str, required=True, help='rosbag文件路径')
    parser.add_argument('--output-dir', type=str, default='./output', help='输出目录路径')

    # 步骤控制
    parser.add_argument('--skip-extract', action='store_true', help='跳过数据提取步骤')
    parser.add_argument('--skip-vqf', action='store_true', help='跳过VQF姿态估计步骤')
    parser.add_argument('--skip-convert', action='store_true', help='跳过格式转换步骤')
    parser.add_argument('--skip-replace', action='store_true', help='跳过四元数替换步骤')
    parser.add_argument('--skip-dead-reckoning', action='store_true', help='跳过航位推算步骤')

    # IMU 外参参数
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

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 脚本路径（自动从当前脚本位置判断）
    scripts_dir = get_scripts_dir()
    extract_script = os.path.join(scripts_dir, 'extract_sensor_data.py')
    vqf_script = os.path.join(scripts_dir, 'vqf_att_offline_estimate.py')
    convert_script = os.path.join(scripts_dir, 'convert_tum_to_att.py')
    replace_script = os.path.join(scripts_dir, 'replace_imu_quat.py')
    dead_reckoning_script = os.path.join(scripts_dir, 'imu_dead_reckoning.py')

    print(f"\n{'#'*60}")
    print(f"# 姿态批量优化流水线")
    print(f"{'#'*60}")
    print(f"rosbag路径: {args.bag_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"脚本目录: {scripts_dir}")

    # 定义文件路径
    imu0_path = os.path.join(args.output_dir, 'imu0.txt')
    wheel_vel_path = os.path.join(args.output_dir, 'wheel_velocity.txt')
    imu0_vqf_path = os.path.join(args.output_dir, 'imu0_vqf.txt')
    imu0_att_path = os.path.join(args.output_dir, 'imu0_att.txt')
    imu0_replaced_path = os.path.join(args.output_dir, 'imu0_replaced.txt')
    pose_output_path = os.path.join(args.output_dir, 'imu0_pose.txt')


    # ===== 步骤1: 提取传感器数据 =====
    if not args.skip_extract:
        run_step(
            "从rosbag提取传感器数据",
            ['python3', extract_script, args.bag_path, args.output_dir]
        )
    else:
        print("\n跳过步骤1: 数据提取")

    # ===== 步骤2: VQF姿态估计 =====
    if not args.skip_vqf:
        run_step(
            "VQF姿态估计",
            ['python3', vqf_script, imu0_path, imu0_vqf_path]
        )
    else:
        print("\n跳过步骤2: VQF姿态估计")

    # ===== 步骤3: 转换格式 (TUM -> ATT) =====
    if not args.skip_convert:
        run_step(
            "转换VQF输出格式 (TUM -> ATT)",
            ['python3', convert_script, imu0_vqf_path, imu0_att_path]
        )
    else:
        print("\n跳过步骤3: 格式转换")

    # ===== 步骤4: 替换四元数 =====
    if not args.skip_replace:
        run_step(
            "替换IMU四元数 (原始数据 -> VQF姿态)",
            ['python3', replace_script, imu0_path, imu0_att_path, imu0_replaced_path]
        )
    else:
        print("\n跳过步骤4: 四元数替换")

    # ===== 步骤5: 航位推算 =====
    if not args.skip_dead_reckoning:
        run_step(
            "IMU航位推算",
            ['python3', dead_reckoning_script,
             '--wheel-vel', wheel_vel_path,
             '--imu-data', imu0_replaced_path,
             '--output', pose_output_path,
             '--ext-rot-roll', str(args.ext_rot_roll),
             '--ext-rot-pitch', str(args.ext_rot_pitch),
             '--ext-rot-yaw', str(args.ext_rot_yaw),
             '--ext-pos-x', str(args.ext_pos_x),
             '--ext-pos-y', str(args.ext_pos_y),
             '--ext-pos-z', str(args.ext_pos_z),
             '--initial-x', str(args.initial_x),
             '--initial-y', str(args.initial_y),
             '--initial-z', str(args.initial_z)]
        )
    else:
        print("\n跳过步骤5: 航位推算")

    # ===== 完成 =====
    print(f"\n{'#'*60}")
    print(f"# 流水线完成！")
    print(f"{'#'*60}")
    print(f"\n生成的文件:")
    print(f"  - {imu0_path} (原始IMU数据)")
    print(f"  - {wheel_vel_path} (轮速数据)")
    print(f"  - {imu0_vqf_path} (VQF姿态估计 - TUM格式)")
    print(f"  - {imu0_att_path} (VQF姿态 - ATT格式)")
    print(f"  - {imu0_replaced_path} (替换四元数后的IMU数据)")
    print(f"  - {pose_output_path} (最终IMU位姿 - TUM格式)")
    print(f"\n使用 evo 评估轨迹:")
    print(f"  evo_ape tum groundtruth.txt {pose_output_path} -va --plot")


if __name__ == '__main__':
    main()
