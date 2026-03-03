#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据时间戳将 imu0_att.txt 中的四元数替换到 imu0.txt 中

输入：
- imu0.txt: 完整IMU数据，格式: timestamp,ax,ay,az,gx,gy,gz,qx,qy,qz,qw
- imu0_att.txt: 姿态数据，格式: timestamp roll pitch yaw qx qy qz qw

输出：
- 替换四元数后的完整IMU数据
"""

import argparse
import sys


def load_att_quaternions(filepath):
    """加载imu0_att.txt，返回时间戳到四元数的映射"""
    quat_map = {}

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            timestamp = float(parts[0])
            # 取最后4个值作为四元数 (qx, qy, qz, qw)
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            quat_map[timestamp] = (qx, qy, qz, qw)

    print(f'从 {filepath} 加载了 {len(quat_map)} 个姿态数据')
    return quat_map


def replace_quaternions(imu_file, att_quat_map, output_file):
    """将imu0.txt中的四元数替换为att文件中的四元数"""

    replaced_count = 0
    total_count = 0

    with open(imu_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # 写入表头
            if line.startswith('timestamp'):
                f_out.write(line)
                continue

            parts = line.strip().split(',')
            if len(parts) < 11:
                f_out.write(line)
                continue

            timestamp = float(parts[0])
            total_count += 1

            # 查找对应时间戳的四元数
            if timestamp in att_quat_map:
                qx, qy, qz, qw = att_quat_map[timestamp]
                # 替换四元数（索引7, 8, 9, 10）
                parts[7] = str(qx)
                parts[8] = str(qy)
                parts[9] = str(qz)
                parts[10] = str(qw)
                replaced_count += 1
            # 如果找不到，保持原样

            # 写入替换后的行
            f_out.write(','.join(parts) + '\n')

    print(f'处理完成: 共 {total_count} 行，替换了 {replaced_count} 个四元数')


def main():
    parser = argparse.ArgumentParser(description='根据时间戳将 imu0_att.txt 中的四元数替换到 imu0.txt 中')
    parser.add_argument('imu0_file', type=str, help='imu0.txt 文件路径（完整IMU数据）')
    parser.add_argument('imu0_att_file', type=str, help='imu0_att.txt 文件路径（姿态数据）')
    parser.add_argument('output_file', type=str, help='输出文件路径')

    args = parser.parse_args()

    print(f'加载姿态数据: {args.imu0_att_file}')
    att_quat_map = load_att_quaternions(args.imu0_att_file)

    print(f'替换四元数: {args.imu0_file} -> {args.output_file}')
    replace_quaternions(args.imu0_file, att_quat_map, args.output_file)

    print(f'输出完成: {args.output_file}')


if __name__ == '__main__':
    main()
