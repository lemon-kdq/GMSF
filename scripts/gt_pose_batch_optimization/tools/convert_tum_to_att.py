#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将VQF输出的TUM格式转换为ATT格式

输入格式（TUM）: timestamp x y z qx qy qz qw
输出格式（ATT）: timestamp roll pitch yaw qx qy qz qw
"""

import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R


def convert_tum_to_att(input_file, output_file):
    """
    将TUM格式转换为ATT格式

    Args:
        input_file: TUM格式文件路径
        output_file: ATT格式文件路径
    """
    print(f"读取TUM格式文件: {input_file}")

    # 读取TUM格式数据
    # 格式: timestamp x y z qx qy qz qw
    tum_data = np.loadtxt(input_file)

    if len(tum_data) == 0:
        print("错误: 输入文件为空")
        return

    timestamps = tum_data[:, 0]
    quats = tum_data[:, 4:8]  # qx, qy, qz, qw

    # 将四元数转换为欧拉角 (roll, pitch, yaw)
    print("计算欧拉角...")
    rotation = R.from_quat(quats)
    euler = rotation.as_euler('xyz', degrees=False)

    # 构造ATT格式数据
    # 格式: timestamp roll pitch yaw qx qy qz qw
    att_data = np.zeros((len(timestamps), 8))
    att_data[:, 0] = timestamps
    att_data[:, 1:4] = euler  # roll, pitch, yaw
    att_data[:, 4:8] = quats  # qx, qy, qz, qw

    # 保存文件
    print(f"保存ATT格式文件: {output_file}")
    np.savetxt(output_file, att_data, fmt='%.6f')
    print(f"转换完成: {len(timestamps)} 条数据")

    # 统计信息
    print(f"\n欧拉角统计:")
    print(f"  Roll: [{euler[:, 0].min():.6f}, {euler[:, 0].max():.6f}] rad")
    print(f"  Pitch: [{euler[:, 1].min():.6f}, {euler[:, 1].max():.6f}] rad")
    print(f"  Yaw: [{euler[:, 2].min():.6f}, {euler[:, 2].max():.6f}] rad")


def main():
    parser = argparse.ArgumentParser(description='将TUM格式转换为ATT格式')
    parser.add_argument('input', type=str, help='TUM格式输入文件路径')
    parser.add_argument('output', type=str, help='ATT格式输出文件路径')

    args = parser.parse_args()

    convert_tum_to_att(args.input, args.output)


if __name__ == '__main__':
    main()
