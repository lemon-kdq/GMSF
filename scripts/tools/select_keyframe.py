#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR关键帧选择

输入：
- cam0.txt: 图像时间戳（CSV格式，第一列：timestamp，第二列：format）
- imu0_pose.txt: IMU位姿数据（TUM格式：timestamp x y z qx qy qz qw）
- lid0.txt: LiDAR时间戳（CSV格式，第一列：timestamp）

输出：
- txt文件，第一列是LiDAR关键帧时间戳，第二列是对应的图像关键帧时间戳，第三列是图像格式

算法：
遍历IMU帧，查找最近的LiDAR和图像时间戳。
仅当IMU与LiDAR的时间差≤0.01s、IMU与图像的时间差≤0.01s时，才选择为关键帧。
"""

import numpy as np
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_angle_diff_deg(q1, q2, order="xyzw"):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    # SciPy 使用 [x, y, z, w]
    if order == "wxyz":
        q1 = [q1[1], q1[2], q1[3], q1[0]]
        q2 = [q2[1], q2[2], q2[3], q2[0]]

    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)

    # 相对旋转
    r_rel = r1.inv() * r2

    # 轴角模长（弧度）
    angle_rad = r_rel.magnitude()

    return np.degrees(angle_rad)

def find_closest_timestamp(target_timestamps, query_timestamp):
    """
    在目标时间戳数组中找到最接近查询时间戳的值

    Args:
        target_timestamps: 目标时间戳数组（如图像或LiDAR时间戳）
        query_timestamp: 查询时间戳（如IMU时间戳）

    Returns:
        最接近的时间戳及其时间差
    """
    if len(target_timestamps) == 0:
        return None, None
    idx = np.argmin(np.abs(target_timestamps - query_timestamp))
    closest_ts = target_timestamps[idx]
    time_diff = abs(closest_ts - query_timestamp)
    return closest_ts, time_diff


def find_closest_camera(camera_timestamps, query_timestamp):
    """
    在图像时间戳数组中找到最接近查询时间戳的值及其格式

    Args:
        camera_timestamps: 图像时间戳数组
        camera_formats: 图像格式数组
        query_timestamp: 查询时间戳

    Returns:
        最接近的时间戳、时间差和对应的格式
    """
    if len(camera_timestamps) == 0:
        return None, None, None
    idx = np.argmin(np.abs(camera_timestamps - query_timestamp))
    closest_ts = camera_timestamps[idx]
    time_diff = abs(closest_ts - query_timestamp)
    return closest_ts, time_diff


def select_keyframes(imu_pose_file, lidar_timestamps, camera_timestamps, 
                    sync_thres,match_thres,pose_thres,angle_thres,time_thres):
    """
    根据IMU轨迹选择关键帧

    Args:
        imu_pose_file: IMU位姿文件（TUM格式）
        lidar_timestamps: LiDAR时间戳数组
        camera_timestamps: 图像时间戳数组
        sync_threshold: 同步时间差阈值（秒），默认0.01s

    Returns:
        keyframe_timestamps: 选中的关键帧时间戳列表
    """
    # 读取IMU位姿数据（TUM格式）
    print(f"读取IMU位姿数据: {imu_pose_file}")
    imu_data = np.loadtxt(imu_pose_file)
    if len(imu_data) == 0:
        print("错误: IMU位姿文件为空")
        return None

    imu_timestamps = imu_data[:, 0]  # 第一列是时间戳
    positions = imu_data[:, 1:4]  # 第2-4列是 x, y, z
    quaternions = imu_data[:, 4:8]  # 第5-8列是 qx, qy, qz, qw

    print(f"共 {len(imu_timestamps)} 个IMU帧")
    print(f"LiDAR帧数: {len(lidar_timestamps)}")
    print(f"图像帧数: {len(camera_timestamps)}")
    print(f"\n开始选择关键帧...")
    print(f"同步时间差阈值: {sync_thres} s")

    keyframe_timestamps = []

    last_state = {"t": 0.,
                  "pos": [0.,0.,0],
                  "quat":[0.,0.,0.,1.0] 
                  }
    
    for i in range(len(imu_timestamps)):
        imu_ts = imu_timestamps[i]
        pos = positions[i]
        quat = quaternions[i] 
        
        d_pos = np.linalg.norm(pos - last_state["pos"])
        d_angle = quat_angle_diff_deg(quat,last_state["quat"])
        if imu_ts - last_state["t"] > time_thres or d_pos > pose_thres or d_angle > angle_thres: 
            
            # 查找最近的LiDAR时间戳
            lidar_ts, lidar_diff = find_closest_timestamp(lidar_timestamps, imu_ts)

            # 查找最近的图像时间戳（包括格式）
            camera_ts, camera_diff = find_closest_timestamp(camera_timestamps, imu_ts)
    
            diff_time = abs(camera_ts - lidar_ts)
    
            # 判断是否为关键帧：两个时间差都≤阈值 
            is_keyframe = (diff_time <= match_thres) and (camera_diff <= sync_thres)

            if is_keyframe:
                keyframe_timestamps.append(imu_ts)
                last_state["t"] = imu_ts 
                last_state["pos"] = pos 
                last_state["quat"] = quat 
                print(f"  帧 {i} ({imu_ts:.6f}) 被选为关键帧: LiDAR差={lidar_diff:.4f}s, 图像和lidar差={diff_time:.4f}s")
            else:
                # 输出跳过原因
                skip_reasons = []
                if lidar_diff > sync_thres:
                    skip_reasons.append(f"LiDAR差={lidar_diff:.4f}s")
                if camera_diff > match_thres:
                    skip_reasons.append(f"图像差={diff_time:.4f}s")
                if skip_reasons:
                    print(f"  帧 {i} ({imu_ts:.6f}) 跳过: {'; '.join(skip_reasons)}")

    print(f"\n关键帧选择完成: 共选中 {len(keyframe_timestamps)} 个关键帧")
    print(f"  关键帧占比: {len(keyframe_timestamps) / len(imu_timestamps) * 100:.2f}%")

    return keyframe_timestamps


def match_lidar_to_camera(lidar_timestamps, camera_timestamps, camera_formats, keyframe_timestamps, time_threshold=0.05):
    """
    将LiDAR关键帧匹配到对应的图像帧

    Args:
        lidar_timestamps: LiDAR时间戳数组
        camera_timestamps: 图像时间戳数组
        camera_formats: 图像格式数组
        keyframe_timestamps: 关键帧时间戳列表（从IMU轨迹选择）
        time_threshold: 时间差阈值（秒），默认0.05s

    Returns:
        matched_pairs: [(lidar_timestamp, camera_timestamp, camera_format), ...] 匹配对
    """
    print(f"\n匹配LiDAR关键帧到图像帧...")
    print(f"  LiDAR关键帧数: {len(keyframe_timestamps)}")
    print(f"  时间差阈值: {time_threshold} s")

    matched_pairs = []

    for kf_ts in keyframe_timestamps:
        camera_ts, time_diff = find_closest_camera(camera_timestamps, kf_ts)

        if camera_ts is not None and time_diff <= time_threshold:
            matched_pairs.append((kf_ts, camera_ts))
        else:
            reason = "未找到图像" if camera_ts is None else f"时间差={time_diff:.4f}s, 格式={camera_format}"
            print(f"  关键帧 {kf_ts:.6f} 跳过: {reason}")

    print(f"匹配完成: {len(matched_pairs)}/{len(keyframe_timestamps)} 个LiDAR关键帧匹配到图像")

    return matched_pairs


def main():
    parser = argparse.ArgumentParser(description='LiDAR关键帧选择')
    parser.add_argument('cam0', type=str, help='cam0.txt 文件路径（图像时间戳）')
    parser.add_argument('imu0_pose', type=str, help='imu0_pose.txt 文件路径（IMU位姿，TUM格式）')
    parser.add_argument('lid0', type=str, help='lid0.txt 文件路径（LiDAR时间戳）')
    parser.add_argument('output', type=str, help='输出文件路径')
    parser.add_argument('--sync-threshold', type=float, default=0.01,
                    help='同步时间差阈值（秒），默认: 0.01s')
    parser.add_argument('--match-threshold', type=float, default=0.05,
                    help='匹配时间差阈值（秒），默认: 0.05s')
    parser.add_argument('--pose-threshold', type=float, default=5,
                    help='关键帧pose距离，默认: 5m')
    parser.add_argument('--angle-threshold', type=float, default=15,
                    help='关键帧angle距离，默认: 15度')
    parser.add_argument('--time-threshold', type=float, default=10,
                    help='关键帧时间距离，默认: 10s')
    args = parser.parse_args()

    # 读取输入文件
    print(f"读取图像时间戳: {args.cam0}")
    camera_data = np.loadtxt(args.cam0, dtype=str, delimiter=',', skiprows=1)
    camera_timestamps = camera_data[:, 0].astype(float)
    camera_formats = camera_data[:, 1]  # 第二列是格式字符串（如"jpeg", "png"）

    print(f"读取LiDAR时间戳: {args.lid0}")
    lidar_data = np.loadtxt(args.lid0, delimiter=',', skiprows=1)
    lidar_timestamps = lidar_data[:, 0].astype(float)

    sync_thres = args.sync_threshold
    match_thres = args.match_threshold

    pose_thres = args.pose_threshold
    angle_thres = args.angle_threshold
    time_thres = args.time_threshold

    # 选择关键帧
    keyframe_timestamps = select_keyframes(
        args.imu0_pose,
        lidar_timestamps,
        camera_timestamps,
        sync_thres,
        match_thres,
        pose_thres,
        angle_thres,
        time_thres
    )

    if keyframe_timestamps is None:
        print("错误: 关键帧选择失败")
        return

    # 匹配关键帧到图像帧
    matched_pairs = match_lidar_to_camera(
        lidar_timestamps,
        camera_timestamps,
        camera_formats,
        keyframe_timestamps,
        time_threshold=args.match_threshold
    )

    # 保存结果
    print(f"\n保存结果到: {args.output}")
    with open(args.output, 'w') as f:
        # 写入表头
        f.write('lidar_timestamp,camera_timestamp,camera_format\n')
        # 写入匹配对
        for lidar_ts, camera_ts in matched_pairs:
            f.write(f'{lidar_ts:.6f},{camera_ts:.6f}\n')

    print(f"完成! 共输出 {len(matched_pairs)} 个关键帧对")
    print(f"LiDAR时间范围: [{matched_pairs[0][0]:.3f}, {matched_pairs[-1][0]:.3f}]")
    print(f"图像时间范围: [{matched_pairs[0][1]:.3f}, {matched_pairs[-1][1]:.3f}]")
    
if __name__ == "__main__":
    main()