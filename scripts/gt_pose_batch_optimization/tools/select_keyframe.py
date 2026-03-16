#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR关键帧选择

输入：
- cam0.txt: 图像时间戳（CSV格式，第一列：timestamp，第二列：format）
- imu0_pose.txt: IMU位姿数据（TUM格式：timestamp x y z qx qy qz qw）
- lid0.txt: LiDAR时间戳（CSV格式，第一列：timestamp）

输出：
- txt文件，第一列是LiDAR关键帧时间戳（原始字符串），第二列是对应的图像关键帧时间戳（原始字符串）

算法：
遍历IMU帧，查找最近的LiDAR和图像时间戳。
仅当IMU与LiDAR的时间差≤sync_threshold、IMU与图像的时间差≤match_threshold时，才选择为关键帧。
"""

import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

def quat_angle_diff_deg(q1, q2, order="xyzw"):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    if order == "wxyz":
        q1 = [q1[1], q1[2], q1[3], q1[0]]
        q2 = [q2[1], q2[2], q2[3], q2[0]]

    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)

    r_rel = r1.inv() * r2
    angle_rad = r_rel.magnitude()
    return np.degrees(angle_rad)

def quat_angle_diff_euler_deg(q1, q2, order="xyzw",euler_seq="xyz"):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    if order == "wxyz":
        q1 = [q1[1], q1[2], q1[3], q1[0]]
        q2 = [q2[1], q2[2], q2[3], q2[0]]

    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)

    r_rel = r1.inv() * r2
    euler_deg = r_rel.as_euler(euler_seq, degrees=True)
    return euler_deg  # [roll_err, pitch_err, yaw_err]

def find_closest_timestamp(target_timestamps, query_timestamp):
    if len(target_timestamps) == 0:
        return None, None
    idx = np.argmin(np.abs(target_timestamps - query_timestamp))
    closest_ts = target_timestamps[idx]
    time_diff = abs(closest_ts - query_timestamp)
    return closest_ts, time_diff

def select_keyframes(imu_pose_file, lidar_timestamps, camera_timestamps, 
                    sync_thres, match_thres, pose_thres, angle_thres, tilt_thres, time_thres):
    imu_data = np.loadtxt(imu_pose_file)
    if len(imu_data) == 0:
        print("错误: IMU位姿文件为空")
        return None

    imu_timestamps = imu_data[:, 0]  # 第一列是时间戳
    positions = imu_data[:, 1:4]
    quaternions = imu_data[:, 4:8]

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
        d_roll,d_pitch,d_yaw = quat_angle_diff_euler_deg(quat,last_state["quat"])
        if imu_ts - last_state["t"] > time_thres or d_pos > pose_thres or abs(d_yaw) > angle_thres or abs(d_roll) > tilt_thres or abs(d_pitch) > tilt_thres: 
            lidar_ts, lidar_diff = find_closest_timestamp(lidar_timestamps, imu_ts)
            camera_ts, camera_diff = find_closest_timestamp(camera_timestamps, imu_ts)
            diff_time = abs(camera_ts - lidar_ts)
            is_keyframe = (diff_time <= match_thres) and (camera_diff <= sync_thres)

            if is_keyframe:
                keyframe_timestamps.append(imu_ts)
                last_state["t"] = imu_ts 
                last_state["pos"] = pos 
                last_state["quat"] = quat 

    return keyframe_timestamps

def match_lidar_to_camera(lidar_ts_float_list, camera_ts_float_list, keyframe_timestamps,
                          lidar_map, camera_map, time_threshold=0.05):
    """
    匹配关键帧LiDAR到图像，并输出原始字符串
    """
    matched_pairs = []
    for kf_ts in keyframe_timestamps:
        # 找最接近的 LiDAR 时间戳
        idx_lidar = np.argmin(np.abs(lidar_ts_float_list - kf_ts))
        lidar_ts_float = lidar_ts_float_list[idx_lidar]
        lidar_str = lidar_map[lidar_ts_float]

        # 找最接近的图像时间戳
        idx_camera = np.argmin(np.abs(camera_ts_float_list - kf_ts))
        camera_ts_float = camera_ts_float_list[idx_camera]
        diff = abs(camera_ts_float - kf_ts)

        if diff <= time_threshold:
            camera_str = camera_map[camera_ts_float]
            matched_pairs.append((lidar_str, camera_str))

    return matched_pairs

def main():
    parser = argparse.ArgumentParser(description='LiDAR关键帧选择')
    parser.add_argument('cam0', type=str, help='cam0.txt 文件路径（图像时间戳）')
    parser.add_argument('lid0', type=str, help='lid0.txt 文件路径（LiDAR时间戳）')
    parser.add_argument('imu0_pose', type=str, help='imu0_pose.txt 文件路径（IMU位姿，TUM格式）')
    parser.add_argument('output', type=str, help='输出文件路径')
    parser.add_argument('--sync-threshold', type=float, default=0.01, help='同步时间差阈值（秒）')
    parser.add_argument('--match-threshold', type=float, default=0.05, help='匹配时间差阈值（秒）')
    parser.add_argument('--pose-threshold', type=float, default=10, help='关键帧pose距离阈值（米）')
    parser.add_argument('--angle-threshold', type=float, default=15, help='关键帧angle阈值（度）')
    parser.add_argument('--tilt-angle-threshold', type=float, default=1.0, help='关键帧angle阈值（度）')
    parser.add_argument('--time-threshold', type=float, default=10, help='关键帧时间间隔阈值（秒）')
    args = parser.parse_args()

    # --- 读取数据，保留原始字符串 ---
    camera_raw = np.loadtxt(args.cam0, dtype=str, delimiter=',', skiprows=1)
    camera_ts_str_list = camera_raw[:]  # 原始字符串
    camera_ts_float_list = camera_ts_str_list.astype(float)  # 用于匹配
    camera_map = {float(v): v for v in camera_ts_str_list}  # float->原始字符串映射

    lidar_raw = np.loadtxt(args.lid0, dtype=str, delimiter=',', skiprows=1)
    lidar_ts_str_list = lidar_raw[:]
    lidar_ts_float_list = lidar_ts_str_list.astype(float)
    lidar_map = {float(v): v for v in lidar_ts_str_list}

    # 选择关键帧
    keyframe_timestamps = select_keyframes(
        args.imu0_pose,
        lidar_ts_float_list,
        camera_ts_float_list,
        args.sync_threshold,
        args.match_threshold,
        args.pose_threshold,
        args.angle_threshold,
        args.tilt_angle_threshold,
        args.time_threshold
    )

    if keyframe_timestamps is None:
        print("错误: 关键帧选择失败")
        return

    # 匹配LiDAR到图像，输出原始字符串
    matched_pairs = match_lidar_to_camera(
        lidar_ts_float_list,
        camera_ts_float_list,
        keyframe_timestamps,
        lidar_map,
        camera_map,
        time_threshold=args.match_threshold
    )

    # 保存结果
    with open(args.output, 'w') as f:
        f.write('lidar_timestamp,camera_timestamp\n')
        for lidar_str, camera_str in matched_pairs:
            f.write(f'{lidar_str},{camera_str}\n')

    print(f"完成! 共输出 {len(matched_pairs)} 个关键帧对")

if __name__ == "__main__":
    main()