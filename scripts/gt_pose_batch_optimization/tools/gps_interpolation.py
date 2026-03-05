import numpy as np
import pandas as pd
import argparse
import os

def interpolate_gps_to_pose(gps_csv_path, pose_tum_path, output_path):
    """
    根据 Pose 的时间戳对 GPS 数据进行线性插值
    """
    if not os.path.exists(gps_csv_path):
        print(f"Error: GPS file '{gps_csv_path}' not found.")
        return
    if not os.path.exists(pose_tum_path):
        print(f"Error: Pose file '{pose_tum_path}' not found.")
        return

    print(f"Reading GPS data: {gps_csv_path}")
    # 读取 GPS (CSV 格式, 带表头: timestamp,x,y,z,status,status_service)
    gps_df = pd.read_csv(gps_csv_path)
    gps_ts = gps_df['timestamp'].values
    gps_x = gps_df['x'].values
    gps_y = gps_df['y'].values
    gps_z = gps_df['z'].values
    gps_status = gps_df['status'].values

    print(f"Reading Pose data: {pose_tum_path}")
    # 读取 Pose (TUM 格式: timestamp tx ty tz qx qy qz qw)
    # 使用 np.genfromtxt 自动跳过注释行，并只取第一列时间戳
    pose_data = np.genfromtxt(pose_tum_path)
    if pose_data.ndim == 1: # 处理只有一个位姿的情况
        pose_ts = np.array([pose_data[0]])
    else:
        pose_ts = pose_data[:, 0]

    # --- 开始插值 ---
    # 过滤掉不在 GPS 记录范围内的 Pose 时间戳
    mask = (pose_ts >= gps_ts[0]) & (pose_ts <= gps_ts[-1])
    valid_pose_ts = pose_ts[mask]
    
    if len(valid_pose_ts) == 0:
        print("Error: No overlapping timestamps found between GPS and Pose files.")
        return

    print(f"Interpolating {len(valid_pose_ts)} points...")

    # 使用 numpy.interp 进行线性插值 (x, y, z)
    interp_x = np.interp(valid_pose_ts, gps_ts, gps_x)
    interp_y = np.interp(valid_pose_ts, gps_ts, gps_y)
    interp_z = np.interp(valid_pose_ts, gps_ts, gps_z)
    
    # 对于 status 状态，采用最近邻插值 (线性插值后四舍五入)
    interp_status = np.interp(valid_pose_ts, gps_ts, gps_status)
    interp_status = np.round(interp_status).astype(int)

    # --- 保存结果 ---
    # 构建结果矩阵: [timestamp, x, y, z, status]
    result = np.column_stack((valid_pose_ts, interp_x, interp_y, interp_z, interp_status))
    
    header = "timestamp,x,y,z,status"
    np.savetxt(output_path, result, delimiter=',', header=header, comments='', fmt='%.6f,%.6f,%.6f,%.6f,%d')
    
    print("-" * 30)
    print(f"Successfully saved to: {output_path}")
    print(f"Original Pose count: {len(pose_ts)}")
    print(f"Interpolated count:  {len(result)}")
    if len(valid_pose_ts) < len(pose_ts):
        skipped = len(pose_ts) - len(valid_pose_ts)
        print(f"Notice: {skipped} poses were outside GPS time range and skipped.")
    print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description="Interpolate GPS ENU data to match Pose (TUM) timestamps.")
    
    # 定义命令行参数
    parser.add_argument("-g", "--gps", required=True, help="Path to the GPS ENU CSV file.")
    parser.add_argument("-p", "--pose", required=True, help="Path to the Pose TUM txt file.")
    parser.add_argument("-o", "--output", default="interpolated_gps.csv", help="Output CSV path (default: interpolated_gps.csv).")
    
    args = parser.parse_args()

    # 执行插值
    interpolate_gps_to_pose(args.gps, args.pose, args.output)

if __name__ == "__main__":
    main()