import numpy as np
import argparse
import pandas as pd
from scipy.spatial.transform import Rotation as R_tool

def umeyama_alignment(source_pts, target_pts, estimate_scale=True):
    """全自由度对齐 (7-DoF: R, t, s)"""
    num_pts = source_pts.shape[0]
    dim = source_pts.shape[1]
    src_mean = np.mean(source_pts, axis=0)
    dst_mean = np.mean(target_pts, axis=0)
    src_centered = source_pts - src_mean
    dst_centered = target_pts - dst_mean
    H = (dst_centered.T @ src_centered) / num_pts
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[dim-1, :] *= -1
        R = U @ Vt
    scale = 1.0
    if estimate_scale:
        src_var = np.mean(np.sum(src_centered**2, axis=1))
        scale = np.trace(np.diag(S)) / src_var
    translation = dst_mean - scale * (R @ src_mean)
    return scale, R, translation

def yaw_xyz_alignment(source_pts, target_pts):
    """仅对齐 Yaw 和 XYZ (4-DoF: Yaw, x, y, z)"""
    src_mean = np.mean(source_pts, axis=0)
    dst_mean = np.mean(target_pts, axis=0)
    src_centered = source_pts - src_mean
    dst_centered = target_pts - dst_mean
    H_2d = (dst_centered[:, :2].T @ src_centered[:, :2])
    U, _, Vt = np.linalg.svd(H_2d)
    R_2d = U @ Vt
    if np.linalg.det(R_2d) < 0:
        Vt[1, :] *= -1
        R_2d = U @ Vt
    R_3d = np.eye(3)
    R_3d[:2, :2] = R_2d
    scale = 1.0
    translation = dst_mean - R_3d @ src_mean
    return scale, R_3d, translation

def main():
    parser = argparse.ArgumentParser(description="Align Full Trajectory and GPS.")
    parser.add_argument("-p", "--pose", required=True, help="Path to TUM pose file.")
    parser.add_argument("-g", "--gps", required=True, help="Path to interpolated GPS CSV file.")
    parser.add_argument("-m", "--mode", choices=['umeyama', 'yaw_xyz'], default='yaw_xyz', help="Alignment mode.")
    parser.add_argument("-o", "--output_dir", default=".", help="Output directory.")
    parser.add_argument("--status", type=int, default=None, help="Filter GPS by status.")
    parser.add_argument("--min_dist", type=float, default=30.0, help="Min distance (m) to calculate alignment.")
    args = parser.parse_args()

    # 1. 数据读取与时间戳对齐
    poses_df = pd.read_csv(args.pose, sep='\s+', header=None, 
                           names=['timestamp','x','y','z','qx','qy','qz','qw'])
    gps_df = pd.read_csv(args.gps)
    poses_df['timestamp'] = poses_df['timestamp'].round(6)
    gps_df['timestamp'] = gps_df['timestamp'].round(6)

    # 这里的 merged 包含了所有 GPS 和 Pose 重合的时间点（全量）
    merged = pd.merge(poses_df, gps_df, on='timestamp', suffixes=('_pose', '_gps'))
    if args.status is not None:
        merged = merged[merged['status'] == args.status]
    
    if len(merged) < 3:
        print("Error: Not enough matching data."); return

    # 2. 确定用于“计算外参”的数据段
    diffs = np.diff(merged[['x_pose', 'y_pose', 'z_pose']].values, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    
    idx_threshold = np.where(cum_dist >= args.min_dist)[0]
    if len(idx_threshold) == 0:
        calc_data = merged
        print(f"Warning: Total distance {cum_dist[-1]:.2f}m < {args.min_dist}m. Using all matching points.")
    else:
        end_idx = idx_threshold[0]
        calc_data = merged.iloc[:end_idx + 1]
        print(f"Alignment parameters calculated using initial segment: {cum_dist[end_idx]:.2f} meters.")

    # 3. 计算对齐外参 (基于截取的段)
    src_pts_calc = calc_data[['x_pose', 'y_pose', 'z_pose']].values
    dst_pts_calc = calc_data[['x_gps', 'y_gps', 'z_gps']].values
    
    if args.mode == 'umeyama':
        s, R_mat, t = umeyama_alignment(src_pts_calc, dst_pts_calc)
    else:
        s, R_mat, t = yaw_xyz_alignment(src_pts_calc, dst_pts_calc)

    # 4. 应用变换到“全量位姿”
    print(f"Applying transformation to ALL {len(poses_df)} poses...")
    all_raw_pts = poses_df[['x', 'y', 'z']].values
    all_raw_quat = poses_df[['qx', 'qy', 'qz', 'qw']].values
    
    all_transformed_pts = s * (all_raw_pts @ R_mat.T) + t
    r_align = R_tool.from_matrix(R_mat)
    r_raw = R_tool.from_quat(all_raw_quat)
    all_transformed_quat = (r_align * r_raw).as_quat()

    # 5. 保存结果文件
    # (1) 保存计算出的外参
    q_align = r_align.as_quat()
    params = np.array([t[0], t[1], t[2], q_align[0], q_align[1], q_align[2], q_align[3], s])
    np.savetxt(f"{args.output_dir}/alignment_params.txt", params.reshape(1, -1), 
               header="tx ty tz qx qy qz qw scale", fmt="%.8f")

    # (2) 保存全量变换后的 Pose 轨迹 (TUM)
    full_pose_tum = np.column_stack((poses_df['timestamp'].values, all_transformed_pts, all_transformed_quat))
    np.savetxt(f"{args.output_dir}/transformed_poses_full_tum.txt", full_pose_tum, 
               fmt="%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f")

    # (3) 保存全量的 GPS 轨迹 (TUM)，包含整个行程中所有匹配的点
    full_gps_pts = merged[['x_gps', 'y_gps', 'z_gps']].values
    full_gps_quat = np.tile(np.array([0, 0, 0, 1]), (len(merged), 1))
    full_gps_tum = np.column_stack((merged['timestamp'].values, full_gps_pts, full_gps_quat))
    np.savetxt(f"{args.output_dir}/gps_full_tum.txt", full_gps_tum, fmt="%.6f")

    # 6. 输出对比信息
    rmse_subset = np.sqrt(np.mean(np.sum((dst_pts_calc - (s * (src_pts_calc @ R_mat.T) + t))**2, axis=1)))
    print("-" * 40)
    print(f"Calculation RMSE (on {args.min_dist}m subset): {rmse_subset:.4f} m")
    print(f"Output Files:")
    print(f"  - [Pose Full]: transformed_poses_full_tum.txt ({len(poses_df)} pts)")
    print(f"  - [GPS Full]:  gps_full_tum.txt ({len(merged)} pts)")
    print("-" * 40)

if __name__ == "__main__":
    main()