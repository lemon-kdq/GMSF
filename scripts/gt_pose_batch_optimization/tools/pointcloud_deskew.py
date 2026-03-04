from gt_tools.pcd.pcd_io import load_pcd_at_pointxyzinormal,write_pcd_with_array_in_pointxyzinormal 
import numpy as np
from pathlib import Path
import argparse
from scipy.spatial.transform import Rotation as R, Slerp



def read_tum_poses(tum_file):
    """
    读取TUM格式位姿: timestamp x y z qx qy qz qw
    """
    data = np.loadtxt(tum_file)
    timestamps = data[:,0]
    positions = data[:,1:4]
    quaternions = data[:,4:8]
    return timestamps, positions, quaternions

def interpolate_quat(q0,q1,t0,t1,tt): 
    rots = R.from_quat([q0,q1])
    slerp = Slerp(np.array([t0,t1]),rots)
    rot_t = slerp([tt])[0]
    return rot_t.as_quat()
    
    
def interpolate_pose(timestamps, positions, quaternions, query_time):
    """
    根据时间戳线性插值位姿（位置和旋转）
    """
    if query_time <= timestamps[0]:
        return positions[0], quaternions[0]
    if query_time >= timestamps[-1]:
        return positions[-1], quaternions[-1]

    idx = np.searchsorted(timestamps, query_time) - 1
    t0, t1 = timestamps[idx], timestamps[idx+1]
    p0, p1 = positions[idx], positions[idx+1]
    q0, q1 = quaternions[idx], quaternions[idx+1]

    ratio = (query_time - t0) / (t1 - t0)
    # 线性插值位置
    p_interp = p0 + ratio * (p1 - p0)
    # slerp 旋转
    q_interp = interpolate_quat(q0,q1,t0,t1,query_time)
    return p_interp, q_interp

def deskew_pointcloud(pc_np, lidar_ts, target_ts, pose_timestamps, positions, quaternions):
    """
    对点云逐点正畸
    pc_np: N x 8 array
    curvature 列存储点相对于第一点的时间偏差
    lidar_ts: LiDAR时间戳
    target_ts: 目标时间戳
    """
    pc_corrected = np.copy(pc_np)
    N = pc_np.shape[0]
    p_target, q_target = interpolate_pose(pose_timestamps, positions, quaternions, target_ts)
    r_target = R.from_quat(q_target)

    for i in range(N):
        dt = pc_np[i,7]  # curvature列: 点相对于第一个点时间差
        point_time = lidar_ts + dt * 1e-3
        # 插值 LiDAR 当前点位姿
        p_point, q_point = interpolate_pose(pose_timestamps, positions, quaternions, point_time)
        # 将点从 LiDAR frame 转到世界，再再转换到目标时刻 frame
        r_point = R.from_quat(q_point)
        pt = pc_np[i,:3]
        pt_world = r_point.apply(pt) + p_point
        pt_corrected = r_target.inv().apply(pt_world - p_target)
        pc_corrected[i,:3] = pt_corrected
    return pc_corrected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pose_file", help="TUM格式位姿文件")
    parser.add_argument("timestamp_txt", help="txt文件: 第一列LiDAR时间戳，第二列目标时间戳")
    parser.add_argument("pcd_dir", help="pcd文件夹路径")
    parser.add_argument("--out_dir", help="保存正畸pcd", default="./deskewed")
    args = parser.parse_args()

    pose_timestamps, positions, quaternions = read_tum_poses(args.pose_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.timestamp_txt) as f:
        lines = f.readlines()

    for line in lines[1:]:  # 如果第一行是表头
        lidar_ts_str, target_ts_str = line.strip().split(',')
        lidar_ts = float(lidar_ts_str)
        target_ts = float(target_ts_str)
        pcd_path = Path(args.pcd_dir) / f"{lidar_ts_str}.pcd"
        if not pcd_path.exists():
            print(f"PCD不存在: {pcd_path}")
            continue
        print(f"pcd : {pcd_path}")
        pc_np = load_pcd_at_pointxyzinormal(pcd_path)
        
        pc_corrected = deskew_pointcloud(pc_np, lidar_ts, target_ts, pose_timestamps, positions, quaternions)

        out_path = out_dir / f"{target_ts}.pcd"
        # 保存为新的pcd
        write_pcd_with_array_in_pointxyzinormal(pc_corrected,out_path)
        print(f"保存正畸点云: {out_path}")

if __name__ == "__main__":
    main()