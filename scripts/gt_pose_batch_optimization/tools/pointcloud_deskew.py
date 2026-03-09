from gt_tools.pcd.pcd_io import load_pcd_at_pointrgbal,write_pcd_with_array_in_pointrgbal 
import numpy as np
from pathlib import Path
import argparse
from scipy.spatial.transform import Rotation as R, Slerp
from dataclasses import dataclass

@dataclass
class PoseTUM:
    timestamp: float
    position: np.ndarray   # (3,)
    quaternion: np.ndarray # (4,) xyzw

    def to_tum_line(self) -> str:
        x, y, z = self.position
        qx, qy, qz, qw = self.quaternion
        return f"{self.timestamp:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"

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



def deskew_pointcloud_batch(
    pc_np,
    lidar_ts,
    target_ts,
    pose_timestamps,
    positions,
    quaternions,
    batch_ms=1.0
):
    """
    按 curvature 时间分桶（每 batch_ms ms）进行批量正畸
    """

    # ---------- 1. 按 curvature 排序 ----------
    order = np.argsort(pc_np[:, 4] * 1e-6) 
    pc = pc_np[order]

    pc_corrected = np.copy(pc)

    # ---------- 2. 目标时刻位姿 ----------
    p_target, q_target = interpolate_pose(
        pose_timestamps, positions, quaternions, target_ts
    )
    r_target = R.from_quat(q_target)

    target_pose = PoseTUM(
        timestamp=target_ts,
        position=p_target,
        quaternion=q_target
    )
    # ---------- 3. 生成时间 batch id ----------
    # curvature 是 ms
    batch_ids = np.floor(pc[:, 4] * 1e-6 / batch_ms).astype(np.int64)
    unique_batches = np.unique(batch_ids)

    # ---------- 4. 按 batch 处理 ----------
    for bid in unique_batches:
        mask = batch_ids == bid
        idx = np.where(mask)[0]

        # 该 batch 的代表时间（用第一个点即可）
        dt_ns = pc[idx[0], 4]
        point_time = lidar_ts + dt_ns * 1e-9

        # 插值该 batch 的位姿
        p_point, q_point = interpolate_pose(
            pose_timestamps, positions, quaternions, point_time
        )
        r_point = R.from_quat(q_point)

        # ---------- 批量旋转 ----------
        pts = pc[idx, :3]                          # (M,3)
        pts_world = r_point.apply(pts) + p_point   # (M,3)
        pts_corrected = r_target.inv().apply(
            pts_world - p_target
        )

        pc_corrected[idx, :3] = pts_corrected

    # ---------- 5. 恢复原始点顺序 ----------
    pc_out = np.empty_like(pc_corrected)
    pc_out[order] = pc_corrected

    return pc_out,target_pose

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
        dt = pc_np[i,4]  # curvature列: 点相对于第一个点时间差
        point_time = lidar_ts + dt * 1e-9
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
    parser.add_argument("keyframe_file", help="txt文件: 第一列LiDAR时间戳，第二列目标时间戳")
    parser.add_argument("pcd_dir", help="pcd文件夹路径")
    parser.add_argument("out_dir", help="保存正畸pcd", default="./deskewed")
    args = parser.parse_args()

    pose_timestamps, positions, quaternions = read_tum_poses(args.pose_file)
    ds_pcd_dir = Path(f"{args.out_dir}/pcd")
    ds_pcd_dir.mkdir(parents=True, exist_ok=True)
    txt_path = f"{args.out_dir}/target_pose.txt"
    with open(args.keyframe_file) as f:
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
        pc_np = load_pcd_at_pointrgbal(pcd_path)
        
        pc_corrected,tar_pos = deskew_pointcloud_batch(pc_np, lidar_ts, target_ts, pose_timestamps, positions, quaternions)

        out_path = ds_pcd_dir / f"{target_ts:.6f}.pcd"
        # 保存为新的pcd
        write_pcd_with_array_in_pointrgbal(pc_corrected,out_path)
        #print(f"保存正畸点云: {out_path}")
        with open(txt_path, 'a') as wf:
            wf.write(tar_pos.to_tum_line())
if __name__ == "__main__":
    main()