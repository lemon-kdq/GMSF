import numpy as np
import gtsam
from gtsam.symbol_shorthand import X
import argparse

def load_tum_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            d = list(map(float, line.split()))
            ts = d[0]
            t = gtsam.Point3(d[1], d[2], d[3])
            q = gtsam.Rot3.Quaternion(d[7], d[4], d[5], d[6])
            poses.append((ts, gtsam.Pose3(q, t)))
    return sorted(poses, key=lambda x: x[0])

def interpolate_pose(t, t1, p1, t2, p2):
    ratio = (t - t1) / (t2 - t1)
    rot = p1.rotation().slerp(ratio, p2.rotation())
    pos = p1.translation() + ratio * (p2.translation() - p1.translation())
    return gtsam.Pose3(rot, pos)

def main():
    parser = argparse.ArgumentParser(description="Smooth trajectory with dynamic noise.")
    parser.add_argument("-hf", "--high_freq", required=True)
    parser.add_argument("-kf", "--keyframes", required=True)
    parser.add_argument("-o", "--output", default="smoothed_trajectory_v1.txt")
    args = parser.parse_args()

    hf_data = load_tum_poses(args.high_freq)
    kf_data = load_tum_poses(args.keyframes)
    
    if not hf_data or not kf_data: return

    # 数据合并逻辑保持不变
    all_points = []
    hf_idx = 0
    num_hf = len(hf_data)
    for kf_ts, kf_pose in kf_data:
        while hf_idx < num_hf and hf_data[hf_idx][0] < kf_ts:
            all_points.append((hf_data[hf_idx][0], hf_data[hf_idx][1], False, None))
            hf_idx += 1
        if hf_idx == 0: p_init = hf_data[0][1]
        elif hf_idx >= num_hf: p_init = hf_data[-1][1]
        else:
            t_prev, p_prev = hf_data[hf_idx-1]
            t_next, p_next = hf_data[hf_idx]
            if abs(kf_ts - t_prev) < 1e-7: p_init = p_prev
            else: p_init = interpolate_pose(kf_ts, t_prev, p_prev, t_next, p_next)
        all_points.append((kf_ts, p_init, True, kf_pose))
    while hf_idx < num_hf:
        all_points.append((hf_data[hf_idx][0], hf_data[hf_idx][1], False, None))
        hf_idx += 1
    all_points.sort(key=lambda x: x[0])

    # --- 优化部分：引入动态噪声 ---
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    
    # 定义单位时间（1秒）的基础噪声
    # 数组意义：[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    base_sigma = np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3])
    # 关键帧先验噪声（稍微调大一点，不要太“硬”，给平滑留一点余地）
    kf_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]))

    for i in range(len(all_points)):
        ts, p_init, is_kf, kf_pose = all_points[i]
        initial_values.insert(X(i), p_init)
        
        if is_kf:
            graph.add(gtsam.PriorFactorPose3(X(i), kf_pose, kf_prior_noise))
            
        if i > 0:
            dt = ts - all_points[i-1][0]
            if dt <= 0: dt = 1e-6 # 防止时间戳完全重合
            
            # 【关键修改】动态计算噪声：噪声随 dt 线性缩放
            # 时间差越小，sigma 越小，权重越大，约束越硬
            step_sigma = base_sigma * dt 
            odom_noise = gtsam.noiseModel.Diagonal.Sigmas(step_sigma)
            
            p_prev_init = all_points[i-1][1]
            rel_pose = p_prev_init.between(p_init)
            graph.add(gtsam.BetweenFactorPose3(X(i-1), X(i), rel_pose, odom_noise))

    print(f"Optimizing {len(all_points)} states with dynamic noise...")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values)
    result = optimizer.optimize()

    # 保存结果
    output_buffer = []
    for i in range(len(all_points)):
        ts = all_points[i][0]
        pose = result.atPose3(X(i))
        t = pose.translation(); q = pose.rotation().toQuaternion()
        output_buffer.append((ts, t[0], t[1], t[2], q.x(), q.y(), q.z(), q.w()))

    with open(args.output, 'w') as f:
        for row in output_buffer:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} "
                    f"{row[4]:.6f} {row[5]:.6f} {row[6]:.6f} {row[7]:.6f}\n")
    print("Done.")

if __name__ == "__main__":
    main()