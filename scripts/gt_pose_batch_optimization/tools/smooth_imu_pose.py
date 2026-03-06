import numpy as np
import gtsam
from gtsam.symbol_shorthand import X
import argparse

def load_tum_poses(file_path):
    """读取 TUM 格式文件并按时间戳排序"""
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            d = list(map(float, line.split()))
            ts = d[0]
            t = gtsam.Point3(d[1], d[2], d[3])
            q = gtsam.Rot3.Quaternion(d[7], d[4], d[5], d[6]) # w, x, y, z
            poses.append((ts, gtsam.Pose3(q, t)))
    # 强制排序
    return sorted(poses, key=lambda x: x[0])

def interpolate_pose(t, t1, p1, t2, p2):
    """在 t1 和 t2 之间插值得到 t 时刻的位姿"""
    ratio = (t - t1) / (t2 - t1)
    # 旋转插值：SLERP (球面线性插值)
    rot = p1.rotation().slerp(ratio, p2.rotation())
    # 位移插值：Linear
    pos = p1.translation() + ratio * (p2.translation() - p1.translation())
    return gtsam.Pose3(rot, pos)

def main():
    parser = argparse.ArgumentParser(description="Smooth high-freq trajectory with sorted output.")
    parser.add_argument("-hf", "--high_freq", required=True, help="Original high-freq TUM poses")
    parser.add_argument("-kf", "--keyframes", required=True, help="Optimized keyframe TUM poses")
    parser.add_argument("-o", "--output", default="smoothed_trajectory.txt", help="Output TUM path")
    args = parser.parse_args()

    # 1. 加载并排序数据
    hf_data = load_tum_poses(args.high_freq)
    kf_data = load_tum_poses(args.keyframes)
    
    if not hf_data or not kf_data:
        print("Error: Input files are empty."); return

    # 2. 构建统一的时间轴
    # all_points 存储结构: (timestamp, initial_guess_pose, is_keyframe, target_kf_pose)
    all_points = []
    hf_idx = 0
    num_hf = len(hf_data)

    for kf_ts, kf_pose in kf_data:
        # 将当前关键帧时间戳之前的所有高频点加入
        while hf_idx < num_hf and hf_data[hf_idx][0] < kf_ts:
            all_points.append((hf_data[hf_idx][0], hf_data[hf_idx][1], False, None))
            hf_idx += 1
        
        # 处理关键帧：如果时间戳不完全重合，则插值得到初值
        # 寻找最近的高频点进行插值
        if hf_idx == 0:
            # 关键帧在第一个高频点之前
            p_init = hf_data[0][1]
        elif hf_idx >= num_hf:
            # 关键帧在最后一个高频点之后
            p_init = hf_data[-1][1]
        else:
            # 在 hf_idx-1 和 hf_idx 之间插值
            t_prev, p_prev = hf_data[hf_idx-1]
            t_next, p_next = hf_data[hf_idx]
            # 如果时间非常接近，直接用高频点，否则插值
            if abs(kf_ts - t_prev) < 1e-7:
                p_init = p_prev
            else:
                p_init = interpolate_pose(kf_ts, t_prev, p_prev, t_next, p_next)
        
        all_points.append((kf_ts, p_init, True, kf_pose))

    # 添加剩余的高频点
    while hf_idx < num_hf:
        all_points.append((hf_data[hf_idx][0], hf_data[hf_idx][1], False, None))
        hf_idx += 1

    # 再次确保整体排序（防止由于微小浮点误差导致的乱序）
    all_points.sort(key=lambda x: x[0])

    # 3. 构建因子图
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    
    # 噪声：Between 权重设大（保证平滑），KF Prior 权重设极大（保证经过锚点）
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]))
    kf_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]))

    for i in range(len(all_points)):
        ts, p_init, is_kf, kf_pose = all_points[i]
        initial_values.insert(X(i), p_init)
        
        # 约束1：如果该节点是关键帧，强行锚定
        if is_kf:
            graph.add(gtsam.PriorFactorPose3(X(i), kf_pose, kf_prior_noise))
            
        # 约束2：连接相邻帧
        if i > 0:
            # 计算原始轨迹中的相对增量作为观测值
            p_prev_init = all_points[i-1][1]
            rel_pose = p_prev_init.between(p_init)
            graph.add(gtsam.BetweenFactorPose3(X(i-1), X(i), rel_pose, odom_noise))

    # 4. 执行优化
    print(f"Optimizing trajectory with {len(all_points)} states...")
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("TERMINATION")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
    result = optimizer.optimize()

    # 5. 收集结果并最终排序
    output_buffer = []
    for i in range(len(all_points)):
        ts = all_points[i][0]
        pose = result.atPose3(X(i))
        t = pose.translation()
        q = pose.rotation().toQuaternion()
        output_buffer.append((ts, t[0], t[1], t[2], q.x(), q.y(), q.z(), q.w()))

    # 写入前按时间戳排序
    output_buffer.sort(key=lambda x: x[0])

    print(f"Saving smoothed trajectory to {args.output}")
    with open(args.output, 'w') as f:
        for row in output_buffer:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} "
                    f"{row[4]:.6f} {row[5]:.6f} {row[6]:.6f} {row[7]:.6f}\n")
    print("Optimization and saving completed.")

if __name__ == "__main__":
    main()