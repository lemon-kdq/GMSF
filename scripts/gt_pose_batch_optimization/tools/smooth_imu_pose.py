import numpy as np
import gtsam
from gtsam.symbol_shorthand import X
import argparse

def load_tum_poses(file_path):
    """读取 TUM 格式文件"""
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            d = list(map(float, line.split()))
            ts = d[0]
            t = gtsam.Point3(d[1], d[2], d[3])
            q = gtsam.Rot3.Quaternion(d[7], d[4], d[5], d[6]) # w, x, y, z
            poses.append((ts, gtsam.Pose3(q, t)))
    return sorted(poses, key=lambda x: x[0])

def interpolate_pose(t, t1, p1, t2, p2):
    """在 t1 和 t2 之间插值得到 t 时刻的位姿"""
    ratio = (t - t1) / (t2 - t1)
    # 旋转插值：SLERP
    rot = p1.rotation().slerp(ratio, p2.rotation())
    # 位移插值：Linear
    pos = p1.translation() + ratio * (p2.translation() - p1.translation())
    return gtsam.Pose3(rot, pos)

def main():
    parser = argparse.ArgumentParser(description="Smooth high-freq trajectory using optimized keyframes.")
    parser.add_argument("-hf", "--high_freq", required=True, help="Original high-freq TUM poses")
    parser.add_argument("-kf", "--keyframes", required=True, help="Optimized keyframe TUM poses")
    parser.add_argument("-o", "--output", default="smoothed_trajectory.txt", help="Output TUM path")
    args = parser.parse_args()

    # 1. 加载数据
    hf_data = load_tum_poses(args.high_freq)
    kf_data = load_tum_poses(args.keyframes)
    
    # 2. 合并并排序所有需要优化的时间点
    # 我们需要在高频轨迹中插入关键帧所在的时间点
    hf_timestamps = [d[0] for d in hf_data]
    kf_timestamps = [d[0] for d in kf_data]
    
    # 构建初始 Values 和 骨架位姿
    # 为了方便查找，先转成字典
    hf_dict = dict(hf_data)
    
    all_points = [] # 存储 (timestamp, initial_guess_pose, is_keyframe, kf_pose)
    
    hf_idx = 0
    for kf_ts, kf_pose in kf_data:
        # 寻找 kf_ts 在 hf_data 中的位置并插值
        while hf_idx < len(hf_data) - 1 and hf_data[hf_idx+1][0] < kf_ts:
            all_points.append((hf_data[hf_idx][0], hf_data[hf_idx][1], False, None))
            hf_idx += 1
            
        # 插值得到关键帧时刻的原始位姿作为初值
        t1, p1 = hf_data[hf_idx]
        t2, p2 = hf_data[hf_idx+1]
        p_interp = interpolate_pose(kf_ts, t1, p1, t2, p2)
        all_points.append((kf_ts, p_interp, True, kf_pose))
        
    # 添加剩余的高频点
    for i in range(hf_idx + 1, len(hf_data)):
        all_points.append((hf_data[i][0], hf_data[i][1], False, None))

    # 3. 构建因子图
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    
    # 噪声模型
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]))
    kf_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]))

    for i in range(len(all_points)):
        ts, p_init, is_kf, kf_pose = all_points[i]
        initial_values.insert(X(i), p_init)
        
        # 如果是关键帧，添加位姿先验约束（锁定到优化后的位置）
        if is_kf:
            graph.add(gtsam.PriorFactorPose3(X(i), kf_pose, kf_prior_noise))
            
        # 连接相邻帧（保持原始 IMU 轨迹的形状）
        if i > 0:
            p_prev_init = all_points[i-1][1]
            rel_pose = p_prev_init.between(p_init)
            graph.add(gtsam.BetweenFactorPose3(X(i-1), X(i), rel_pose, odom_noise))

    # 4. 优化
    print("Optimizing trajectory...")
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
    result = optimizer.optimize()

    # 5. 保存结果
    print(f"Saving smoothed trajectory to {args.output}")
    with open(args.output, 'w') as f:
        for i in range(len(all_points)):
            ts = all_points[i][0]
            pose = result.atPose3(X(i))
            t = pose.translation()
            q = pose.rotation().toQuaternion()
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q.x():.6f} {q.y():.6f} {q.z():.6f} {q.w():.6f}\n")

if __name__ == "__main__":
    main()