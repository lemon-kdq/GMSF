import numpy as np
import gtsam
from gtsam.symbol_shorthand import X
import argparse 

class LidarPlaneOptimizer:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.timestamps = [] # 用于保存原始时间戳

        # --- 噪声模型 ---
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05]))
        self.horizontal_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001]))
        self.plane_noise_model = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(0.1),
            gtsam.noiseModel.Isotropic.Sigma(1, 0.05)
        )

    def load_tum_poses(self, file_path):
        """
        读取 TUM 位姿并保存时间戳
        """
        poses = []
        self.timestamps = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                d = list(map(float, line.split()))
                self.timestamps.append(d[0]) # 保存第一列时间戳
                t = gtsam.Point3(d[1], d[2], d[3])
                # TUM: qx(4), qy(5), qz(6), qw(7)
                # GTSAM: Quaternion(w, x, y, z)
                q = gtsam.Rot3.Quaternion(d[7], d[4], d[5], d[6]) 
                poses.append(gtsam.Pose3(q, t))
        return poses

    def build_graph(self, initial_poses):
        num_poses = len(initial_poses)
        up_vector = gtsam.Unit3(np.array([0.0, 0.0, 1.0]))

        for i in range(num_poses):
            pose_i = initial_poses[i]
            self.initial_values.insert(X(i), pose_i)

            if i == 0:
                self.graph.add(gtsam.PriorFactorPose3(X(0), pose_i, 
                               gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))))
                continue

            # 相邻帧约束
            rel_pose = initial_poses[i-1].between(pose_i)
            self.graph.add(gtsam.BetweenFactorPose3(X(i-1), X(i), rel_pose, self.odom_noise))

            # 水平姿态保持 (锁定初值中的 Roll/Pitch)
            n_body_init = pose_i.rotation().rotate(up_vector)
            self.graph.add(gtsam.Pose3AttitudeFactor(X(i), n_body_init, 
                                                    self.horizontal_noise, up_vector))

    def optimize(self):
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_values, params)
        return optimizer.optimize()

    def save_results_tum(self, result, output_path):
        """
        按照 TUM 格式保存结果: timestamp tx ty tz qx qy qz qw
        """
        print(f"Saving results to {output_path}...")
        with open(output_path, 'w') as f:
            for i in range(len(self.timestamps)):
                key = X(i)
                if result.exists(key):
                    pose = result.atPose3(key)
                    
                    # 1. 获取位移 (返回的是 numpy 数组)
                    t = pose.translation() 
                    
                    # 2. 获取四元数 (GTSAM Quaternion 对象)
                    q = pose.rotation().toQuaternion()
                    
                    # 3. 写入文件
                    # t[0], t[1], t[2] 分别对应 x, y, z
                    # q.x(), q.y(), q.z(), q.w() 是提取四元数分量的标准做法
                    f.write(f"{self.timestamps[i]:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                            f"{q.x():.6f} {q.y():.6f} {q.z():.6f} {q.w():.6f}\n")
        print("Done.")

# --- 执行流程 ---
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='使用轮速和IMU姿态进行航位推算（完整外参模 型）')
    parser.add_argument('--pose_file', type=str, required=True, help='轮速数据文件路径')
    parser.add_argument('--output_file', type=str, required=True, help='轮速数据文件路径')

    args = parser.parse_args()    
    opt = LidarPlaneOptimizer()
    
    # 1. 加载并保存时间戳
    poses_guess = opt.load_tum_poses(args.pose_file)
    
    # 2. 构网
    opt.build_graph(poses_guess)
    
    # 3. 此处可以添加你的点面约束 (add_point_to_plane_constraint)
    # ...
    
    # 4. 优化
    final_result = opt.optimize()
    
    # 5. 保存为 TUM 格式
    opt.save_results_tum(final_result, args.output_file)