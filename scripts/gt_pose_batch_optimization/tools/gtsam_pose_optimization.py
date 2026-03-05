import numpy as np
import gtsam
from gt_tools.pcd.pointcloud_feature_detection import find_point_to_plane_matches
from gt_tools.pcd.pcd_io import load_ply
from gt_tools.pcd.pointcloud_filter import voxel_filter
from gtsam.symbol_shorthand import X
import argparse 
import os
class LidarPlaneOptimizer:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.timestamps = [] # 用于保存原始时间戳

        # --- 噪声模型 ---
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.2, 2.0, 2.0, 0.5]))
        self.horizontal_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01]))
        self.plane_noise_model = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(0.1),
            gtsam.noiseModel.Isotropic.Sigma(1, 0.1)
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
    
    def pose3_to_4x4_manual(self,pose):
        # 1. 提取旋转矩阵 (3x3 numpy array)
        R = pose.rotation().matrix()

        # 2. 提取平移向量 (numpy array [x, y, z])
        t = pose.translation()

        # 3. 构造 4x4 恒等矩阵
        T = np.eye(4)

        # 4. 填充旋转部分
        T[:3, :3] = R

        # 5. 填充平移部分 (注意：t 可能是 [3,] 或 [3,1]，直接赋值给 [0:3, 3] 即可)
        T[:3, 3] = t

        return T
    def add_point_to_plane_constraint(self, i_prev, i_curr, p_curr, n_prev, d_prev):
        """
        修正后的点面约束函数
        p_curr: numpy array [3] - 当前帧坐标系下的点
        n_prev: numpy array [3] - 前一帧坐标系下的法向量
        d_prev: float - 前一帧坐标系下的平面截距 (n.dot(p) = d)
        """

        # 强制转换为 float64，防止 GTSAM 内部类型不匹配
        p_curr = p_curr.astype(np.float64)
        n_prev = n_prev.astype(np.float64)

        def error_func(this_factor, values, jacobians=None):
            try:
                # 1. 提取当前位姿
                pose_prev = values.atPose3(X(i_prev))
                pose_curr = values.atPose3(X(i_curr))

                # 2. 计算相对变换 T_rel = pose_prev^-1 * pose_curr
                # H_prev 是 T_rel 对 pose_prev 的导数 (6x6)
                # H_curr 是 T_rel 对 pose_curr 的导数 (6x6)
                H_prev = np.zeros((6, 6), order='F')
                H_curr = np.zeros((6, 6), order='F')
                T_rel = pose_prev.between(pose_curr, H_prev, H_curr)

                # 3. 将点变换到 prev 坐标系: p_in_prev = T_rel * p_curr
                # D_p_Trel 是变换后的点对 T_rel 的导数 (3x6)
                # D_p_pt 是变换后的点对原始点 p_curr 的导数 (3x3)，此处不使用但必须传入
                D_p_Trel = np.zeros((3, 6), order='F')
                D_p_pt = np.zeros((3, 3), order='F')
                p_in_prev = T_rel.transformFrom(p_curr, D_p_Trel, D_p_pt)

                # 4. 计算残差 e = n^T * p' - d
                # 确保结果是一个 size=1 的 numpy 数组
                error_val = np.dot(n_prev, p_in_prev) - d_prev

                # 5. 链式法则计算雅可比
                if jacobians is not None:
                    # de/dp' = n_prev^T (1x3)
                    de_dp = n_prev.reshape(1, 3)

                    # de/dT_rel = de/dp' * dp'/dT_rel (1x6)
                    de_dTrel = de_dp @ D_p_Trel

                    # 最终对两个位姿变量的导数
                    jacobians[0] = de_dTrel @ H_prev # 对 X(i-1)
                    jacobians[1] = de_dTrel @ H_curr # 对 X(i)

                return np.array([error_val])

            except Exception as e:
                print(f"Error in CustomFactor: {e}")
                return np.array([0.0])  

        # --- 关键：噪声平衡 ---
        # 如果点面约束很多，Sigma 不能设得太小，否则会压制里程计
        # 建议设为 0.05m ~ 0.2m
        factor = gtsam.CustomFactor(self.plane_noise_model, [X(i_prev), X(i_curr)], error_func)
        self.graph.add(factor)
        
    def add_point_cloud_constraints(self, i , j, src_pcd, tar_pcd, Tws, Twt):
        """
        feature_points: 当前帧 (i_curr) 提取的特征点列表 [p1, p2, p3, ...]
        planes_in_prev: 对应的前一帧 (i_prev) 平面参数 [(n1, d1), (n2, d2), ...]
        """

        matched_infos = find_point_to_plane_matches(src_pcd,tar_pcd,Tws, Twt)
        print(f"{i}-{j} p2p size: ",len(matched_infos))
        
        count = 0
        for m in matched_infos:
            # 为每一个匹配对调用之前定义的函数
            # 每一个点都会关联到一个 Huber 鲁棒核函数，独立处理离群点
            count = count + 1
            if count < 1000:
                self.add_point_to_plane_constraint( j,i, m["p_src"], m["n_tar"], m["d_tar"]) 
    
      
            

    def build_graph(self, initial_poses,pcd_folder,voxel_size = 0.5):
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
            
            # 点到面约束
            Tws = self.pose3_to_4x4_manual(pose_i)
            j = i - 1
            if j > 0: 
                src_path = os.path.join(pcd_folder,f"{self.timestamps[i]:.6f}.pcd")
                tar_path = os.path.join(pcd_folder,f"{self.timestamps[j]:.6f}.pcd")
                src_pcd = load_ply(src_path,True)
                src_vf_pcd = voxel_filter(src_pcd,voxel_size)
                tar_pcd = load_ply(tar_path,True)
                Twt = self.pose3_to_4x4_manual(initial_poses[j])
                self.add_point_cloud_constraints(i,j, src_vf_pcd, tar_pcd, Tws, Twt)
                

    def optimize(self):
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")
        params.setMaxIterations(20)
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
    parser.add_argument('--pcd_folder', type=str, required=True, help='pcd folder')
    parser.add_argument('--output_file', type=str, required=True, help='轮速数据文件路径')

    args = parser.parse_args()    
    opt = LidarPlaneOptimizer()
    
    # 1. 加载并保存时间戳
    poses_guess = opt.load_tum_poses(args.pose_file)
    
    # 2. 构网
    opt.build_graph(poses_guess, args.pcd_folder)
    

    # 4. 优化
    final_result = opt.optimize()
    
    # 5. 保存为 TUM 格式
    opt.save_results_tum(final_result, args.output_file)