import numpy as np
import gtsam
import pandas as pd
import argparse 
import os
from gtsam.symbol_shorthand import X, A, L # X:位姿, A:对齐外参, L:GPS杆臂

# 你的自定义工具库
from gt_tools.pcd.pointcloud_feature_detection import find_point_to_plane_matches
from gt_tools.pcd.pcd_io import load_ply
from gt_tools.pcd.pointcloud_filter import voxel_filter

class LidarPlaneOptimizer:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.timestamps = [] 

        # --- 噪声模型 ---
        # 1. 里程计噪声 (旋转, 平移)
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.1, 0.5, 0.5, 0.3]))
        # 2. 水平姿态噪声 (Roll, Pitch)
        self.horizontal_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01]))
        # 3. 点面约束噪声 (带 Huber 鲁棒核)
        self.plane_noise_model = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(0.1),
            gtsam.noiseModel.Isotropic.Sigma(1, 0.1)
        )
        # 4. GPS 噪声 (带 Huber 鲁棒核, 重点加大 Z 轴噪声)
        self.gps_noise_model = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(0.5),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([0.25, 0.25, 0.5])) # E, N, U
        )

    def _skew(self, v):
        """反对称矩阵"""
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def load_tum_poses(self, file_path):
        poses = []
        self.timestamps = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                d = list(map(float, line.split()))
                self.timestamps.append(d[0]) 
                t = gtsam.Point3(d[1], d[2], d[3])
                q = gtsam.Rot3.Quaternion(d[7], d[4], d[5], d[6]) 
                poses.append(gtsam.Pose3(q, t))
        return poses
    
    def pose3_to_4x4_manual(self, pose):
        R = pose.rotation().matrix()
        t = pose.translation()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    # --- GPS 约束逻辑 (三元因子) ---
    def add_gps_constraint(self, i, gps_measurement):
        """模型: z = Rew * (pwi + Rwi * Pig) + Pew"""
        def error_func(this_factor, values, jacobians=None):
            T_WI = values.atPose3(X(i))
            T_EW = values.atPose3(A(0)) # 全局对齐外参
            l_IG = values.atPoint3(L(0)) # 杆臂 Pig
            
            R_WI, p_WI = T_WI.rotation().matrix(), T_WI.translation()
            R_EW, p_EW = T_EW.rotation().matrix(), T_EW.translation()
            
            p_WG = p_WI + R_WI @ l_IG
            z_hat = R_EW @ p_WG + p_EW
            error = z_hat - gps_measurement
            
            if jacobians is not None:
                # 对 X(i) 的导数
                H_pose = np.zeros((3, 6), order='F')
                H_pose[:, :3] = R_EW @ (-R_WI @ self._skew(l_IG))
                H_pose[:, 3:] = R_EW
                jacobians[0] = H_pose
                # 对 A(0) 的导数
                H_align = np.zeros((3, 6), order='F')
                H_align[:, :3] = -self._skew(R_EW @ p_WG)
                H_align[:, 3:] = np.eye(3)
                jacobians[1] = H_align
                # 对 L(0) 的导数
                jacobians[2] = R_EW @ R_WI
            return error

        factor = gtsam.CustomFactor(self.gps_noise_model, [X(i), A(0), L(0)], error_func)
        self.graph.add(factor)

    # --- 原始点到面约束逻辑 ---
    def add_point_to_plane_constraint(self, i_prev, i_curr, p_curr, n_prev, d_prev):
        p_curr = p_curr.astype(np.float64)
        n_prev = n_prev.astype(np.float64)

        def error_func(this_factor, values, jacobians=None):
            try:
                pose_prev = values.atPose3(X(i_prev))
                pose_curr = values.atPose3(X(i_curr))
                H_prev, H_curr = np.zeros((6, 6), order='F'), np.zeros((6, 6), order='F')
                T_rel = pose_prev.between(pose_curr, H_prev, H_curr)
                D_p_Trel, D_p_pt = np.zeros((3, 6), order='F'), np.zeros((3, 3), order='F')
                p_in_prev = T_rel.transformFrom(p_curr, D_p_Trel, D_p_pt)
                error_val = np.dot(n_prev, p_in_prev) - d_prev
                if jacobians is not None:
                    de_dp = n_prev.reshape(1, 3)
                    de_dTrel = de_dp @ D_p_Trel
                    jacobians[0] = de_dTrel @ H_prev 
                    jacobians[1] = de_dTrel @ H_curr 
                return np.array([error_val])
            except Exception as e:
                return np.array([0.0])  

        factor = gtsam.CustomFactor(self.plane_noise_model, [X(i_prev), X(i_curr)], error_func)
        self.graph.add(factor)
        
    def add_point_cloud_constraints(self, i, j, src_pcd, tar_pcd, Tws, Twt):
        matched_infos = find_point_to_plane_matches(src_pcd, tar_pcd, Tws, Twt)
        print(f"Frame {i}-{j} matched size: {len(matched_infos)}")
        count = 0
        for m in matched_infos:
            count += 1
            if count < 1000:
                self.add_point_to_plane_constraint(j, i, m["p_src"], m["n_tar"], m["d_tar"]) 

    # --- 完整的图构建逻辑 ---
    def build_graph(self, initial_poses, pcd_folder, gps_df=None, align_init=None, lever_arm_init=None, voxel_size=0.5):
        num_poses = len(initial_poses)
        up_vector = gtsam.Unit3(np.array([0.0, 0.0, 1.0]))

        # 1. 初始化全局外参 A(0) 和 杆臂 L(0)
        if align_init is not None:
            self.initial_values.insert(A(0), align_init)
            # 锁定 Roll/Pitch (1e-6), 允许优化 Yaw (0.1) 和 XYZ (1.0)
            stiff_sigmas = np.array([1e-6, 1e-6, 0.1, 1.0, 1.0, 1.0])
            self.graph.add(gtsam.PriorFactorPose3(A(0), align_init, gtsam.noiseModel.Diagonal.Sigmas(stiff_sigmas)))
        
        if lever_arm_init is not None:
            l_arm = gtsam.Point3(*lever_arm_init)
            self.initial_values.insert(L(0), l_arm)
            self.graph.add(gtsam.PriorFactorPoint3(L(0), l_arm, gtsam.noiseModel.Isotropic.Sigma(3, 0.05)))

        # 2. 遍历每一帧位姿
        for i in range(num_poses):
            pose_i = initial_poses[i]
            ts_i = self.timestamps[i]
            self.initial_values.insert(X(i), pose_i)

            if i == 0:
                #self.graph.add(gtsam.PriorFactorPose3(X(0), pose_i, gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))))
                continue

            # A. 相邻帧增量约束
            rel_pose = initial_poses[i-1].between(pose_i)
            self.graph.add(gtsam.BetweenFactorPose3(X(i-1), X(i), rel_pose, self.odom_noise))

            # B. 水平姿态约束 (修正 Roll/Pitch)
            n_body_meas = pose_i.rotation().rotate(up_vector)
            self.graph.add(gtsam.Pose3AttitudeFactor(X(i), n_body_meas, self.horizontal_noise, up_vector))
            
            # C. GPS 全局位置约束
            if gps_df is not None:
                gps_row = gps_df[np.isclose(gps_df['timestamp'], ts_i, atol=1e-3)]
                if not gps_row.empty:
                    self.add_gps_constraint(i, gps_row[['x', 'y', 'z']].values[0])

            # D. 点到面帧间匹配约束 (你的原始逻辑)
            continue
            Tws = self.pose3_to_4x4_manual(pose_i)
            j = i - 1
            if j >= 0: 
                src_path = os.path.join(pcd_folder, f"{ts_i:.6f}.pcd")
                tar_path = os.path.join(pcd_folder, f"{self.timestamps[j]:.6f}.pcd")
                if os.path.exists(src_path) and os.path.exists(tar_path):
                    src_pcd = load_ply(src_path, True)
                    src_vf_pcd = voxel_filter(src_pcd, voxel_size)
                    tar_pcd = load_ply(tar_path, True)
                    Twt = self.pose3_to_4x4_manual(initial_poses[j])
                    self.add_point_cloud_constraints(i, j, src_vf_pcd, tar_pcd, Tws, Twt)

    def optimize(self):
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")
        params.setMaxIterations(20)
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_values, params)
        return optimizer.optimize()

    def save_results_tum(self, result, output_path):
        print(f"Saving optimized trajectory to {output_path}...")
        with open(output_path, 'w') as f:
            for i in range(len(self.timestamps)):
                if result.exists(X(i)):
                    p = result.atPose3(X(i))
                    t, q = p.translation(), p.rotation().toQuaternion()
                    f.write(f"{self.timestamps[i]:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                            f"{q.x():.6f} {q.y():.6f} {q.z():.6f} {q.w():.6f}\n")
        print("Done.")

# --- 执行入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Integrated GTSAM Optimizer')
    parser.add_argument('--pose_file', type=str, required=True)
    parser.add_argument('--pcd_folder', type=str, required=True)
    parser.add_argument('--gps_file', type=str, help='Interpolated GPS CSV file')
    parser.add_argument('--align_file', type=str, help='Alignment params file (Umeyama)')
    parser.add_argument('--lever_arm', type=float, nargs=3, default=[0,0,0], help='Lever arm x y z')
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()    

    opt = LidarPlaneOptimizer()
    
    # 1. 加载初值位姿
    init_poses = opt.load_tum_poses(args.pose_file)
    
    # 2. 加载 GPS 数据
    gps_data = pd.read_csv(args.gps_file) if args.gps_file else None
    
    # 3. 加载对齐参数初值 (A)
    align_init = None
    if args.align_file:
        raw = np.loadtxt(args.align_file) # tx ty tz qx qy qz qw scale
        t_ew = gtsam.Point3(raw[0], raw[1], raw[2])
        q_ew = gtsam.Rot3.Quaternion(raw[6], raw[3], raw[4], raw[5]) # w, x, y, z
        align_init = gtsam.Pose3(q_ew, t_ew)

    # 4. 构网
    opt.build_graph(init_poses, args.pcd_folder, gps_data, align_init, args.lever_arm)

    # 5. 优化并保存
    final_result = opt.optimize()
    opt.save_results_tum(final_result, args.output_file) 
    
    # 打印最终外参
    if final_result.exists(A(0)):
        print("\nOptimized T_EW (Global Alignment):\n", final_result.atPose3(A(0)))
    if final_result.exists(L(0)):
        print("Optimized Lever Arm (IMU->GPS):", final_result.atPoint3(L(0)))