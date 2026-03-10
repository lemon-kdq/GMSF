import argparse
import rospy
import os 
from pathlib import Path
from gt_pose_batch_optimization.tools.gtsam_pose_optimization import LidarPlaneOptimizer 
from gt_tools.camera.project import project_points_to_image,visualize_projection
from gt_tools.gt.gt_cmd import load_gt_param_according_to_rosbag
from gt_record_config.get_parameter import get_extrinsic_from_vehicle_and_box_config
from gt_tools.pcd.pcd_io import load_pcd_at_pointrgbal
from gt_tools.pcd.pointcloud_process import merge_pointcloud

import cv2 
import open3d as o3d  
import numpy as np
import sys


def get_arround_pointcloud(map,Tcm,range = 200):  
    map_in_cam = map.transform(Tcm)
    points = np.asarray(map_in_cam.points)
    
    # Initialize indices array first
    indices = np.arange(len(points))
    # Calculate distances from camera origin (0,0,0) to each point
    distances = np.linalg.norm(points, axis=1)
    
    # Create mask for points within range and positive z
    mask = (distances <= range) & (points[:, 2] > 0)
    

    # Ensure mask has correct shape
    if mask.shape[0] != points.shape[0]:
        print(f"Warning: Mask shape {mask.shape} doesn't match points shape {points.shape}")
        return np.array([]), np.array([]), np.array([])
    
    # Apply mask to all arrays
    points = points[mask]
    indices = indices[mask]
    return points, indices

if __name__ == "__main__":
    # 设置 argparse
    parser = argparse.ArgumentParser(description="读取 ROS bag 文件并处理消息")
    parser.add_argument("-c","--camera_folder", type=str, help="camera ROS bag 文件的路径")
    parser.add_argument("-k","--keyframe", type=str, help="camera ROS bag 文件的路径")
    parser.add_argument("-l","--lidar_folder", type=str, help="lidar ROS bag 文件的路径")
    parser.add_argument("-b","--rosbag", type=str, help="lidar ROS bag 文件的路径")
    parser.add_argument("-p","--pose_file", type=str, help="pose file")
    parser.add_argument("--size", type=int, default = 4, help="output image folder")
    parser.add_argument("--catkin_ws", type=str, default= "/workspace/catkin_ws", help="Name of the file output freeze period")
    parser.add_argument("--output_folder", type=str, help="output image folder", default=None)

    args = parser.parse_args()
    config_yaml_folder = load_gt_param_according_to_rosbag(args.catkin_ws,args.rosbag)
    cam_id = os.path.basename(args.camera_folder).replace("-","_")

    print(f"cam_id : {cam_id}")
    gt_equipment_config = rospy.get_param('/gt_equipment_config')
    gt_vehicle_config = rospy.get_param('/gt_vehicle_config')

    os.makedirs(args.output_folder,exist_ok=True)
    Tcl,_,_,sensor_dict = get_extrinsic_from_vehicle_and_box_config("LID_0",cam_id,gt_vehicle_config,gt_equipment_config,False)
    K = sensor_dict[cam_id].intrinsic.K 
    D = sensor_dict[cam_id].intrinsic.D
    cam_model = sensor_dict[cam_id].intrinsic.model   
    Tcl = Tcl.Matrix()
    
    keyframe_pose_interface = LidarPlaneOptimizer()
    keyframe_poses = keyframe_pose_interface.load_tum_poses(args.pose_file)
    
    with open(args.keyframe) as f:
        lines = f.readlines()

    keyframe_size = len(lines)
    for i in range(1,keyframe_size):  # 如果第一行是表头
        _, camera_ts_str = lines[i].strip().split(',')
        cam_path = Path(args.camera_folder) / f"{camera_ts_str}.jpeg"
        base_pose_index = keyframe_pose_interface.find_pose_index(float(camera_ts_str))
    
        if not cam_path.exists():
            print(f"image不存在: {cam_path}")
            continue
        if base_pose_index is None:
            print(f"can't find camera pose: {camera_ts_str}")
            continue 
        Twl0 = keyframe_poses[base_pose_index]
        max_j = min(i+args.size,keyframe_size)
        merged_pc = []
        for j in range(i,max_j):
            _, lidar_ts_str = lines[i].strip().split(',')
            lid_path = Path(args.lidar_folder) / f"{lidar_ts_str}.pcd"
            
            if not lid_path.exists():
                print(f"PCD不存在: {lid_path}")
                continue
            lid_timestamp = float(lidar_ts_str)
            pc_raw = load_pcd_at_pointrgbal(lid_path)
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(pc_raw[:,:3])
            pose_index = keyframe_pose_interface.find_pose_index(lid_timestamp)
            if pose_index is not None: 
                Twl = keyframe_poses[pose_index]
                Tl0l = Twl0.inverse() * Twl 
                Tl0l_M = keyframe_pose_interface.pose3_to_4x4_manual(Tl0l) 
                pc_transformed = pc_o3d.transform(Tl0l_M)    
                merged_pc.append(pc_transformed)
        if len(merged_pc) != 0: 
            try:
                image = cv2.imread(str(cam_path))
                if image is None:
                    raise ValueError("cv2.imread returned None")
            except cv2.error as e:
                print(f"OpenCV error: {e}")

            except Exception as e:
                print(f"Other error: {e}")               
            merged_pcd = merge_pointcloud(merged_pc) 
            points3d, _ = get_arround_pointcloud(merged_pcd,Tcl)
            points2d,_ = project_points_to_image(points3d,K,D,np.eye(4),cam_model)
            prj_image = visualize_projection(image,points2d,points3d[:,2])
            s_image = os.path.join(args.output_folder,f"{camera_ts_str}.jpeg")
            cv2.imwrite(s_image,prj_image)
    rospy.signal_shutdown("Task completed")
    sys.exit(0)