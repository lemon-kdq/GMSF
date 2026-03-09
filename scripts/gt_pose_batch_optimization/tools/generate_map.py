import os
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def load_tum_poses(pose_file):
    """
    读取TUM格式pose
    timestamp tx ty tz qx qy qz qw
    """
    poses = {}

    with open(pose_file, "r") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue

            data = line.split()

            ts = data[0]
            tx, ty, tz = map(float, data[1:4])
            qx, qy, qz, qw = map(float, data[4:8])

            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]

            poses[ts] = T

    return poses


def merge_pcds(pcd_dir, poses, voxel_size=0):
    merged = o3d.geometry.PointCloud()

    files = sorted(os.listdir(pcd_dir))

    for file in files:

        if not file.endswith(".pcd"):
            continue

        ts = os.path.splitext(file)[0]

        if ts not in poses:
            print("skip pose not found:", file)
            continue

        path = os.path.join(pcd_dir, file)

        pcd = o3d.io.read_point_cloud(path)

        # voxel filter
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)

        T = poses[ts]

        pcd.transform(T)

        merged += pcd


    return merged


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("pcd_dir", help="pcd folder")
    parser.add_argument("pose_file", help="tum pose file")
    parser.add_argument("output_pcd", help="output merged pcd")

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.25,
        help="voxel size for downsample (0 means no filtering)"
    )

    args = parser.parse_args()

    poses = load_tum_poses(args.pose_file)

    merged = merge_pcds(args.pcd_dir, poses, args.voxel_size)

    o3d.io.write_point_cloud(args.output_pcd, merged)

    print("saved merged map to:", args.output_pcd)


if __name__ == "__main__":
    main()