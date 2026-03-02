#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import rosbag
import sys
import argparse
from sensor_msgs.msg import NavSatFix, Imu, JointState, PointCloud2, CompressedImage
import struct


def extract_bynav_pose(bag, output_path):
    """提取 /gt/bynav_pose 数据"""
    with open(output_path, 'w') as f:
        # 写入表头
        f.write('timestamp,x,y,z,status,status_service\n')

        for topic, msg, t in bag.read_messages(topics=['/gt/bynav_pose']):
            timestamp = msg.header.stamp.to_sec()
            f.write(f'{timestamp},{msg.latitude},{msg.longitude},{msg.altitude},{msg.status.status},{msg.status.service}\n')

    print(f'已提取 bynav_pose 到 {output_path}')


def extract_imu(bag, output_path):
    """提取 /gt/imu0 数据"""
    with open(output_path, 'w') as f:
        # 写入表头
        f.write('timestamp,ax,ay,az,gx,gy,gz,qx,qy,qz,qw\n')

        for topic, msg, t in bag.read_messages(topics=['/gt/imu0']):
            timestamp = msg.header.stamp.to_sec()
            ax = msg.linear_acceleration.x
            ay = msg.linear_acceleration.y
            az = msg.linear_acceleration.z
            gx = msg.angular_velocity.x
            gy = msg.angular_velocity.y
            gz = msg.angular_velocity.z
            qx = msg.orientation.x
            qy = msg.orientation.y
            qz = msg.orientation.z
            qw = msg.orientation.w
            f.write(f'{timestamp},{ax},{ay},{az},{gx},{gy},{gz},{qx},{qy},{qz},{qw}\n')

    print(f'已提取 imu0 到 {output_path}')


def extract_wheel_velocity(bag, output_path):
    """提取 /gt/wheel_velocity 数据"""
    with open(output_path, 'w') as f:
        # 写入表头
        f.write('timestamp')

        # 第一次读取时获取joint名称
        first_msg = True
        joint_names = []

        for topic, msg, t in bag.read_messages(topics=['/gt/wheel_velocity']):
            if first_msg:
                joint_names = msg.name
                for name in joint_names:
                    f.write(f',velocity_{name}')
                f.write('\n')
                first_msg = False

            timestamp = msg.header.stamp.to_sec()
            f.write(f'{timestamp}')
            for vel in msg.velocity:
                f.write(f',{vel}')
            f.write('\n')

    print(f'已提取 wheel_velocity 到 {output_path}')


def extract_lidar(bag, output_path):
    """提取 /gt/lid0 数据 (PointCloud2)"""
    with open(output_path, 'w') as f:
        # 写入表头 - 只保存时间戳和点云基本信息
        f.write('timestamp,width,height,point_step,row_step,point_num\n')

        for topic, msg, t in bag.read_messages(topics=['/gt/lid0']):
            timestamp = msg.header.stamp.to_sec()
            point_num = msg.width * msg.height
            f.write(f'{timestamp},{msg.width},{msg.height},{msg.point_step},{msg.row_step},{point_num}\n')

    print(f'已提取 lid0 (点云摘要) 到 {output_path}')


def extract_compressed_image(bag, topic, output_path):
    """提取压缩图像数据的时间戳"""
    with open(output_path, 'w') as f:
        # 写入表头
        f.write('timestamp,format\n')

        for _, msg, t in bag.read_messages(topics=[topic]):
            timestamp = msg.header.stamp.to_sec()
            f.write(f'{timestamp},{msg.format}\n')

    print(f'已提取 {topic} 到 {output_path}')


def main():
    parser = argparse.ArgumentParser(description='从rosbag中提取/gt开头的传感器数据到txt文件')
    parser.add_argument('bag_path', type=str, help='rosbag文件路径')
    parser.add_argument('output_dir', type=str, help='输出目录路径')

    args = parser.parse_args()

    print(f'正在打开 rosbag: {args.bag_path}')
    bag = rosbag.Bag(args.bag_path)

    # 提取各个topic的数据
    extract_bynav_pose(bag, f'{args.output_dir}/bynav_pose.txt')
    extract_imu(bag, f'{args.output_dir}/imu0.txt')
    extract_wheel_velocity(bag, f'{args.output_dir}/wheel_velocity.txt')
    extract_lidar(bag, f'{args.output_dir}/lid0.txt')
    # 提取图像数据
    extract_compressed_image(bag, '/GT23/CAM_0/compressed_image', f'{args.output_dir}/cam0.txt')
    extract_compressed_image(bag, '/GT23/CAM_1/compressed_image', f'{args.output_dir}/cam1.txt')

    bag.close()
    print(f'所有数据已提取完成，保存在: {args.output_dir}')


if __name__ == '__main__':
    main()
