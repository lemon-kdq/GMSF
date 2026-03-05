import numpy as np
import argparse
import os

def convert_imu_data(input_file, output_file, gravity=9.80665):
    """
    转换 IMU 数据格式
    输入: timestamp ax ay az gx gy gz (空格分隔)
    输出: timestamp,ax,ay,az,gx,gy,gz,qx,qy,qz,qw (CSV)
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"Processing IMU data: {input_file}")
    
    # 1. 读取数据 (自动处理空格或制表符分隔)
    # 假设输入列顺序为: t, ax, ay, az, gx, gy, gz
    raw_data = np.loadtxt(input_file)
    
    if raw_data.shape[1] < 7:
        print("Error: Input file must have at least 7 columns (t, ax, ay, az, gx, gy, gz).")
        return

    # 2. 提取分量
    timestamps = raw_data[:, 0]
    accel = raw_data[:, 1:4]
    gyro = raw_data[:, 4:7]
    
    # 3. 加速度单位转换 (从 g 转换到 m/s^2)
    # 根据你提供的输出示例，az 从 ~1.0 变成了 ~9.8，说明需要乘以重力常数
    accel_m_s2 = accel * gravity
    
    # 4. 补全四元数 (根据你的要求，输出为 0, 0, 0, 0)
    num_samples = len(timestamps)
    quats = np.zeros((num_samples, 4)) # qx, qy, qz, qw
    
    # 5. 合并数据
    # 顺序: timestamp, ax, ay, az, gx, gy, gz, qx, qy, qz, qw
    result = np.column_stack((timestamps, accel_m_s2, gyro, quats))
    
    # 6. 保存为 CSV
    header = "timestamp,ax,ay,az,gx,gy,gz,qx,qy,qz,qw"
    np.savetxt(output_file, result, delimiter=',', header=header, comments='', fmt='%.6f')
    
    print("-" * 30)
    print(f"Successfully saved to: {output_file}")
    print(f"Total samples: {num_samples}")
    print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description="Convert raw IMU data (g) to CSV (m/s^2) with quaternion placeholders.")
    
    # 命令行参数
    parser.add_argument("-i", "--input", required=True, help="Path to raw IMU txt file (space separated).")
    parser.add_argument("-o", "--output", default="imu_converted.csv", help="Output CSV path.")
    parser.add_argument("-g", "--gravity", type=float, default=9.81, help="Gravity constant (default: 9.80665).")
    
    args = parser.parse_args()

    convert_imu_data(args.input, args.output, args.gravity)

if __name__ == "__main__":
    main()