import vqf
import numpy as np
import pandas as pd
import argparse
import os

def estimate_attitude(input_file, output_file):
    # 读取IMU数据
    # 期望列名: timestamp, ax, ay, az, gx, gy, gz
    print(f"Reading IMU data from {input_file}...")
    data = pd.read_csv(input_file)
    
    # 提取加速度、角速度和时间戳
    # 根据文件内容：timestamp,ax,ay,az,gx,gy,gz,qx,qy,qz,qw
    # VQF 需要 C-contiguous 的内存布局
    acc = np.ascontiguousarray(data[['ax', 'ay', 'az']].values)
    gyr = np.ascontiguousarray(data[['gx', 'gy', 'gz']].values)
    timestamps = data['timestamp'].values
    
    # 计算采样周期 (假设采样率恒定，取平均值或逐帧计算)
    # VQF offline 模式支持统一的采样率
    dt = np.diff(timestamps)
    mean_dt = np.mean(dt)
    print(f"Mean sampling period: {mean_dt:.6f} s (approx {1/mean_dt:.2f} Hz)")
    
    # 初始化 VQF
    # 默认参数对于大多数应用效果良好
    filter = vqf.VQF(mean_dt)
    
    # 批量处理 (Offline estimation)
    print("Estimating attitude using VQF...")
    out = filter.updateBatch(gyr, acc)
    
    # 获取四元数 (VQF 输出格式通常是 qw, qx, qy, qz)
    quat = out['quat6D'] # 使用 6D 融合结果 (不依赖磁力计)
    
    # TUM 格式: timestamp x y z qx qy qz qw
    # 位置默认为 0
    print(f"Saving attitude to {output_file} in TUM format...")
    
    # 构造 TUM 格式数据
    # 注意：vqf 的 quat 是 [w, x, y, z]，TUM 格式是 [x, y, z, w]
    tum_data = np.zeros((len(timestamps), 8))
    tum_data[:, 0] = timestamps
    # tum_data[:, 1:4] = 0 (位置 x, y, z)
    tum_data[:, 4] = quat[:, 1] # qx
    tum_data[:, 5] = quat[:, 2] # qy
    tum_data[:, 6] = quat[:, 3] # qz
    tum_data[:, 7] = quat[:, 0] # qw
    
    # 保存文件
    np.savetxt(output_file, tum_data, fmt='%.6f', delimiter=' ')
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Offline IMU attitude estimation using VQF.')
    parser.add_argument('input', help='Path to input IMU data file (csv format)')
    parser.add_argument('output', help='Path to output attitude file (TUM format)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
    else:
        estimate_attitude(args.input, args.output)
