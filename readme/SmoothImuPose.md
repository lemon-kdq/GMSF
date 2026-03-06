这是一份为您准备好的 Markdown 文档。它总结了轨迹平滑中速度尖峰产生的原因，以及**动态噪声模型（Dynamic Noise Scaling）**的解决方案和代码实现。

---

# 轨迹平滑优化：解决速度突变与毛刺问题的方案

在处理高频轨迹（如 IMU 预积分或视觉里程计）并结合低频优化位姿（关键帧）进行平滑时，最常见的问题是轨迹在关键帧附近出现**速度尖峰**或**高频毛刺**。

## 1. 问题分析：为什么会出现尖峰？

在使用 GTSAM 构建因子图时，我们通常使用 `BetweenFactor` 来约束相邻帧 $X_{i-1}$ 和 $X_i$ 的相对位姿。

*   **原因一：静态噪声模型的局限性**
    如果我们给所有相邻帧设置统一的噪声（如 `1e-3`），当轨迹频率很高时（如 100Hz，$\Delta t = 0.01s$），`1e-3` 的误差容忍度对于这么短的时间间隔来说太大了。
*   **原因二：权重冲突**
    关键帧（Keyframes）通常被赋予极高的权重（`PriorFactor` 噪声极小）。如果相邻帧的约束不够紧，优化器为了强行满足关键帧的绝对位置，会“折弯”中间的小路段，导致局部产生巨大的位姿变化，表现为速度突变。

## 2. 核心解决方案：动态噪声缩放（Dynamic Scaling）

**原理：** 噪声的标准差（$\sigma$）应当与采样时间间隔 $\Delta t$ 成正比。
$$ \sigma_{current} = \sigma_{base} \times \Delta t $$

*   当 $\Delta t$ 极小时，$\sigma$ 随之变小，约束变得极其“硬”，强制保证运动的平滑性。
*   这种方法在物理上模拟了“随机游走”或“匀速假设”的不确定性随时间累积的特性。

---

## 3. 代码实现

以下是集成动态噪声模型的轨迹平滑脚本：

```python
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X
import argparse

def load_tum_poses(file_path):
    """读取并解析 TUM 格式轨迹"""
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            d = list(map(float, line.split()))
            # TUM: ts, tx, ty, tz, qx, qy, qz, qw
            # GTSAM Quaternion: w, x, y, z
            q = gtsam.Rot3.Quaternion(d[7], d[4], d[5], d[6])
            t = gtsam.Point3(d[1], d[2], d[3])
            poses.append((d[0], gtsam.Pose3(q, t)))
    return sorted(poses, key=lambda x: x[0])

def interpolate_pose(t, t1, p1, t2, p2):
    """线性+SLERP插值"""
    ratio = (t - t1) / (t2 - t1)
    rot = p1.rotation().slerp(ratio, p2.rotation())
    pos = p1.translation() + ratio * (p2.translation() - p1.translation())
    return gtsam.Pose3(rot, pos)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-hf", "--high_freq", required=True, help="原始高频轨迹")
    parser.add_argument("-kf", "--keyframes", required=True, help="优化后的关键帧")
    parser.add_argument("-o", "--output", default="smoothed.txt")
    args = parser.parse_args()

    # 1. 加载数据
    hf_data = load_tum_poses(args.high_freq)
    kf_data = load_tum_poses(args.keyframes)
    
    # 2. 合并时间轴 (此处省略详细合并逻辑，确保 all_points 包含所有点)
    # all_points 结构: [(ts, p_init, is_kf, kf_pose), ...]
    # ... (合并逻辑同前) ...

    # 3. 构建因子图
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    
    # --- 噪声参数设置 ---
    # base_sigma 代表“每秒”可能产生的漂移
    base_sigma = np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]) 
    # 关键帧先验：给一个较小但不为0的噪声，允许微量调节
    kf_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([5e-4, 5e-4, 5e-4, 5e-3, 5e-3, 5e-3]))

    for i in range(len(all_points)):
        ts, p_init, is_kf, kf_pose = all_points[i]
        initial_values.insert(X(i), p_init)
        
        # 约束1：关键帧绝对位置约束
        if is_kf:
            graph.add(gtsam.PriorFactorPose3(X(i), kf_pose, kf_prior_noise))
            
        # 约束2：相邻帧相对运动约束 (动态权重核心)
        if i > 0:
            dt = ts - all_points[i-1][0]
            dt = max(dt, 1e-6) # 防止除零
            
            # 【动态噪声逻辑】
            # 时间越短，噪声越小，约束越强
            step_sigma = base_sigma * dt 
            odom_noise = gtsam.noiseModel.Diagonal.Sigmas(step_sigma)
            
            # 观测值为原始轨迹中的相对增量
            rel_pose = all_points[i-1][1].between(p_init)
            graph.add(gtsam.BetweenFactorPose3(X(i-1), X(i), rel_pose, odom_noise))

    # 4. 执行优化
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values)
    result = optimizer.optimize()

    # 5. 保存结果 (TUM 格式)
    # ... (保存逻辑省略) ...

if __name__ == "__main__":
    main()
```

---

## 4. 方案总结与调参建议

| 参数 | 建议值 | 作用 |
| :--- | :--- | :--- |
| `base_sigma` (平移) | `1e-3` ~ `1e-2` | 控制轨迹的平滑程度。值越小，越倾向于保持原始轨迹形状；值越大，越容易被关键帧拉伸。 |
| `base_sigma` (旋转) | `1e-4` ~ `1e-3` | **对速度毛刺影响最大**。如果旋转有尖峰，减小此值。 |
| `kf_prior_noise` | `1e-4` ~ `1e-3` | 关键帧的“锚定”强度。如果不希望轨迹被强行拽向某个可能不准的关键帧，可以适当放大。 |

### 为什么这个方法有效？
通过 `sigma * dt`，我们实际上在因子图中建立了一个**阻尼机制**。当 $\Delta t \to 0$ 时，相邻位姿之间的不确定性趋近于零，这意味着优化器几乎不允许任何瞬时的速度跳变，从而从根本上消除了“毛刺”。

---