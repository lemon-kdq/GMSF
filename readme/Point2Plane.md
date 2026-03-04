发散（Divergence）在非线性优化中通常由三个原因引起：**坐标系定义冲突**、**雅可比矩阵（Jacobian）正负号反了**、以及**约束权重不平衡**。

以下我先为你梳理详细的算法思路，随后给出修正后的完整代码。

---

# 算法思路详细说明 (Algorithm Analysis)

## 1. 坐标系与变量定义
我们要确保所有几何量在同一个“逻辑链条”上：
*   $X_{i-1}$：前一帧位姿变量，代表从 **Body_{i-1}** 到 **World** 的变换。
*   $X_i$：当前帧位姿变量，代表从 **Body_i** 到 **World** 的变换。
*   $\mathbf{p}_i$：在 **Body_i** 坐标系下观察到的点云坐标。
*   $\mathbf{n}_{i-1}, d_{i-1}$：在 **Body_{i-1}** 坐标系下拟合出的平面法向量和截距（满足 $\mathbf{n}^\top \mathbf{p} = d$）。

## 2. 残差函数（Error Function）的几何推导
我们要约束 $X_i$ 观察到的点，在投影回 $X_{i-1}$ 的局部坐标系后，依然落在 $X_{i-1}$ 观察到的平面上。

### 第一步：计算相对变换 (Relative Pose)
$$T_{rel} = X_{i-1}^{-1} \cdot X_i$$
$T_{rel}$ 的物理意义是将点从第 $i$ 帧坐标系转换到第 $i-1$ 帧坐标系。

### 第二步：点坐标变换
$$\mathbf{p}' = T_{rel} \cdot \mathbf{p}_i = \text{rotate}(R_{rel}, \mathbf{p}_i) + \mathbf{t}_{rel}$$

### 第三步：计算标量残差
$$error = \mathbf{n}_{i-1}^\top \cdot \mathbf{p}' - d_{i-1}$$

## 3. 雅可比矩阵的链式法则 (Chain Rule)
发散最常见的原因是 $\frac{\partial e}{\partial X}$ 的符号与优化器的预期相反。

我们定义 $e$ 对两个位姿的偏导：
1.  $\frac{\partial e}{\partial X_{i-1}}$：当 $X_{i-1}$ 变化时，残差如何变化。
2.  $\frac{\partial e}{\partial X_i}$：当 $X_i$ 变化时，残差如何变化。

GTSAM 的 `pose1.between(pose2, H1, H2)` 提供的 $H1$ 和 $H2$ 分别是：
*   $H1$: $\frac{\partial T_{rel}}{\partial X_{i-1}}$ (6x6)
*   $H2$: $\frac{\partial T_{rel}}{\partial X_i}$ (6x6)

通过 `T_rel.transformFrom(p_i, Dp_Trel, Dp_pi)` 获得：
*   $Dp\_Trel$: $\frac{\partial \mathbf{p}'}{\partial T_{rel}}$ (3x6)

**最终合成：**
$$\text{Jacobian}_{i} = \underbrace{\mathbf{n}_{i-1}^\top}_{1 \times 3} \cdot \underbrace{Dp\_Trel}_{3 \times 6} \cdot \underbrace{H2}_{6 \times 6}$$
$$\text{Jacobian}_{i-1} = \underbrace{\mathbf{n}_{i-1}^\top}_{1 \times 3} \cdot \underbrace{Dp\_Trel}_{3 \times 6} \cdot \underbrace{H1}_{6 \times 6}$$

---

# 常见发散原因排查

1.  **法向量方向**：如果 $d$ 是正的，但 $n$ 的方向反了，残差会越变越大。
2.  **噪声模型（Sigma）过小**：如果你添加了 500 个点面约束，每个噪声设为 0.01，而 `BetweenFactor` 噪声设为 0.1。点面约束的“力量”是里程计的 100 倍，一旦匹配中有一个错点（Outlier），它会像杠杆一样把整个轨迹撬飞。
3.  **TUM 初值不准**：如果初值误差太大，Open3D 找出来的匹配对本身就是错的。

---

# 修正后的代码实现

重点修正了雅可比计算逻辑、Fortran 内存对齐以及噪声平衡。

```python
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X

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
```

### 调试建议：

1.  **先跑里程计优化**：先不调用 `add_point_to_plane_constraint`，只跑 `BetweenFactor` 和 `AttitudeFactor`。如果结果不发散，说明基础框架 OK。
2.  **逐帧添加约束**：如果里程计没问题，先只给其中一对相邻帧添加**一个**点面约束。看优化器是否报错。
3.  **检查法向量一致性**：在 `find_point_to_plane_matches` 中，确保你的 `d = n.dot(centroid)`。
    *   验证方法：在 `error_func` 里打印第一帧的 `error_val`。因为初值是里程计给的，如果初值不错，`error_val` 应该是一个很小的值（如 0.01）。**如果初值状态下 `error_val` 就高达几百，那肯定是坐标系搞错了。**
4.  **限制约束数量**：不要每一帧加 1000 个点，尝试每帧只加 50 个高质量、分布均匀的平面点。

### 平面参数校验：
在你的 Open3D 匹配函数里，请确保：
```python
# 提取法向量后，强制归一化
normal = normal / np.linalg.norm(normal)
# 截距计算必须严格对应
d = np.dot(normal, centroid)
```
如果 `d` 计算时用的坐标系和 `normal` 不一致，优化必发散。