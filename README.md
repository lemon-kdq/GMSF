# 离线多传感器 Pose 全局 Batch 优化方案（含公式定义）

---

## 0. 符号与约定

### 坐标系
- $\mathcal{W}$：世界坐标系（优化目标）
- $\mathcal{B}$：车辆坐标系（body）
- $\mathcal{I}$：IMU 坐标系
- $\mathcal{G}$：GPS 坐标系
- $\mathcal{C}$：相机坐标系
- $\mathcal{L}$：LiDAR 坐标系

### 位姿表示
- $X_i = (R_i, p_i) \in SE(3)$  
  表示第 $i$ 个关键帧车辆在世界坐标系下的位姿

---

## 1. IMU 姿态离线估计（VQF）

### 输出
- 连续时间姿态：
$$
R_{WI}(t) \in SO(3)
$$

### 使用原则
- roll / pitch：可信
- yaw：可能漂移，仅作弱参考或不用

---

## 2. 连续时间运动模型（用于 LiDAR 正畸）

### 姿态插值
$$
R(t) = \text{Interp}\left(R_{WI}(t)\right)
$$

### 平移积分（wheel speed）
$$
p(t) = \int_{t_0}^{t} s_v \cdot v(\tau) \cdot \hat{x}_B(\tau)\, d\tau
$$

其中：
- $v(\tau)$：车速测量
- $s_v$：车速尺度因子（未知，后续优化）
- $\hat{x}_B$：车辆前向单位向量

⚠️ **该轨迹仅用于正畸，不作为 batch 优化约束**

---

## 3. LiDAR 点云正畸（Motion Deskew）

对 LiDAR 点 $P_k$（时间戳 $t_k$）：

$$
P_k^{\text{deskew}} =
T^{-1}(t_{\text{ref}}) \cdot T(t_k) \cdot P_k
$$

其中：
- $T(t) = (R(t), p(t))$ 为连续时间位姿
- $t_{\text{ref}}$ 为帧参考时间（起始或中心）

---

## 4. 关键帧选择

根据连续时间轨迹：

- 位移：
$$
\| p(t_j) - p(t_i) \| > d_{\text{th}}
$$

- 旋转：
$$
\angle(R(t_j) R(t_i)^{-1}) > \theta_{\text{th}}
$$

---

## 5. 优化变量（State）

### 每个关键帧
$$
X_i = (R_i, p_i)
$$

### 全局变量
- GPS 杆臂：
$$
l_{gps} \in \mathbb{R}^3
$$
- GPS → world 对齐：
$$
t_{wg} \in \mathbb{R}^3,\quad \psi_{wg} \in \mathbb{R}
$$
- IMU → 车辆外参：
$$
T_{IB} \in SE(3)
$$
- 车速尺度：
$$
s_v \in \mathbb{R}
$$

---

## 6. 约束因子（Factors）

---

### 6.1 相机重投影误差

对 3D 点 $P$：

$$
r_{\text{cam}} =
\pi\left(R_i^C (R_i P + p_i) + t_i^C\right) - u_{ij}
$$

约束变量：
- $X_i$

---

### 6.2 LiDAR 点–面约束

对 LiDAR 点 $P$ 和平面 $(n, d)$：

$$
r_{\text{lidar}} = n^\top (R_i P + p_i) + d
$$

约束变量：
- $X_i$

---

### 6.3 重力方向约束（关键）

IMU 测得重力方向：
$$
\hat{g}_i^{I}
$$

世界重力：
$$
g_W = \begin{bmatrix} 0 & 0 & -1 \end{bmatrix}^\top
$$

残差：
$$
r_g =
\hat{g}_i^{I}
-
R_{IB}^\top R_i^\top g_W
$$

说明：
- 仅约束 roll / pitch
- 不约束 yaw
- 防止姿态水平面发散

---

### 6.4 尺度约束（累计里程约束）

对关键帧 $i \rightarrow j$：

累计轮速里程：
$$
D_{ij} = \sum_{k=i}^{j-1} v_k \Delta t_k
$$

残差：
$$
r_s =
\| p_j - p_i \|
-
s_v \cdot D_{ij}
$$

特点：
- 只约束**尺度**
- 不约束方向
- 防止隧道 / 退化场景尺度漂移

---

### 6.5 GPS 位置约束

GPS 测量：
$$
p_i^{gps}
$$

预测 GPS 位置：
$$
\hat{p}_i^{gps} =
p_i
+
R_i \cdot l_{gps}
+
t_{wg}
$$

残差：
$$
r_{gps} =
\hat{p}_i^{gps}
-
p_i^{gps}
$$

约束变量：
- $X_i$
- $l_{gps}$
- $t_{wg}$
- $\psi_{wg}$

---

### 6.6 GPS 航向角约束（可选）

$$
r_{\psi} = \psi_i - (\psi_{gps,i} + \psi_{wg})
$$

- 仅在高速度、RTK fix 时启用
- 权重很小

---

## 7. 初始化策略

### 位姿
- roll / pitch：VQF
- yaw：0 或 VQF
- 平移：粗积分或 GPS

### 外参与尺度
$$
l_{gps} = 0,\quad s_v = 1.0
$$

原则：
- **噪声给大**
- **不相信初值**

---

## 8. Batch 优化问题

最终优化问题：

$$
\min_{\Theta}
\sum
\left(
\|r_{\text{cam}}\|^2 +
\|r_{\text{lidar}}\|^2 +
\|r_g\|^2 +
\|r_s\|^2 +
\|r_{gps}\|^2
\right)
$$

其中：
$$
\Theta =
\{ X_i, l_{gps}, t_{wg}, \psi_{wg}, T_{IB}, s_v \}
$$

---

## 9. 核心思想总结

- **点云正畸：连续时间运动模型**
- **姿态稳定：重力方向约束**
- **尺度稳定：累计里程约束**
- **绝对定位：GPS 软约束**
- **所有偏差：作为优化变量**