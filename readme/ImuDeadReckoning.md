这是一个非常经典且实际的问题。在实际安装中，IMU 很难做到与车体轴向完全对齐。

既然你已知 **IMU 的安装位置（杆臂）**、**IMU 与车体的安装偏角（外参旋转）**、**轮速**以及 **IMU 的姿态数据**，我们需要重新构建坐标变换链。

---

### 1. 坐标系重新定义

*   **$W$ (World):** 世界坐标系（如 ENU）。
*   **$V$ (Vehicle):** 车体坐标系（后轴中心，$X$向前，$Y$向左，$Z$向上）。**轮速是在这个系下测量的。**
*   **$I$ (IMU):** IMU 传感器坐标系。**你的姿态角和角速度是基于这个系的。**

**已知参数：**
1.  $\mathbf{R}_I^W$: IMU 当前在世界系下的姿态（由你提到的 IMU 姿态数据转化而来）。
2.  $\mathbf{R}_I^V$: **安装偏角外参**（IMU 相对于车体的旋转）。
3.  $\mathbf{p}_I^V$: **安装位置外参**（IMU 在车体坐标系下的坐标 $[l_x, l_y, l_z]^T$）。
4.  $v_{wheel}$: 轮速计数值（车体 $V$ 系 $X$ 轴速度）。
5.  $\boldsymbol{\omega}_I^I$: IMU 陀螺仪测得的角速度。

---

### 2. 第一步：求解车体在世界系下的姿态 $\mathbf{R}_V^W$

由于你只有 IMU 的姿态 $\mathbf{R}_I^W$，需要通过安装偏角 $\mathbf{R}_I^V$ 转换出车体（车辆本身）的姿态：

$$\mathbf{R}_V^W = \mathbf{R}_I^W \cdot (\mathbf{R}_I^V)^T$$

*注：$\mathbf{R}_I^V$ 通常由离线标定得到。如果车辆水平静止时，IMU 显示 Pitch 2°，则说明 $\mathbf{R}_I^V$ 包含了这 2° 的偏差。*

---

### 3. 第二步：计算 IMU 在车体系下的速度 $\mathbf{v}_I^V$

轮速计测量的是车体后轴中心 $V$ 的速度 $\mathbf{v}_V^V = [v_{wheel}, 0, 0]^T$。
由于 IMU 安装在 $\mathbf{p}_I^V$ 处，当车辆转弯时，IMU 的速度不等于轮速，必须补偿**杆臂效应**。

首先，将 IMU 测得的角速度 $\boldsymbol{\omega}_I^I$ 转到车体坐标系下：
$$\boldsymbol{\omega}_V^V = (\mathbf{R}_I^V)^T \cdot \boldsymbol{\omega}_I^I$$

然后计算 IMU 在车体系下的速度：
$$\mathbf{v}_I^V = \begin{bmatrix} v_{wheel} \\ 0 \\ 0 \end{bmatrix} + \boldsymbol{\omega}_V^V \times \mathbf{p}_I^V$$

**展开公式（叉乘部分）：**
$$\mathbf{v}_I^V = \begin{bmatrix} v_{wheel} + (\omega_{Vy} l_z - \omega_{Vz} l_y) \\ \omega_{Vz} l_x - \omega_{Vx} l_z \\ \omega_{Vx} l_y - \omega_{Vy} l_x \end{bmatrix}$$

---

### 4. 第三步：将 IMU 速度投影到世界系并递推

现在你有了 IMU 在车体系下的精确速度 $\mathbf{v}_I^V$，将其旋转到世界系：

$$\mathbf{v}_I^W = \mathbf{R}_V^W \cdot \mathbf{v}_I^V$$

**位置递推公式：**
$$\mathbf{p}_I^W(k) = \mathbf{p}_I^W(k-1) + \mathbf{v}_I^W \cdot \Delta t$$

---

### 5. 综合完整递推公式

将上述步骤合并，直接得到从 $k-1$ 时刻递推到 $k$ 时刻 IMU 位置的公式：

$$\mathbf{p}_I^W(k) = \mathbf{p}_I^W(k-1) + \underbrace{\left( \mathbf{R}_I^W(k) (\mathbf{R}_I^V)^T \right)}_{\text{车体世界姿态}} \cdot \underbrace{\left( \begin{bmatrix} v_{wheel} \\ 0 \\ 0 \end{bmatrix} + \left( (\mathbf{R}_I^V)^T \boldsymbol{\omega}_I^I \right) \times \mathbf{p}_I^V \right)}_{\text{IMU在车体内部的速度}} \cdot \Delta t$$

---

### 6. 关键细节处理

#### (1) 关于旋转矩阵 $\mathbf{R}_I^V$ 的构建
如果你已知的是 IMU 相对于车体的安装欧拉角（例如：Roll偏移 $\Delta\phi$, Pitch偏移 $\Delta\theta$, Yaw偏移 $\Delta\psi$），通常构造方式为：
$$\mathbf{R}_I^V = \text{Rotz}(\Delta\psi) \text{Roty}(\Delta\theta) \text{Rotx}(\Delta\phi)$$
*注意：这里的偏角定义必须与标定时的定义严格一致。*

#### (2) 为什么不直接用 $\mathbf{R}_I^W$ 乘 $v_{wheel}$？
如果你直接用 `IMU姿态 * 轮速`，会产生两个错误：
1.  **方向误差：** IMU 没装正，它的 X 轴指向和车辆前进方向（轮速方向）不一致，会导致推算轨迹斜着走。
2.  **转弯误差（杆臂）：** 车辆转弯时，车头、车尾、车轴中心的速度矢量是不一样的。不加 $\omega \times p$ 项，轨迹在转弯时会产生剧烈漂移。

#### (3) 提高精度的技巧
*   **中值积分：** 在计算 $\mathbf{v}_I^W$ 时，建议使用 $k$ 时刻和 $k-1$ 时刻姿态/速度的平均值，减少由于离散采样导致的积分误差。
*   **角速度去偏：** 陀螺仪 $\boldsymbol{\omega}_I^I$ 在计算前必须扣除零偏（Bias），否则 $\omega \times p$ 补偿项会引入恒定的速度偏差。
*   **轮速有效性：** 检查轮速计是否已经过车辆自身的转弯半径补偿（有些车提供的轮速是四个轮子的平均，有些是后轴中心，通常默认为后轴中心）。

### 总结流程：
1.  **输入：** IMU姿态 $\mathbf{R}_I^W$、陀螺仪 $\boldsymbol{\omega}_I^I$、轮速 $v_{wheel}$。
2.  **转换：** 将 IMU 角速度转到车体系，计算受杆臂影响后的 IMU 相对速度。
3.  **对齐：** 将车体速度通过“IMU姿态+安装偏角”转回世界系。
4.  **积分：** 累加位移得到位置。