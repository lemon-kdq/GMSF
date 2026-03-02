# Data Definition

## Rosbag File Information

| Property | Value |
|----------|-------|
| Path | `/mnt/d/gt_pipeline_data/record_20260116/20260116_134000/mins_all_sensors_with_calibrated_wheel.bag` |
| Version | 2.0 |
| Duration | 4:59s (299s) |
| Start Time | Jan 16 2026 13:40:00.10 (1768542000.10) |
| End Time | Jan 16 2026 13:44:59.99 (1768542299.99) |
| Size | 6.8 GB |
| Total Messages | 101,032 |
| Compression | None |

## Message Types

| Type | Message ID |
|------|------------|
| `sensor_msgs/CompressedImage` | 8f7a12909da2c9d3332d540a0977563f |
| `sensor_msgs/Imu` | 6a62c6daae103f4ff57a132d6f95cec2 |
| `sensor_msgs/JointState` | 3066dcd76a6cfaef579bd0f34173e9fd |
| `sensor_msgs/NavSatFix` | 2d3a8cd499b9b4a0249fb98fd05cfa48 |
| `sensor_msgs/PointCloud2` | 1158d486dd51d683ce2f1be655c3c181 |

## Topics

| Topic Name | Messages | Type | Description |
|------------|----------|------|-------------|
| `/GT23/CAM_0/compressed_image` | 2,999 | `sensor_msgs/CompressedImage` | Camera 0 compressed image data |
| `/GT23/CAM_1/compressed_image` | 2,999 | `sensor_msgs/CompressedImage` | Camera 1 compressed image data |
| `/gt/bynav_pose` | 29,967 | `sensor_msgs/NavSatFix` | GNSS/Bynav pose position data |
| `/gt/imu0` | 47,727 | `sensor_msgs/Imu` | IMU sensor data (accelerometer & gyroscope) |
| `/gt/lid0` | 2,344 | `sensor_msgs/PointCloud2` | LiDAR point cloud data |
| `/gt/wheel_velocity` | 14,996 | `sensor_msgs/JointState` | Wheel velocity/joint state data |

## Sensor Summary

This dataset contains multi-sensor fusion data from a robotic platform:

1. **Visual Sensors**: 2 cameras (CAM_0, CAM_1) providing compressed image streams
2. **Positioning**: GNSS/Bynav pose for global positioning reference
3. **IMU**: High-frequency inertial measurements (~159 Hz based on message count)
4. **LiDAR**: 3D point cloud data (~7.8 Hz)
5. **Odometry**: Wheel velocity/encoder data (~50 Hz)
