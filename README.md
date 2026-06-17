# the-body-shape

人体关键点球面显示。

## RealSense D455 实时版本

代码在 [`realsense_pose_surface/`](./realsense_pose_surface)。

功能：

- Linux 下读取 Intel RealSense D455 RGB-D 数据
- MediaPipe Pose 实时识别人体 33 个关键点
- 使用 D455 深度图将 2D 关键点转换为 3D 相机坐标
- Open3D 实时显示关键点球面和骨骼胶囊曲面

快速开始：

```bash
cd realsense_pose_surface
bash install_ubuntu.sh
source .venv/bin/activate
python run_realsense_pose_surface.py
```
