# RealSense D455 人体关键点球面显示

这个子项目在 Linux 上使用 Intel RealSense D455 实时采集 RGB-D 画面，利用 MediaPipe Pose 检测人体 33 个关键点，再用深度图把 2D 关键点反投影为 3D 点，并用 Open3D 把关键点显示成球面、骨段显示成胶囊状曲面。

## 硬件和系统

- Intel RealSense D455
- Ubuntu 20.04/22.04/24.04 或其他支持 `librealsense2` 的 Linux
- Python 3.10 或更新版本
- USB 3.0 接口

## 安装

```bash
cd realsense_pose_surface
bash install_ubuntu.sh
source .venv/bin/activate
```

如果 `pyrealsense2` 安装失败，请先按 Intel 官方文档安装 `librealsense2` 和 udev rules，然后重新执行：

```bash
pip install -r requirements.txt
```

## 运行

```bash
python run_realsense_pose_surface.py
```

常用参数：

```bash
python run_realsense_pose_surface.py --width 848 --height 480 --fps 30
python run_realsense_pose_surface.py --sphere-radius 0.06 --capsule-radius 0.03
python run_realsense_pose_surface.py --model-complexity 2
python run_realsense_pose_surface.py --no-render
```

`--no-render` 会关闭 Open3D 3D 窗口，只保留 RGB 关键点预览，适合先检查 D455 和 MediaPipe 是否能正常工作。

## 实现说明

- RealSense depth stream 会对齐到 color stream。
- 每个 MediaPipe 关键点用附近像素窗口的有效深度中位数去噪。
- `rs2_deproject_pixel_to_point` 把像素坐标和深度值转换成相机坐标系下的 3D 点。
- Open3D 中每个关键点是一个球面，相邻关键点之间是圆柱胶囊段，形成贴合人体骨架的球状曲面网络。
- 默认用指数滑动平均平滑 3D 点，减少深度噪声导致的抖动。

## 排错

- `No RealSense device was detected`：检查 USB 线、USB 3.0 端口、`realsense-viewer` 是否能看到设备。
- 权限错误：确认已经安装 RealSense udev rules，然后重新插拔 D455。
- Open3D 窗口无法打开：确认当前 Linux 环境有桌面显示；远程服务器可先用 `--no-render` 验证采集链路。
- 姿态点跳动：增大 `--smooth`，或增大 `--depth-window`。
