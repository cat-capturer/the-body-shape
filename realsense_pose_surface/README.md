# RealSense 人体关键点球面显示

在 Linux 上采集 RealSense RGB-D 图像，用 MediaPipe Pose 检测单个人体的 33 个关键点，再由对齐深度反投影为相机坐标（米）。Open3D 使用球体和圆柱骨段显示骨架。

## 安装

- 目标环境：Ubuntu、Python **3.10–3.12**、USB 3.0、带彩色和深度流的 RealSense 相机。
- 原始目标相机是 D455；其他设备需要在 `realsense-viewer` 中确认支持的分辨率和帧率，尚未逐型号实测。
- 当前代码使用旧的 `mp.solutions.pose`，因此固定 MediaPipe 0.10.21。新版已移除该 API，直接升级会无法启动（[官方说明](https://github.com/google-ai-edge/mediapipe/issues/6192)）。
- 只安装 `opencv-contrib-python`，避免它和 `opencv-python` 同时覆盖 `cv2`。建议使用新的虚拟环境。

```bash
cd realsense_pose_surface
bash install_ubuntu.sh
source .venv/bin/activate
python run_realsense_pose_surface.py
```

可用 `PYTHON=python3.11 bash install_ubuntu.sh` 选择解释器。RealSense 驱动和 udev rules 请按 [SDK 官方安装说明](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md) 配置。

## 按设备和场景调整

```bash
# 彩色、深度使用不同分辨率；替换成设备支持的组合
python run_realsense_pose_surface.py --width 640 --height 480 --depth-width 848 --depth-height 480 --fps 30

# 有多台相机时指定序列号
python run_realsense_pose_surface.py --serial YOUR_CAMERA_SERIAL

# 按拍摄距离、噪声和遮挡情况调整
python run_realsense_pose_surface.py --min-depth 0.3 --max-depth 3.5 --depth-window 5 --depth-tolerance 0.15
python run_realsense_pose_surface.py --model-complexity 2 --detection-confidence 0.6 --tracking-confidence 0.6

# 只显示 RGB 预览 / 关闭全部窗口（Ctrl+C 退出）
python run_realsense_pose_surface.py --no-render
python run_realsense_pose_surface.py --no-render --no-preview
```

`--help` 只需要 NumPy，不会导入相机、MediaPipe 或 Open3D。`--no-render` 不会导入 Open3D。无窗口运行仍需相机 SDK、MediaPipe 和 OpenCV，也不会自动保存结果。

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--width`, `--height`, `--fps` | 848, 480, 30 | 彩色流配置；深度流默认沿用 |
| `--depth-width`, `--depth-height` | 随彩色流 | 单独配置深度分辨率，对齐后按实际图像尺寸计算 |
| `--min-visibility` | 0.55 | 忽略低可见度关键点 |
| `--detection-confidence`, `--tracking-confidence` | 0.5, 0.5 | Pose 检测和跟踪阈值 |
| `--depth-window` | 5 | 正奇数窗口；1 表示只读中心像素 |
| `--min-depth`, `--max-depth` | 0.2, 4.5 米 | 在求中位数前剔除超范围深度 |
| `--depth-tolerance` | 关闭 | 仅保留与有效中心像素深度差不超过阈值的邻域像素 |
| `--smooth` | 0.68 | 30 FPS 下上一帧的权重；越大越平滑，但延迟也越大 |
| `--reset-distance` | 0.75 米 | 位移超过阈值时直接采用新位置，避免跨大跳变拖尾 |
| `--reset-after` | 0.5 秒 | 帧间隔过大时清空平滑历史 |
| `--sphere-radius`, `--capsule-radius` | 0.045, 0.022 米 | 显示用球体和骨段半径 |
| `--mirror` | 关闭 | 只镜像预览图，不改变三维坐标和关键点编号 |

## 实现与适用范围

- 深度对齐到彩色流，使用对齐深度的内参和设备报告的深度单位反投影；相机坐标 X 向右、Y 向下、Z 向前。没有到机器人坐标系的外参标定。
- `pose_geometry.py` 仅依赖 NumPy，包含深度采样、像素映射、按时间平滑和骨段变换，便于后续接入其他 RGB-D 来源。实际采集适配器目前仍是 RealSense。
- 无效深度、NaN、越界点和低可见度点不参与三维显示；丢失点立即隐藏，重新出现时重新初始化。
- EMA 使用相机时间戳，将 30 FPS 的权重换算到实际帧间隔，减小不同处理帧率造成的平滑差异。MediaPipe 自带的二维平滑仍保留。
- Open3D 缓存每个球体和骨段的网格及原始顶点；连续可见时只更新顶点，避免每帧创建网格、删除和重新注册全部对象。首次检测到人体时调整视野。
- `--depth-tolerance` 可减少身体边缘混入远处背景的问题，但中心深度也可能错误。中心无效时退回邻域中位数，不能保证区分前景和背景；阈值过小也可能丢弃斜面上的有效像素。
- 深度测量来自可见表面，不是真实关节中心；球面和骨段不表示真实体型或人体表面重建。
- 仍为单人 Pose 模型，没有训练新模型或增加身份跟踪。人物切换、小幅错检、严重遮挡仍可能造成错误连接。此次提升的是设备/参数适配和处理稳定性，尚无跨人群、光照、姿态的准确率提升数据。

## 验证

在仓库根目录运行无需相机的测试：

```bash
python -m pip install 'numpy>=1.24,<2'
python -m unittest discover -s tests -v
```

测试覆盖深度窗口、无效值、距离过滤、画面边缘、单位换算、帧率变化、丢失/重现、时间间隔和骨段几何等。GitHub Actions 在 Python 3.10/3.12 上运行这些测试。它们不能替代相机采集、MediaPipe 推理和 Open3D 窗口的实机验证。

实机建议：在相同场景对比修改前后，记录不同距离、衣着、光照和遮挡下的有效关键点比例、静止抖动、动作延迟及实际 FPS，再确定场景参数。

## 排错

- 无设备：检查 USB 3.0、udev 权限和 `realsense-viewer`。
- 无法启动流：检查序列号、彩色/深度分辨率与帧率组合。
- `mp.solutions` 不存在：在新虚拟环境中安装本项目的 `requirements.txt`。
- Open3D 无窗口：检查桌面环境，或加 `--no-render`；完全无显示环境还需加 `--no-preview`。
- 抖动明显：先检查深度质量和背景污染，再调整窗口、可见度阈值和平滑。增大平滑会增加动作延迟。
