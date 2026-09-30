# the-body-shape

基于 **Intel RealSense RGB-D、MediaPipe Pose 和 Open3D** 的实时人体关键点三维可视化工具。

程序检测单个人体的 33 个关键点，结合对齐深度将其转换为相机坐标，用球体表示关键点、圆柱表示骨段。完整代码和使用文档均在 `main` 分支。

[![Geometry tests](https://github.com/cat-capturer/the-body-shape/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/cat-capturer/the-body-shape/actions/workflows/tests.yml)

## 功能

- **RGB-D 采集与三维定位**：深度对齐到彩色图，使用实际帧尺寸、相机内参和深度单位反投影。
- **设备与场景配置**：支持指定相机序列号、独立设置彩色和深度分辨率，以及调整检测和跟踪置信度。
- **深度过滤与平滑**：过滤无效值、越界点和超范围深度；支持邻域中位数及可选的中心深度容差，按相机时间戳平滑关键点。
- **实时显示**：RGB 骨架预览、三维球体与骨段；复用显示网格，丢失关键点时隐藏对应几何。
- **可独立复用的核心**：深度采样、像素映射、平滑和骨段变换仅依赖 NumPy，配有无需相机的回归测试。

## 环境要求

- Linux / Ubuntu，Python **3.10–3.12**。
- 带彩色和深度流的 RealSense 相机，原始目标设备为 **D455**，建议使用 USB 3.0。
- 已配置 RealSense SDK / udev 权限；先用 `realsense-viewer` 确认相机及流配置可用。
- 三维窗口需要桌面显示环境。

依赖由 [requirements.txt](realsense_pose_surface/requirements.txt) 管理。当前实现使用旧的 MediaPipe Pose API，固定为 **MediaPipe 0.10.21**，请在独立虚拟环境中安装。

## 快速开始

```bash
git clone https://github.com/cat-capturer/the-body-shape.git
cd the-body-shape/realsense_pose_surface

bash install_ubuntu.sh
source .venv/bin/activate

python run_realsense_pose_surface.py
```

需要选择解释器时，可将安装命令替换为 `PYTHON=python3.11 bash install_ubuntu.sh`。

在 RGB 预览窗口按 **q**、关闭三维窗口或在终端按 **Ctrl+C** 退出。

## 常用运行方式

以下命令在 `realsense_pose_surface/` 目录执行。

```bash
# 查看全部参数
python run_realsense_pose_surface.py --help

# 只显示 RGB 预览
python run_realsense_pose_surface.py --no-render

# 无窗口运行；使用 Ctrl+C 退出
python run_realsense_pose_surface.py --no-render --no-preview

# 按拍摄距离调整深度过滤
python run_realsense_pose_surface.py --min-depth 0.3 --max-depth 3.5 --depth-window 5 --depth-tolerance 0.15

# 指定相机（替换为实际序列号）
python run_realsense_pose_surface.py --serial YOUR_CAMERA_SERIAL
```

彩色和深度分辨率可分别用 `--width / --height` 与 `--depth-width / --depth-height` 设置。请使用设备支持的分辨率与帧率组合。

完整参数、算法说明及排错方法见 **[详细使用文档](realsense_pose_surface/README.md)**。

## 代码结构

```text
realsense_pose_surface/
├── run_realsense_pose_surface.py  # 相机采集、姿态推理、预览与三维显示
├── pose_geometry.py              # 深度采样、时间平滑与骨段几何
├── requirements.txt              # 运行依赖
├── install_ubuntu.sh             # 创建虚拟环境并安装依赖
└── README.md                     # 参数和排错说明
tests/
├── test_pose_geometry.py         # 数值处理与参数验证
└── test_runtime.py               # 设备接口、网格复用与资源释放
.github/workflows/tests.yml       # Python 3.10 / 3.12 自动测试
```

## 测试

在仓库根目录、已激活的虚拟环境中执行：

```bash
python -m pip install 'numpy>=1.24,<2'
python -m unittest discover -s tests -v
```

21 项测试覆盖边界像素、深度单位和异常值、不同帧率、关键点丢失与重现、骨段变换、网格复用及异常退出清理。相机和窗口接口使用测试替身，GitHub Actions 在 Python 3.10、3.12 上自动执行。

## 适用范围

- 三维坐标以米为单位，位于相机坐标系：X 向右、Y 向下、Z 向前。接入机器人前需要另做外参标定。
- 本项目显示人体关键点和骨架；可见表面深度与真实关节中心存在差异，球体和骨段不代表真实人体表面。
- 当前使用单人 Pose 模型，无身份跟踪；严重遮挡、背景深度污染或人物切换可能影响结果。
- 无窗口模式不会自动保存数据。其他 RGB-D 设备需要编写采集适配器。
- 自动测试已覆盖处理逻辑；D455 采集、实际模型推理、三维窗口及跨场景准确率仍需实机验证。
