#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else "Use Python 3.10–3.12 for the pinned MediaPipe runtime.")'
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

cat <<'MSG'

Install complete.

Before running, make sure librealsense2 and udev rules are installed:
  https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

Run:
  source .venv/bin/activate
  python run_realsense_pose_surface.py

MSG
