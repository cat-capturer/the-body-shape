#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 -m venv .venv
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
