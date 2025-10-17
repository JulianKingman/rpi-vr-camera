#!/usr/bin/env bash
# Bootstrap the Python virtual environment for rpi-vr-camera.
set -euo pipefail

: "${PYTHON:=python3}"

if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel
pip install aiortc opencv-python numpy picamera2 pyyaml PySide6

"$PYTHON" - <<'PY'
from pathlib import Path
import sys

site = Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
site.mkdir(parents=True, exist_ok=True)
pth = site / 'rpi_libcamera.pth'
pth.write_text('/usr/lib/python3/dist-packages\n')
print(f'Linked libcamera bindings via {pth}')
PY
