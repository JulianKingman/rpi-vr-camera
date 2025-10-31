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

# Detect if running on macOS vs Linux/RPI
if [[ "$(uname)" == "Darwin" ]]; then
  echo "Detected macOS - installing Mac-compatible packages..."
  # Mac: Skip picamera2 (RPI-only)
  pip install aiortc aiohttp opencv-python numpy pyyaml PySide6 || true
  echo "Note: picamera2 is RPI-only and skipped on Mac"
else
  # Linux/RPI: Install all packages including picamera2
  pip install aiortc aiohttp opencv-python numpy pyyaml PySide6
  pip install picamera2 || echo "Warning: picamera2 installation failed (expected on non-RPI systems)"
  
  # Link libcamera bindings (RPI only)
  "$PYTHON" - <<'PY'
from pathlib import Path
import sys

site = Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
site.mkdir(parents=True, exist_ok=True)
pth = site / 'rpi_libcamera.pth'
libcamera_path = Path('/usr/lib/python3/dist-packages')
if libcamera_path.exists():
    pth.write_text('/usr/lib/python3/dist-packages\n')
    print(f'Linked libcamera bindings via {pth}')
else:
    print('libcamera not found (expected on non-RPI systems)')
PY
fi

echo ""
echo "Installation complete! To activate the environment:"
echo "  source .venv/bin/activate"
