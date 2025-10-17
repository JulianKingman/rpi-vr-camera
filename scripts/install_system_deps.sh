#!/usr/bin/env bash
# Install system-level dependencies for rpi-vr-camera.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt update
apt install -y \
  build-essential git cmake meson ninja-build pkg-config \
  libcap-dev \
  libcamera-dev libcamera-apps python3-libcamera python3-kms++ python3-prctl \
  gstreamer1.0-tools gstreamer1.0-libav \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  gstreamer1.0-alsa gstreamer1.0-plugins-base \
  ffmpeg
