#!/usr/bin/env python3
"""Quick test script to verify camera adapter works on Mac."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from camera_adapter import create_camera, list_available_cameras

if __name__ == "__main__":
    print("Testing camera adapter on macOS...")
    print(f"Available cameras: {list_available_cameras()}")
    print()

    # Test with OpenCV (Mac camera)
    if os.getenv("CAMERA_MODE") != "test":
        print("Testing OpenCV camera (Mac)...")
        try:
            cam = create_camera(0, (1280, 720), 30.0, mode="opencv")
            print(f"Camera resolution: {cam.sensor_resolution}")
            frame = cam.capture_array()
            print(f"Frame shape: {frame.shape}")
            cam.close()
            print("✓ OpenCV camera works")
        except Exception as e:
            print(f"✗ OpenCV camera failed: {e}")
        print()

    # Test with test patterns
    print("Testing test pattern generator...")
    try:
        cam = create_camera(0, (1920, 1080), 60.0, mode="test")
        print(f"Test camera resolution: {cam.sensor_resolution}")
        start = time.time()
        frames = 0
        for _ in range(30):  # Capture 30 frames
            frame = cam.capture_array()
            frames += 1
        elapsed = time.time() - start
        fps = frames / elapsed
        print(f"Captured {frames} frames in {elapsed:.2f}s ({fps:.1f} fps)")
        cam.close()
        print("✓ Test patterns work")
    except Exception as e:
        print(f"✗ Test patterns failed: {e}")


