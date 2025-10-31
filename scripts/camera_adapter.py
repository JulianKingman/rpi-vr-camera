"""Camera abstraction supporting both RPI (picamera2) and Mac (OpenCV/test patterns)."""

from __future__ import annotations

import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np

# Try importing picamera2, but don't fail if not available (e.g., on Mac)
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    Picamera2 = None  # type: ignore[assignment, misc]


class CameraInterface(ABC):
    """Abstract interface for camera access."""

    @abstractmethod
    def capture_array(self) -> np.ndarray:
        """Capture a frame as numpy array (BGR format)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release camera resources."""
        pass

    @property
    @abstractmethod
    def sensor_resolution(self) -> Tuple[int, int]:
        """Return sensor resolution (width, height)."""
        pass


class Picamera2Adapter(CameraInterface):
    """Adapter for picamera2 on Raspberry Pi."""

    def __init__(self, camera_num: int, resolution: Tuple[int, int], framerate: float, **kwargs):
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError("picamera2 not available (not on Raspberry Pi?)")
        self.cam = Picamera2(camera_num=camera_num)
        config = self.cam.create_video_configuration(
            main={"size": resolution},
            controls={"FrameRate": framerate},
            buffer_count=6,
        )
        self.cam.configure(config)
        self.cam.start()
        self._resolution = resolution
        # Apply runtime controls if provided
        runtime_controls = kwargs.get("controls", {})
        if runtime_controls:
            self.cam.set_controls(runtime_controls)

    def capture_array(self) -> np.ndarray:
        return self.cam.capture_array()

    def close(self) -> None:
        self.cam.close()

    @property
    def sensor_resolution(self) -> Tuple[int, int]:
        return self.cam.sensor_resolution


class OpenCVCameraAdapter(CameraInterface):
    """Adapter using OpenCV VideoCapture (works on Mac/Windows/Linux)."""

    def __init__(
        self,
        camera_num: int,
        resolution: Tuple[int, int],
        framerate: float,
        **kwargs,
    ):
        self.cap = cv2.VideoCapture(camera_num)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_num}")
        # Set resolution (may not be honored by all cameras)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, framerate)
        # Read actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolution = (actual_width, actual_height)
        self._target_resolution = resolution
        self._framerate = framerate
        print(f"[INFO] Camera {camera_num}: {actual_width}x{actual_height} @ {framerate} fps")

    def capture_array(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from camera")
        # Resize to target resolution if needed
        if frame.shape[:2][::-1] != self._target_resolution:
            frame = cv2.resize(frame, self._target_resolution, interpolation=cv2.INTER_AREA)
        return frame

    def close(self) -> None:
        self.cap.release()

    @property
    def sensor_resolution(self) -> Tuple[int, int]:
        return self._resolution


class TestPatternCamera(CameraInterface):
    """Generates synthetic test patterns for latency testing without hardware."""

    def __init__(
        self,
        resolution: Tuple[int, int],
        framerate: float,
        pattern_type: str = "moving_bars",
        **kwargs,
    ):
        self._resolution = resolution
        self._framerate = framerate
        self._pattern_type = pattern_type
        self._frame_count = 0
        self._start_time = time.monotonic()
        # Frame time in seconds
        self._frame_interval = 1.0 / framerate if framerate > 0 else 1.0 / 60.0

    def capture_array(self) -> np.ndarray:
        # Generate frame based on pattern type
        if self._pattern_type == "moving_bars":
            frame = self._generate_moving_bars()
        elif self._pattern_type == "color_cycle":
            frame = self._generate_color_cycle()
        elif self._pattern_type == "timestamp":
            frame = self._generate_timestamp_pattern()
        else:
            frame = self._generate_checkerboard()

        self._frame_count += 1
        # Throttle to target framerate
        elapsed = time.monotonic() - self._start_time
        expected_time = self._frame_count * self._frame_interval
        if elapsed < expected_time:
            time.sleep(expected_time - elapsed)

        return frame

    def _generate_moving_bars(self) -> np.ndarray:
        """Generate moving vertical bars pattern for motion/latency testing."""
        width, height = self._resolution
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Calculate bar position based on frame count
        bar_width = width // 20
        offset = (self._frame_count * 2) % (width + bar_width * 2) - bar_width
        for x in range(-bar_width, width + bar_width, bar_width * 2):
            pos = x + offset
            if 0 <= pos < width:
                frame[:, max(0, pos) : min(width, pos + bar_width)] = [255, 255, 255]
        return frame

    def _generate_color_cycle(self) -> np.ndarray:
        """Generate color cycling pattern."""
        width, height = self._resolution
        hue = (self._frame_count * 2) % 360
        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        hsv[:, :, 0] = hue
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = 255
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame

    def _generate_timestamp_pattern(self) -> np.ndarray:
        """Generate pattern with visible timestamp for latency measurement."""
        width, height = self._resolution
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background checkerboard
        square_size = 32
        for y in range(0, height, square_size):
            for x in range(0, width, square_size):
                color = 128 if ((x // square_size) + (y // square_size)) % 2 == 0 else 64
                frame[y : y + square_size, x : x + square_size] = [color, color, color]

        # Overlay timestamp text
        timestamp = time.monotonic()
        text = f"T:{timestamp:.4f} F:{self._frame_count}"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        return frame

    def _generate_checkerboard(self) -> np.ndarray:
        """Generate static checkerboard pattern."""
        width, height = self._resolution
        square_size = 64
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(0, height, square_size):
            for x in range(0, width, square_size):
                color = 255 if ((x // square_size) + (y // square_size)) % 2 == 0 else 0
                frame[y : y + square_size, x : x + square_size] = [color, color, color]
        return frame

    def close(self) -> None:
        pass

    @property
    def sensor_resolution(self) -> Tuple[int, int]:
        return self._resolution


def create_camera(
    index: int,
    resolution: Tuple[int, int],
    framerate: float,
    mode: Optional[str] = None,
    **kwargs,
) -> CameraInterface:
    """
    Create a camera adapter based on availability and mode.

    Args:
        index: Camera index (0, 1, etc. for real cameras)
        resolution: Target resolution (width, height)
        framerate: Target framerate
        mode: One of 'auto', 'rpi', 'opencv', 'test'. If None, auto-detects.
        **kwargs: Additional arguments passed to camera adapter

    Returns:
        CameraInterface instance
    """
    if mode is None:
        mode = os.getenv("CAMERA_MODE", "auto")

    if mode == "test":
        pattern = os.getenv("TEST_PATTERN", "moving_bars")
        return TestPatternCamera(resolution, framerate, pattern_type=pattern, **kwargs)

    if mode == "opencv" or (mode == "auto" and not PICAMERA2_AVAILABLE):
        return OpenCVCameraAdapter(index, resolution, framerate, **kwargs)

    if mode == "rpi" or (mode == "auto" and PICAMERA2_AVAILABLE):
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError(
                "picamera2 requested but not available. "
                "Use mode='opencv' or mode='test' for Mac testing."
            )
        return Picamera2Adapter(index, resolution, framerate, **kwargs)

    raise ValueError(f"Unknown camera mode: {mode}")


def list_available_cameras(mode: Optional[str] = None) -> list[int]:
    """List available camera indices."""
    if mode is None:
        mode = os.getenv("CAMERA_MODE", "auto")

    if mode == "test":
        return [0, 1]  # Test cameras always available

    if mode == "opencv" or (mode == "auto" and not PICAMERA2_AVAILABLE):
        available = []
        for i in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    if mode == "rpi" or PICAMERA2_AVAILABLE:
        # On RPI, assume cameras 0 and 1 if picamera2 is available
        return [0, 1]

    return []


