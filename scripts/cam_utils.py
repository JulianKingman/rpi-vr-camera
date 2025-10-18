"""Shared helpers for Raspberry Pi dual camera utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import sys

SYSTEM_DIST = Path("/usr/lib/python3/dist-packages")
if SYSTEM_DIST.exists() and str(SYSTEM_DIST) not in sys.path:
    sys.path.append(str(SYSTEM_DIST))

import yaml
from libcamera import Transform, controls
from picamera2 import Picamera2

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "camera_profiles.yaml"

_AWB_MODE_CANDIDATES = [
    ("auto", "Auto"),
    ("incandescent", "Incandescent"),
    ("tungsten", "Tungsten"),
    ("fluorescent", "Fluorescent"),
    ("indoor", "Indoor"),
    ("daylight", "Daylight"),
    ("cloudy", "Cloudy"),
    ("shade", "Shade"),
    ("sunlight", "Sunlight"),
    ("horizon", "Horizon"),
    ("custom", "Custom"),
]

AWB_MODE_MAP: Dict[str, int] = {}
for key, enum_name in _AWB_MODE_CANDIDATES:
    value = getattr(controls.AwbModeEnum, enum_name, None)
    if value is not None:
        AWB_MODE_MAP[key] = value


@dataclass
class Crop:
    width: int
    height: int

    @classmethod
    def parse(cls, value: str) -> "Crop":
        width_str, height_str = value.lower().split("x", maxsplit=1)
        width = int(width_str)
        height = int(height_str)
        if width <= 0 or height <= 0:
            raise ValueError("Crop dimensions must be positive")
        return cls(width, height)


def parse_resolution(value: str) -> Tuple[int, int]:
    width_str, height_str = value.lower().split("x", maxsplit=1)
    width = int(width_str)
    height = int(height_str)
    if width <= 0 or height <= 0:
        raise ValueError("Resolution must be positive")
    return width, height


def build_transform(rotation: int, hflip: bool, vflip: bool) -> Transform:
    rotation = rotation % 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Rotation must be 0/90/180/270 degrees")
    return Transform(rotation=rotation, hflip=hflip, vflip=vflip)


def apply_center_crop(picam: Picamera2, crop: Crop) -> None:
    sensor_width, sensor_height = picam.sensor_resolution
    if crop.width > sensor_width or crop.height > sensor_height:
        raise ValueError(
            f"Crop {crop.width}x{crop.height} exceeds sensor resolution "
            f"{sensor_width}x{sensor_height}"
        )
    x = max(0, (sensor_width - crop.width) // 2)
    y = max(0, (sensor_height - crop.height) // 2)
    picam.set_controls({"ScalerCrop": (x, y, crop.width, crop.height)})


def load_profiles(path: Optional[Path] = None) -> Dict[str, Any]:
    profile_path = path or CONFIG_PATH
    data = yaml.safe_load(profile_path.read_text())
    return data.get("profiles", {})


def load_profile(name: str, path: Optional[Path] = None) -> Dict[str, Any]:
    profiles = load_profiles(path)
    if name not in profiles:
        raise KeyError(f"Profile '{name}' not found in {CONFIG_PATH}")
    return profiles[name]


def get_awb_mode_names() -> Tuple[str, ...]:
    return tuple(AWB_MODE_MAP.keys())


def resolve_awb_mode(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    return AWB_MODE_MAP.get(str(name).lower())


def open_camera(
    index: int,
    transform: Transform,
    size: Optional[Tuple[int, int]],
    controls: Optional[Dict[str, Any]] = None,
) -> Picamera2:
    cam = Picamera2(camera_num=index)
    config = cam.create_video_configuration(
        main={"size": size} if size else None,
        controls=controls,
        transform=transform,
        buffer_count=6,
    )
    cam.configure(config)
    return cam
