#!/usr/bin/env python3
"""Capture calibration frames for stereo alignment."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cam_utils import Crop, apply_center_crop, build_transform, load_profile, open_camera  # noqa: E402


def save_frame(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Failed to write {path}")


def capture_frames(
    cam_index: int,
    count: int,
    output_dir: Path,
    profile_name: str,
    delay: float,
) -> None:
    profile = load_profile(profile_name)
    resolution = tuple(profile.get("resolution", [1536, 864]))
    rotation = profile.get("rotation", 0)
    hflip = profile.get("hflip", False)
    vflip = profile.get("vflip", False)
    crop_vals = profile.get("crop")

    transform = build_transform(rotation, hflip, vflip)
    cam = open_camera(cam_index, transform, resolution, controls={"FrameRate": 120.0})

    if crop_vals:
        crop = Crop(width=int(crop_vals[0]), height=int(crop_vals[1]))
        apply_center_crop(cam, crop)

    cam.start()
    time.sleep(1.0)  # allow exposure to settle

    for idx in range(count):
        request = cam.capture_request()
        frame = request.make_array("main")
        request.release()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"cam{cam_index}_{timestamp}_{idx:03d}.png"
        save_frame(frame, filename)
        print(f"Saved {filename}")
        time.sleep(delay)

    cam.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, required=True, help="camera index (0 or 1)")
    parser.add_argument(
        "--profile",
        type=str,
        default="cam0",
        help="profile name in config/camera_profiles.yaml",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="number of frames to capture",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="seconds between captures",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/calibration"),
        help="directory to store captured frames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        capture_frames(args.camera, args.frames, args.output, args.profile, args.delay)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
