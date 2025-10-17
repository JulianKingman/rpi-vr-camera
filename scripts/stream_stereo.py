#!/usr/bin/env python3
"""Real-time side-by-side preview or headless pipeline using Picamera2."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from picamera2 import Picamera2

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "camera_profiles.yaml"


def load_profile(key: str) -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file missing: {CONFIG_PATH}")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    profiles = config.get("profiles", {})
    if key not in profiles:
        raise KeyError(f"Profile '{key}' not found in {CONFIG_PATH}")
    return profiles[key]


def setup_camera(index: int, profile: dict) -> tuple[Picamera2, tuple[int, int], dict]:
    resolution = tuple(profile.get("resolution", [1536, 864]))
    controls = {"FrameRate": profile.get("frame_rate", 60)}
    cam = Picamera2(camera_num=index)
    config = cam.create_video_configuration(
        main={"size": resolution},
        controls=controls,
        buffer_count=6,
    )
    cam.configure(config)
    cam.start()
    return cam, resolution, profile


def process_frame(frame: np.ndarray, profile: dict, source_resolution: tuple[int, int]) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crop_w, crop_h = profile.get("crop", source_resolution)
    offset_x = profile.get("offset_x", 0)
    offset_y = profile.get("offset_y", 0)
    hflip = profile.get("hflip", False)
    vflip = profile.get("vflip", False)
    rotation = profile.get("rotation", 0)

    max_x0 = max(0, source_resolution[0] - crop_w)
    max_y0 = max(0, source_resolution[1] - crop_h)
    x0 = int(np.clip(source_resolution[0] // 2 - crop_w // 2 + offset_x, 0, max_x0))
    y0 = int(np.clip(source_resolution[1] // 2 - crop_h // 2 + offset_y, 0, max_y0))
    x1 = int(np.clip(x0 + crop_w, 0, source_resolution[0]))
    y1 = int(np.clip(y0 + crop_h, 0, source_resolution[1]))
    x0 = max(0, x1 - crop_w)
    y0 = max(0, y1 - crop_h)
    cropped = frame[y0:y1, x0:x1].copy()

    if hflip:
        cropped = cv2.flip(cropped, 1)
    if vflip:
        cropped = cv2.flip(cropped, 0)

    rot_map = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    rot_code = rot_map.get(rotation % 360)
    if rot_code is not None:
        cropped = cv2.rotate(cropped, rot_code)

    side = min(cropped.shape[0], cropped.shape[1])
    if side <= 0:
        raise ValueError("Invalid crop dimensions; verify calibration settings.")
    start_x = (cropped.shape[1] - side) // 2
    start_y = (cropped.shape[0] - side) // 2
    square = cropped[start_y : start_y + side, start_x : start_x + side]
    return square


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to camera profiles YAML")
    parser.add_argument("--framerate", type=int, default=60, help="Target frame rate for capture")
    parser.add_argument("--headless", action="store_true", help="Run without GUI (prints frame timing)")
    parser.add_argument("--duration", type=float, default=0.0, help="Optional duration limit in seconds (0=run forever)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global CONFIG_PATH
    CONFIG_PATH = args.config

    try:
        left_profile = load_profile("cam0")
        right_profile = load_profile("cam1")
    except (FileNotFoundError, KeyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        left_profile["frame_rate"] = args.framerate
        right_profile["frame_rate"] = args.framerate
        left_cam, left_res, _ = setup_camera(0, left_profile)
        right_cam, right_res, _ = setup_camera(1, right_profile)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to start cameras: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        if not args.headless:
            cv2.namedWindow("Stereo Preview", cv2.WINDOW_NORMAL)
        start_time = time.time()
        frame_count = 0
        while True:
            left_frame = left_cam.capture_array()
            right_frame = right_cam.capture_array()

            left_processed = process_frame(left_frame, left_profile, left_res)
            right_processed = process_frame(right_frame, right_profile, right_res)

            side = min(left_processed.shape[0], right_processed.shape[0])
            left_processed = cv2.resize(left_processed, (side, side), interpolation=cv2.INTER_AREA)
            right_processed = cv2.resize(right_processed, (side, side), interpolation=cv2.INTER_AREA)
            combined = np.hstack([left_processed, right_processed])
            frame_count += 1

            if args.headless:
                if frame_count % 120 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed else 0.0
                    print(f"[INFO] Frames: {frame_count}, elapsed: {elapsed:.2f}s, approx FPS: {fps:.2f}")
            else:
                cv2.imshow("Stereo Preview", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if args.duration and (time.time() - start_time) >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Streaming failure: {exc}", file=sys.stderr)
    finally:
        left_cam.close()
        right_cam.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
