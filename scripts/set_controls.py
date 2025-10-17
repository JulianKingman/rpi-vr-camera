#!/usr/bin/env python3
"""Apply framerate, exposure, and gain presets to a Raspberry Pi camera."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cam_utils import build_transform, load_profile, open_camera  # noqa: E402


CONTROL_MAP = {
    "fps": "FrameRate",
    "exposure": "ExposureTime",
    "gain": "AnalogueGain",
    "awb": "AwbEnable",
    "ae": "AeEnable",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, required=True, help="camera index (0 or 1)")
    parser.add_argument("--profile", type=str, default="cam0", help="profile name for defaults")
    parser.add_argument("--fps", type=float, help="target frame rate")
    parser.add_argument("--exposure", type=int, help="exposure time in microseconds")
    parser.add_argument("--gain", type=float, help="analogue gain multiplier")
    parser.add_argument("--disable-ae", action="store_true", help="disable automatic exposure")
    parser.add_argument("--disable-awb", action="store_true", help="disable auto white balance")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds to keep camera active")
    return parser.parse_args()


def build_controls(args: argparse.Namespace) -> Dict[str, object]:
    controls: Dict[str, object] = {}
    if args.fps:
        controls[CONTROL_MAP["fps"]] = float(args.fps)
    if args.exposure:
        controls[CONTROL_MAP["exposure"]] = int(args.exposure)
    if args.gain:
        controls[CONTROL_MAP["gain"]] = float(args.gain)
    if args.disable_ae:
        controls[CONTROL_MAP["ae"]] = False
    if args.disable_awb:
        controls[CONTROL_MAP["awb"]] = False
    return controls


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)
    resolution = tuple(profile.get("resolution", [1536, 864]))
    rotation = profile.get("rotation", 0)
    hflip = profile.get("hflip", False)
    vflip = profile.get("vflip", False)

    transform = build_transform(rotation, hflip, vflip)
    user_controls = build_controls(args)

    if not user_controls:
        raise SystemExit("No controls specified. Use --fps/--exposure/--gain/etc.")

    cam = open_camera(args.camera, transform, resolution, controls=None)

    cam.start()
    time.sleep(0.5)
    cam.set_controls(user_controls)
    print(f"Applied controls to camera {args.camera}: {user_controls}")

    time.sleep(args.duration)
    metadata = cam.capture_metadata()
    print("Resulting metadata:")
    for key in ("FrameDuration", "FrameRate", "ExposureTime", "AnalogueGain"):
        if key in metadata:
            print(f"  {key}: {metadata[key]}")

    cam.close()


if __name__ == "__main__":
    main()
