#!/usr/bin/env python3
"""Preview frames from a Raspberry Pi camera with optional rotation and crop."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SYSTEM_DIST = Path("/usr/lib/python3/dist-packages")
if SYSTEM_DIST.exists() and str(SYSTEM_DIST) not in sys.path:
    sys.path.append(str(SYSTEM_DIST))

from picamera2 import Picamera2, Preview

from cam_utils import (  # noqa: E402
    Crop,
    apply_center_crop,
    build_transform,
    open_camera,
    parse_resolution,
)


def attach_metadata_logger(picam: Picamera2, show_metadata: bool) -> None:
    if not show_metadata:
        return

    last_print = 0.0

    def _callback(request) -> None:
        nonlocal last_print
        now = time.monotonic()
        if now - last_print < 0.25:  # throttle output
            return
        last_print = now
        metadata = request.get_metadata()
        exposure = metadata.get("ExposureTime")
        gain = metadata.get("AnalogueGain")
        frame_time = metadata.get("FrameDuration")
        af_state = metadata.get("AfState")
        text = (
            f"Exposure: {exposure} µs | Gain: {gain:.2f} | "
            f"Frame: {frame_time / 1000:.2f} ms"
        )
        if af_state is not None:
            text += f" | AF: {af_state}"
        print(text, end="\r", flush=True)

    picam.post_callback = _callback


def choose_preview(name: str):  # noqa: ANN201
    if name == "qtgl":
        return Preview.QTGL
    if name == "drm":
        return Preview.DRM
    return Preview.NULL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument(
        "--resolution",
        type=str,
        default="1536x864",
        help="target capture size WxH (default 1536x864)",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=0,
        help="rotation to apply (0/90/180/270)",
    )
    parser.add_argument("--hflip", action="store_true", help="apply horizontal flip")
    parser.add_argument("--vflip", action="store_true", help="apply vertical flip")
    parser.add_argument(
        "--crop",
        type=str,
        help="center crop (e.g. 1536x864) applied before scaling",
    )
    parser.add_argument(
        "--preview",
        choices=["qtgl", "drm", "null"],
        default="qtgl",
        help="preview backend (qtgl, drm, null)",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="periodically print exposure/gain/frame metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        target_size = parse_resolution(args.resolution)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    transform = build_transform(args.rotation, args.hflip, args.vflip)
    cam = open_camera(args.camera, transform, target_size, controls={"FrameRate": 120.0})

    if args.crop:
        try:
            crop = Crop.parse(args.crop)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        apply_center_crop(cam, crop)

    attach_metadata_logger(cam, args.metadata)

    preview = choose_preview(args.preview)
    try:
        cam.start_preview(preview)
    except Exception as exc:  # noqa: BLE001
        print(f"Preview backend failed ({preview}): {exc}. Falling back to NULL preview.")
        cam.start_preview(Preview.NULL)

    cam.start()

    stop = False

    def _signal_handler(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        while not stop:
            time.sleep(0.1)
    finally:
        cam.stop_preview()
        cam.close()


if __name__ == "__main__":
    main()
