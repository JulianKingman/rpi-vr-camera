#!/usr/bin/env python3
"""Enumerate Raspberry Pi cameras and available modes using Picamera2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SYSTEM_DIST = Path("/usr/lib/python3/dist-packages")
if SYSTEM_DIST.exists() and str(SYSTEM_DIST) not in sys.path:
    sys.path.append(str(SYSTEM_DIST))

from picamera2 import Picamera2


def format_mode(mode: dict[str, Any]) -> str:
    size = mode.get("size") or (0, 0)
    fps = mode.get("fps")
    unpacked = mode.get("format", "?")
    width, height = size
    fps_text = f"{fps:.2f} fps" if fps else "fps ?"
    return f"{width}x{height} @ {fps_text} ({unpacked})"


def list_cameras(show_controls: bool = False) -> None:
    picams = Picamera2.global_camera_info()
    if not picams:
        print("No cameras detected", file=sys.stderr)
        sys.exit(1)

    for idx, info in enumerate(picams):
        model = info.get("Model", "Unknown")
        flags = info.get("Flags")
        print(f"Camera {idx}: {model}" + (f" ({flags})" if flags else ""))
        location = info.get("Location")
        if location:
            print(f"  Location: {location}")
        resolution = info.get("Resolution")
        if resolution:
            print(f"  Resolution: {resolution}")
        modes = info.get("modes", [])
        if modes:
            print("  Modes:")
            for mode in modes:
                print(f"    - {format_mode(mode)}")
        if show_controls:
            controls = info.get("Controls", {})
            if controls:
                print("  Controls:")
                for name, meta in controls.items():
                    print(f"    - {name}: {meta}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-controls",
        action="store_true",
        help="include per-camera control metadata",
    )
    args = parser.parse_args()
    list_cameras(args.show_controls)


if __name__ == "__main__":
    main()
