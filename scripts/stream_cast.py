#!/usr/bin/env python3
"""Dual-camera live preview with simultaneous network stream."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml
from picamera2 import Picamera2

from cam_utils import resolve_awb_mode

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "camera_profiles.yaml"


def load_profile(name: str, config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    data = yaml.safe_load(config_path.read_text())
    profiles = data.get("profiles", {})
    if name not in profiles:
        raise KeyError(f"Profile '{name}' missing in {config_path}")
    return profiles[name]


def setup_camera(index: int, profile: dict, framerate: int) -> Tuple[Picamera2, Tuple[int, int]]:
    resolution = tuple(profile.get("resolution", [2304, 1296]))
    target_rate = float(profile.get("frame_rate", framerate))
    cam = Picamera2(camera_num=index)
    config = cam.create_video_configuration(
        main={"size": resolution},
        controls={"FrameRate": target_rate},
        buffer_count=6,
    )
    cam.configure(config)
    cam.start()
    runtime_controls: dict[str, object] = {}
    awb_enable = profile.get("awb_enable")
    gains = profile.get("colour_gains")
    mode_value = resolve_awb_mode(profile.get("awb_mode"))
    if awb_enable is not None:
        runtime_controls["AwbEnable"] = bool(awb_enable)
        if awb_enable:
            if mode_value is not None:
                runtime_controls["AwbMode"] = mode_value
    manual_wb = awb_enable is not None and not bool(awb_enable)
    if gains and len(gains) == 2 and manual_wb:
        runtime_controls["ColourGains"] = (float(gains[0]), float(gains[1]))
    if runtime_controls:
        cam.set_controls(runtime_controls)
    return cam, resolution


def process_frame(frame: np.ndarray, profile: dict, source_resolution: Tuple[int, int]) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crop_w, crop_h = profile.get("crop", source_resolution)
    offset_x = profile.get("offset_x", 0)
    offset_y = profile.get("offset_y", 0)
    hflip = profile.get("hflip", False)
    vflip = profile.get("vflip", False)
    rotation = profile.get("rotation", 0)

    max_x0 = max(0, source_resolution[0] - crop_w)
    max_y0 = max(0, source_resolution[1] - crop_h)
    x0_nominal = source_resolution[0] // 2 - crop_w // 2 + offset_x
    y0_nominal = source_resolution[1] // 2 - crop_h // 2 + offset_y
    x0 = int(np.clip(x0_nominal, 0, max_x0))
    y0 = int(np.clip(y0_nominal, 0, max_y0))
    x1 = int(np.clip(x0 + crop_w, 0, source_resolution[0]))
    y1 = int(np.clip(y0 + crop_h, 0, source_resolution[1]))
    x0 = max(0, x1 - crop_w)
    y0 = max(0, y1 - crop_h)
    cropped = frame_rgb[y0:y1, x0:x1].copy()

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
        raise ValueError("Invalid crop; adjust calibration settings.")
    start_x = (cropped.shape[1] - side) // 2
    start_y = (cropped.shape[0] - side) // 2
    return cropped[start_y : start_y + side, start_x : start_x + side].copy()


def spawn_ffmpeg(
    width: int,
    height: int,
    framerate: int,
    endpoint: str,
    bitrate_bps: Optional[int],
) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(framerate),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
    ]
    if bitrate_bps:
        cmd.extend(
            [
                "-b:v",
                str(bitrate_bps),
                "-maxrate",
                str(bitrate_bps),
                "-bufsize",
                str(max(bitrate_bps // 2, 1_000_000)),
            ]
        )
    cmd.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rtp" if endpoint.startswith("rtp") else "mpegts",
            endpoint,
        ]
    )
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Ensure it is installed and on PATH.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to calibration profiles")
    parser.add_argument("--framerate", type=int, default=56, help="Capture frame rate")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="udp://127.0.0.1:5000",
        help="Streaming endpoint (e.g. udp://host:port or rtp://host:port)",
    )
    parser.add_argument("--preview-scale", type=float, default=0.4, help="Scale factor for on-screen preview")
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV preview window")
    parser.add_argument("--duration", type=float, default=0.0, help="Optional duration limit in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        left_profile = load_profile("cam0", args.config)
        right_profile = load_profile("cam1", args.config)
    except (FileNotFoundError, KeyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    left_profile["frame_rate"] = args.framerate
    right_profile["frame_rate"] = args.framerate

    bitrate_mbps = left_profile.get("bitrate_mbps") or right_profile.get("bitrate_mbps")
    try:
        bitrate_bps: Optional[int] = int(float(bitrate_mbps) * 1_000_000) if bitrate_mbps is not None else None
    except Exception:  # noqa: BLE001
        bitrate_bps = None

    try:
        left_cam, left_res = setup_camera(0, left_profile, args.framerate)
        right_cam, right_res = setup_camera(1, right_profile, args.framerate)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to initialise cameras: {exc}", file=sys.stderr)
        sys.exit(1)

    combined_width = combined_height = None
    ffmpeg_proc: subprocess.Popen | None = None

    if not args.headless:
        cv2.namedWindow("Stereo Preview", cv2.WINDOW_NORMAL)

    start_time = time.time()
    frame_counter = 0

    try:
        while True:
            left_frame = left_cam.capture_array()
            right_frame = right_cam.capture_array()

            left_processed = process_frame(left_frame, left_profile, left_res)
            right_processed = process_frame(right_frame, right_profile, right_res)

            side = min(left_processed.shape[0], right_processed.shape[0])
            left_processed = cv2.resize(left_processed, (side, side), interpolation=cv2.INTER_AREA)
            right_processed = cv2.resize(right_processed, (side, side), interpolation=cv2.INTER_AREA)
            combined = np.hstack([left_processed, right_processed])

            if ffmpeg_proc is None:
                combined_height, combined_width = combined.shape[:2]
                ffmpeg_proc = spawn_ffmpeg(
                    combined_width,
                    combined_height,
                    args.framerate,
                    args.endpoint,
                    bitrate_bps,
                )
                if ffmpeg_proc.stdin is None:
                    raise RuntimeError("Failed to create ffmpeg stdin pipe.")

            frame_counter += 1

            try:
                ffmpeg_proc.stdin.write(combined.tobytes())
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Failed to write to ffmpeg: {exc}", file=sys.stderr)
                break

            if not args.headless:
                preview = combined
                if 0 < args.preview_scale < 1.0:
                    preview = cv2.resize(
                        combined,
                        None,
                        fx=args.preview_scale,
                        fy=args.preview_scale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                cv2.imshow("Stereo Preview", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                if frame_counter % max(1, args.framerate) == 0:
                    elapsed = time.time() - start_time
                    fps = frame_counter / elapsed if elapsed else 0.0
                    print(f"[INFO] Frames={frame_counter} elapsed={elapsed:.1f}s fps≈{fps:.2f}")

            if args.duration and (time.time() - start_time) >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        left_cam.close()
        right_cam.close()
        if not args.headless:
            cv2.destroyAllWindows()
        if ffmpeg_proc:
            try:
                ffmpeg_proc.stdin.close()
            except Exception:
                pass
            ffmpeg_proc.terminate()
            try:
                ffmpeg_proc.wait(timeout=2)
            except Exception:
                ffmpeg_proc.kill()


if __name__ == "__main__":
    main()
