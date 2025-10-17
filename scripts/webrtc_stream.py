#!/usr/bin/env python3
"""WebRTC/WebXR streamer that serves a side-by-side stereo feed."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Dict, Tuple

import cv2
import numpy as np
import yaml
from aiohttp import web
from aiortc import AudioStreamTrack, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, VideoFrame
from picamera2 import Picamera2

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "camera_profiles.yaml"
STATIC_DIR = Path(__file__).resolve().parent.parent / "web"


def load_profile(name: str, config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    data = yaml.safe_load(config_path.read_text())
    profiles = data.get("profiles", {})
    if name not in profiles:
        raise KeyError(f"Profile '{name}' missing in {config_path}")
    return profiles[name]


def setup_camera(index: int, profile: dict, framerate: int) -> Tuple[Picamera2, Tuple[int, int]]:
    resolution = tuple(profile.get("resolution", [1536, 864]))
    cam = Picamera2(camera_num=index)
    config = cam.create_video_configuration(
        main={"size": resolution},
        controls={"FrameRate": framerate},
        buffer_count=6,
    )
    cam.configure(config)
    cam.start()
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


@dataclass
class StereoFrame:
    timestamp: float
    image: np.ndarray


class StereoCapture:
    def __init__(self, config_path: Path, framerate: int):
        self.config_path = config_path
        self.framerate = framerate
        self.left_profile = load_profile("cam0", config_path)
        self.right_profile = load_profile("cam1", config_path)
        self.left_profile.setdefault("frame_rate", framerate)
        self.right_profile.setdefault("frame_rate", framerate)

        self.left_cam, self.left_res = setup_camera(0, self.left_profile, framerate)
        self.right_cam, self.right_res = setup_camera(1, self.right_profile, framerate)

        self.frame_lock = Lock()
        self.latest: StereoFrame | None = None
        self.stop_event = Event()
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                left_frame = self.left_cam.capture_array()
                right_frame = self.right_cam.capture_array()

                left_processed = process_frame(left_frame, self.left_profile, self.left_res)
                right_processed = process_frame(right_frame, self.right_profile, self.right_res)

                side = min(left_processed.shape[0], right_processed.shape[0])
                left_processed = cv2.resize(left_processed, (side, side), interpolation=cv2.INTER_AREA)
                right_processed = cv2.resize(right_processed, (side, side), interpolation=cv2.INTER_AREA)
                combined = np.hstack([left_processed, right_processed])

                with self.frame_lock:
                    self.latest = StereoFrame(timestamp=time.time(), image=combined)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Capture loop failed: {exc}", file=sys.stderr)
        finally:
            self.left_cam.close()
            self.right_cam.close()

    def get_latest(self) -> StereoFrame:
        while not self.stop_event.is_set():
            with self.frame_lock:
                if self.latest is not None:
                    return self.latest
            time.sleep(0.001)
        raise MediaStreamError

    def shutdown(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)


class StereoVideoTrack(VideoStreamTrack):
    def __init__(self, capture: StereoCapture, framerate: int):
        super().__init__()
        self.capture = capture
        self.framerate = framerate

    async def recv(self) -> VideoFrame:
        frame = self.capture.get_latest()
        pts, time_base = await self.next_timestamp()
        video_frame = VideoFrame.from_ndarray(frame.image, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


class SilenceAudioTrack(AudioStreamTrack):
    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate
        self.samples_per_frame = 960  # 20 ms @ 48 kHz
        self._start = time.time()
        self._frame_count = 0

    async def recv(self) -> AudioFrame:
        frame = AudioFrame(format="s16", layout="mono", samples=self.samples_per_frame)
        frame.pts = int(self._frame_count * self.samples_per_frame)
        frame.sample_rate = self.sample_rate
        for plane in frame.planes:
            plane.update(b"\0" * plane.buffer_size)
        self._frame_count += 1
        await asyncio.sleep(self.samples_per_frame / self.sample_rate)
        return frame


class WebRTCServer:
    def __init__(self, capture: StereoCapture):
        self.capture = capture
        self.pcs: set[RTCPeerConnection] = set()

    async def index(self, request: web.Request) -> web.Response:
        html_path = STATIC_DIR / "index.html"
        if not html_path.exists():
            return web.Response(text="Client not found. Build web/index.html.", status=404)
        return web.FileResponse(html_path)

    async def offer(self, request: web.Request) -> web.Response:
        params = await request.json()
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if pc.connectionState in {"failed", "closed"}:
                await pc.close()
                self.pcs.discard(pc)

        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        await pc.setRemoteDescription(offer)

        video_track = StereoVideoTrack(self.capture, self.capture.framerate)
        audio_track = SilenceAudioTrack()

        video_transceiver = next((t for t in pc.getTransceivers() if t.kind == "video"), None)
        if video_transceiver is None:
            await pc.close()
            self.pcs.discard(pc)
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"error": "Remote offer did not request video."}),
            )

        video_transceiver.direction = "sendonly"
        if video_transceiver.sender:
            video_transceiver.sender.replaceTrack(video_track)

        audio_transceiver = next((t for t in pc.getTransceivers() if t.kind == "audio"), None)
        if audio_transceiver and audio_transceiver.sender:
            audio_transceiver.direction = "sendonly"
            audio_transceiver.sender.replaceTrack(audio_track)

        try:
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception as exc:  # noqa: BLE001
            await pc.close()
            self.pcs.discard(pc)
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": str(exc)}),
            )

        return web.Response(
            content_type="application/json",
            text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
        )

    async def cleanup(self) -> None:
        coros = [pc.close() for pc in list(self.pcs)]
        await asyncio.gather(*coros, return_exceptions=True)
        self.pcs.clear()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind (use 0.0.0.0 for all interfaces)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP/WebSocket port for signaling")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to calibration profiles")
    parser.add_argument("--framerate", type=int, default=60, help="Capture framerate for both cameras")
    parser.add_argument(
        "--ice",
        nargs="*",
        default=["stun:stun.l.google.com:19302"],
        help="ICE servers (pass as stun:host:port or turn:user@pass:host:port)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        capture = StereoCapture(args.config, args.framerate)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Unable to start capture: {exc}", file=sys.stderr)
        sys.exit(1)

    server = WebRTCServer(capture)
    app = web.Application()
    app["rtc_server"] = server
    app.router.add_get("/", server.index)
    app.router.add_post("/offer", server.offer)
    app.router.add_static("/static/", STATIC_DIR, show_index=True)

    semaphore = asyncio.Semaphore()

    async def shutdown(app_: web.Application) -> None:
        async with semaphore:
            await server.cleanup()
            capture.shutdown()

    app.on_shutdown.append(shutdown)

    loop = asyncio.get_event_loop()

    def handle_signal() -> None:
        asyncio.ensure_future(app.shutdown())
        asyncio.ensure_future(app.cleanup())
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
