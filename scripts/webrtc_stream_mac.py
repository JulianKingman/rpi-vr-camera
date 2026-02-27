#!/usr/bin/env python3
"""WebRTC/WebXR streamer for Mac (software encoding with camera adapter)."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import signal
import ssl
import sys
import time
from fractions import Fraction
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import yaml
from dataclasses import dataclass
from aiohttp import web
from aiortc import AudioStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCRtpSender, VideoStreamTrack
from aiortc.codecs import h264 as h264_codecs
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, VideoFrame

from camera_adapter import create_camera
from cam_utils import CONFIG_PATH, load_profile, parse_resolution

STATIC_DIR = Path(__file__).resolve().parent.parent / "web"


def process_frame(frame: np.ndarray, profile: dict, source_resolution: Tuple[int, int]) -> np.ndarray:
    """Process frame according to profile (crop, flip, rotate)."""
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

    return cropped


@dataclass
class StereoFrame:
    timestamp: float
    left_image: np.ndarray
    right_image: np.ndarray


class StereoCapture:
    """Captures from two cameras using camera adapter."""

    def __init__(
        self,
        config_path: Path,
        framerate: int,
        resolution: Optional[Tuple[int, int]] = None,
    ):
        self.config_path = config_path
        self.framerate = framerate
        left_profile = copy.deepcopy(load_profile("cam0", config_path))
        right_profile = copy.deepcopy(load_profile("cam1", config_path))

        if resolution is not None:
            res_list = [int(resolution[0]), int(resolution[1])]
            left_profile["resolution"] = res_list.copy()
            right_profile["resolution"] = res_list.copy()

        left_profile["frame_rate"] = framerate
        right_profile["frame_rate"] = framerate

        left_res = tuple(left_profile.get("resolution", [1920, 1080]))
        right_res = tuple(right_profile.get("resolution", [1920, 1080]))

        # On macOS, camera indices are not stable; Continuity Camera can show up as 0 or 1.
        # Default to 0 (override via MAC_CAMERA_INDEX) to avoid hardcoding assumptions.
        camera_index = int(os.getenv("MAC_CAMERA_INDEX", "0"))
        mode = os.getenv("CAMERA_MODE", "auto")
        self.left_cam = create_camera(camera_index, left_res, float(framerate), mode=mode)
        self.right_cam = create_camera(camera_index, right_res, float(framerate), mode=mode)

        self.left_profile = left_profile
        self.right_profile = right_profile
        self.left_res = left_res
        self.right_res = right_res

        self.frame_lock = Lock()
        self.latest: Optional[StereoFrame] = None
        self._frame_seq = 0
        self.stop_event = Event()
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self) -> None:
        interval = 1.0 / self.framerate
        next_tick = time.monotonic() + interval
        try:
            while not self.stop_event.is_set():
                left_frame = self.left_cam.capture_array()
                right_frame = self.right_cam.capture_array()

                left_processed = process_frame(left_frame, self.left_profile, self.left_res)
                right_processed = process_frame(right_frame, self.right_profile, self.right_res)

                with self.frame_lock:
                    self.latest = StereoFrame(timestamp=time.time(), left_image=left_processed, right_image=right_processed)
                    self._frame_seq += 1

                # Sleep precisely: account for time already spent capturing/processing
                now = time.monotonic()
                sleep_for = next_tick - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                next_tick += interval
                # If we fell behind by more than one frame, reset to avoid burst catch-up
                if next_tick < time.monotonic():
                    next_tick = time.monotonic() + interval
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Capture loop failed: {exc}", file=sys.stderr)
        finally:
            self.left_cam.close()
            self.right_cam.close()

    def get_fresh(self, last_seq: int) -> Tuple[StereoFrame, int]:
        """Block until a frame newer than last_seq is available."""
        while not self.stop_event.is_set():
            with self.frame_lock:
                if self.latest is not None and self._frame_seq > last_seq:
                    return self.latest, self._frame_seq
            time.sleep(0.001)
        raise MediaStreamError

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
    """Video track that streams stereo side-by-side frames."""

    def __init__(self, capture: StereoCapture, framerate: int, side: str):
        super().__init__()
        self.capture = capture
        self.framerate = framerate
        self.side = side  # "left" or "right"
        self._last_seq = 0
        self._t0: Optional[float] = None
        self._time_base = Fraction(1, 90000)  # Standard RTP video clock

    async def recv(self) -> VideoFrame:
        # Wait for a genuinely new frame — capture thread provides the pacing
        loop = asyncio.get_event_loop()
        frame, self._last_seq = await loop.run_in_executor(
            None, self.capture.get_fresh, self._last_seq
        )
        image = frame.left_image if self.side == "left" else frame.right_image

        # Resize if needed to square format
        side_length = min(image.shape[0], image.shape[1])
        if side_length < image.shape[0] or side_length < image.shape[1]:
            start_x = (image.shape[1] - side_length) // 2
            start_y = (image.shape[0] - side_length) // 2
            image = image[start_y : start_y + side_length, start_x : start_x + side_length]

        # Compute PTS from frame timestamp — no extra sleep
        if self._t0 is None:
            self._t0 = frame.timestamp
        pts = int((frame.timestamp - self._t0) * 90000)

        video_frame = VideoFrame.from_ndarray(image, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = self._time_base
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
    def __init__(self, capture: StereoCapture, ca_cert: Path | None = None):
        self.capture = capture
        self.ca_cert = ca_cert
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
        self._stats_channel = None

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:
            self._stats_channel = channel
            # HUD instrumentation channel (RTT ping). Minimal ping/pong protocol.
            @channel.on("message")
            def on_message(message) -> None:
                if not isinstance(message, str):
                    return
                try:
                    payload = json.loads(message)
                except Exception:  # noqa: BLE001
                    return
                if (
                    isinstance(payload, dict)
                    and payload.get("type") == "ping"
                    and isinstance(payload.get("seq"), int)
                    and isinstance(payload.get("t"), (int, float))
                ):
                    channel.send(json.dumps({"type": "pong", "seq": payload["seq"], "t": payload["t"]}))

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if pc.connectionState in {"failed", "closed"}:
                await pc.close()
                self.pcs.discard(pc)

        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        await pc.setRemoteDescription(offer)

        video_track_left = StereoVideoTrack(self.capture, self.capture.framerate, "left")
        video_track_right = StereoVideoTrack(self.capture, self.capture.framerate, "right")
        audio_track = SilenceAudioTrack()

        try:
            video_codecs = RTCRtpSender.getCapabilities("video").codecs  # type: ignore[attr-defined]
        except AttributeError:
            video_codecs = None

        h264_codecs_list = [c for c in video_codecs if c.mimeType.lower() == "video/h264"] if video_codecs else []
        # Prefer baseline / constrained baseline H.264 to reduce reordering/buffering (no B-frames).
        if h264_codecs_list:
            def _profile_level_id(codec) -> str | None:
                fmtp = getattr(codec, "sdpFmtpLine", "") or ""
                m = re.search(r"profile-level-id=([0-9a-fA-F]{6})", fmtp)
                return m.group(1).lower() if m else None

            baseline = [c for c in h264_codecs_list if (_profile_level_id(c) or "").startswith("42")]
            if baseline:
                h264_codecs_list = baseline

        if h264_codecs_list:
            print("[INFO] Preferring H.264 codecs for video stream", flush=True)
        else:
            print("[WARN] H.264 codec not advertised by client; using default preferences", flush=True)

        # Get existing video transceivers from the offer or create new ones
        video_transceivers = [t for t in pc.getTransceivers() if t.kind == "video"]
        
        # If client didn't offer enough transceivers, add more
        while len(video_transceivers) < 2:
            try:
                transceiver = pc.addTransceiver("video")
                video_transceivers.append(transceiver)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to add transceiver: {exc}", file=sys.stderr)
                break

        # Use first two transceivers for left and right
        if len(video_transceivers) >= 2:
            left_transceiver = video_transceivers[0]
            right_transceiver = video_transceivers[1]
        elif len(video_transceivers) == 1:
            # Fallback: only one transceiver available
            left_transceiver = video_transceivers[0]
            right_transceiver = pc.addTransceiver("video")
        else:
            # No transceivers, create both
            left_transceiver = pc.addTransceiver("video")
            right_transceiver = pc.addTransceiver("video")

        # Set direction and codec preferences
        for transceiver in [left_transceiver, right_transceiver]:
            try:
                transceiver.direction = "sendonly"
            except Exception:  # noqa: BLE001
                pass
            if h264_codecs_list:
                try:
                    transceiver.setCodecPreferences(h264_codecs_list)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Failed to set codec preferences: {exc}", file=sys.stderr)

        left_transceiver.sender.replaceTrack(video_track_left)
        right_transceiver.sender.replaceTrack(video_track_right)

        # Set bitrate if configured
        bitrate_mbps = self.capture.left_profile.get("bitrate_mbps") or self.capture.right_profile.get("bitrate_mbps")
        if bitrate_mbps:
            bitrate_bps = int(float(bitrate_mbps) * 1_000_000)
            for sender in [left_transceiver.sender, right_transceiver.sender]:
                encoder = getattr(sender, "_RTCRtpSender__encoder", None)
                if encoder and hasattr(encoder, "target_bitrate"):
                    try:
                        encoder.target_bitrate = bitrate_bps
                    except Exception as exc:  # noqa: BLE001
                        print(f"[WARN] Failed to set encoder bitrate: {exc}", file=sys.stderr)

        # Handle audio transceiver
        audio_transceiver = next((t for t in pc.getTransceivers() if t.kind == "audio"), None)
        if audio_transceiver:
            try:
                audio_transceiver.direction = "sendonly"
            except Exception:  # noqa: BLE001
                pass
            audio_transceiver.sender.replaceTrack(audio_track)
        else:
            # Client didn't offer audio, add one
            audio_transceiver = pc.addTransceiver("audio")
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

        async def log_sender_stats() -> None:
            try:
                while pc.connectionState not in {"failed", "closed"}:
                    stats = await pc.getStats()
                    track_idx = 0
                    for report in stats.values():
                        if report.type == "outbound-rtp" and getattr(report, "kind", None) == "video":
                            mid = getattr(report, "mid", None) or str(track_idx)
                            bytes_sent = getattr(report, "bytesSent", 0)
                            frames_sent = getattr(report, "framesSent", 0)
                            key_frames_sent = getattr(report, "keyFramesSent", 0)
                            print(
                                f"[STATS] mid={mid} "
                                f"bytes={bytes_sent} "
                                f"frames={frames_sent} "
                                f"keyFrames={key_frames_sent}",
                                flush=True,
                            )
                            if self._stats_channel and self._stats_channel.readyState == "open":
                                try:
                                    self._stats_channel.send(json.dumps({
                                        "type": "server_stats",
                                        "mid": mid,
                                        "bytesSent": bytes_sent,
                                        "framesSent": frames_sent,
                                        "keyFramesSent": key_frames_sent,
                                        "droppedFrames": 0,
                                    }))
                                except Exception:
                                    pass
                            track_idx += 1
                    await asyncio.sleep(1.0)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] stats logger stopped: {exc}", file=sys.stderr)

        asyncio.create_task(log_sender_stats())

        return web.Response(
            content_type="application/json",
            text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
        )

    async def cleanup(self) -> None:
        coros = [pc.close() for pc in list(self.pcs)]
        await asyncio.gather(*coros, return_exceptions=True)
        self.pcs.clear()

    async def serve_ca_certificate(self, _request: web.Request) -> web.StreamResponse:
        if not self.ca_cert or not self.ca_cert.exists():
            return web.Response(status=404, text="CA certificate not configured.")
        response = web.FileResponse(self.ca_cert)
        response.headers["Content-Disposition"] = f'attachment; filename=\"{self.ca_cert.name}\"'
        response.headers["Cache-Control"] = "no-store"
        return response


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP to bind")
    parser.add_argument("--port", type=int, default=8443, help="HTTP/WebSocket port for signaling")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to calibration profiles")
    parser.add_argument("--framerate", type=int, default=60, help="Capture framerate for both cameras")
    parser.add_argument(
        "--resolution",
        type=str,
        help="Override sensor resolution as WIDTHxHEIGHT (applies to both cameras)",
    )
    parser.add_argument(
        "--cert",
        type=Path,
        help="Path to TLS certificate (PEM) for HTTPS/WebRTC (requires --key)",
    )
    parser.add_argument(
        "--key",
        type=Path,
        help="Path to TLS private key (PEM) for HTTPS/WebRTC (requires --cert)",
    )
    parser.add_argument(
        "--ca-cert",
        type=Path,
        help="Path to CA certificate (PEM) to expose for download at /ca.crt",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    ca_cert_path: Path | None = None
    resolution_override: Optional[Tuple[int, int]] = None

    if args.resolution:
        try:
            resolution_override = parse_resolution(args.resolution)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(1)
    if args.ca_cert:
        ca_cert_path = args.ca_cert.expanduser()
        if not ca_cert_path.exists():
            print(f"[ERROR] CA certificate not found: {ca_cert_path}", file=sys.stderr)
            sys.exit(1)

    try:
        capture = StereoCapture(args.config, args.framerate, resolution_override)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Unable to start capture: {exc}", file=sys.stderr)
        sys.exit(1)

    server = WebRTCServer(capture, ca_cert_path)
    app = web.Application()
    app["rtc_server"] = server
    app.router.add_get("/", server.index)
    app.router.add_post("/offer", server.offer)
    app.router.add_static("/static/", STATIC_DIR, show_index=True)
    if ca_cert_path:
        app.router.add_get("/ca.crt", server.serve_ca_certificate)

    async def shutdown(app_: web.Application) -> None:
        await server.cleanup()
        capture.shutdown()

    app.on_shutdown.append(shutdown)

    ssl_context = None
    if args.cert or args.key:
        if not (args.cert and args.key):
            print("[ERROR] --cert and --key must be provided together for HTTPS", file=sys.stderr)
            sys.exit(1)
        cert_path = args.cert.expanduser()
        key_path = args.key.expanduser()
        if not cert_path.exists():
            print(f"[ERROR] TLS certificate not found: {cert_path}", file=sys.stderr)
            sys.exit(1)
        if not key_path.exists():
            print(f"[ERROR] TLS private key not found: {key_path}", file=sys.stderr)
            sys.exit(1)
        try:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(
                certfile=str(cert_path),
                keyfile=str(key_path),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to load TLS certificate/key: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] Starting WebRTC server on {args.host}:{args.port}", flush=True)
    print(f"       Open http{'s' if ssl_context else ''}://{args.host}:{args.port} in your browser", flush=True)
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    main()

