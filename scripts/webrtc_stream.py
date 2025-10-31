#!/usr/bin/env python3
"""WebRTC/WebXR streamer that serves dual hardware-encoded stereo feeds."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import signal
import ssl
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple

import yaml
from aiohttp import web
from aiortc import AudioStreamTrack, MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCRtpSender
from aiortc.mediastreams import MediaStreamError
from av.packet import Packet
from av import AudioFrame
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import Output

from cam_utils import build_transform, resolve_awb_mode
from aiortc.codecs import h264 as h264_codecs

_H264_PATCHED = False
_H264_TARGET_BITRATE: Optional[int] = None
MICROSECOND_TIME_BASE = Fraction(1, 1_000_000)


def _ensure_h264_encoder_patch(target_bitrate: int) -> None:
    global _H264_PATCHED
    global _H264_TARGET_BITRATE
    _H264_TARGET_BITRATE = target_bitrate
    if not _H264_PATCHED:
        original_init = h264_codecs.H264Encoder.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if hasattr(self, "target_bitrate") and _H264_TARGET_BITRATE:
                self.target_bitrate = _H264_TARGET_BITRATE

        h264_codecs.H264Encoder.__init__ = patched_init  # type: ignore[assignment]
        _H264_PATCHED = True

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


def parse_resolution(value: str) -> Tuple[int, int]:
    try:
        width_str, height_str = value.lower().split("x", maxsplit=1)
        width = int(width_str)
        height = int(height_str)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid resolution format '{value}'. Use WIDTHxHEIGHT.") from exc
    if width <= 0 or height <= 0:
        raise ValueError("Resolution values must be positive integers.")
    return width, height


@dataclass
class EncodedSample:
    data: bytes
    pts: int
    time_base: Fraction
    keyframe: bool = False


@dataclass
class EncodedStreamSubscription:
    identifier: int
    queue: asyncio.Queue[Optional[EncodedSample]]
    broadcaster: "EncodedStreamBroadcaster"

    def close(self) -> None:
        self.broadcaster.unsubscribe(self.identifier)


class EncodedStreamBroadcaster:
    """Fan-out helper so multiple WebRTC tracks can tap encoded frames."""

    def __init__(self, loop: asyncio.AbstractEventLoop, max_queue: int = 8):
        self._loop = loop
        self._max_queue = max_queue
        self._lock = Lock()
        self._next_identifier = 0
        self._subscribers: Dict[int, asyncio.Queue[Optional[EncodedSample]]] = {}

    def subscribe(self) -> EncodedStreamSubscription:
        queue: asyncio.Queue[Optional[EncodedSample]] = asyncio.Queue(maxsize=self._max_queue)
        with self._lock:
            identifier = self._next_identifier
            self._next_identifier += 1
            self._subscribers[identifier] = queue
        return EncodedStreamSubscription(identifier, queue, self)

    def unsubscribe(self, identifier: int) -> None:
        with self._lock:
            self._subscribers.pop(identifier, None)

    def publish(self, sample: EncodedSample) -> None:
        if not self._subscribers:
            return

        def _dispatch() -> None:
            dead_keys = []
            for key, queue in list(self._subscribers.items()):
                try:
                    try:
                        while True:
                            queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        queue.put_nowait(sample)
                    except asyncio.QueueFull:
                        continue
                except RuntimeError:
                    dead_keys.append(key)
            if dead_keys:
                for key in dead_keys:
                    self._subscribers.pop(key, None)

        self._loop.call_soon_threadsafe(_dispatch)


class HardwareEncoderOutput(Output):
    """Picamera2 Output that forwards hardware encoded NAL units into asyncio queues."""

    def __init__(self, loop: asyncio.AbstractEventLoop, broadcaster: EncodedStreamBroadcaster, name: str):
        super().__init__()
        self._loop = loop
        self._broadcaster = broadcaster
        self._name = name
        self._base_pts: Optional[int] = None

    def outputframe(self, frame, keyframe: bool = True, timestamp: Optional[int] = None, packet=None, audio: bool = False):
        if audio or not self.recording:
            return

        # Picamera2 hands either bytes or an av.Packet. Normalise to bytes + pts.
        if packet is not None:
            data_bytes = bytes(packet)
            pts = packet.pts if packet.pts is not None else timestamp
            time_base = getattr(packet, "time_base", MICROSECOND_TIME_BASE) or MICROSECOND_TIME_BASE
        else:
            if frame is None:
                return
            data_bytes = bytes(frame)
            pts = timestamp
            time_base = MICROSECOND_TIME_BASE

        if pts is None:
            pts = int(time.monotonic_ns() // 1000)

        if self._base_pts is None:
            self._base_pts = pts
        relative_pts = max(0, int(pts - self._base_pts))

        sample = EncodedSample(
            data=data_bytes,
            pts=relative_pts,
            time_base=time_base if isinstance(time_base, Fraction) else MICROSECOND_TIME_BASE,
            keyframe=bool(keyframe),
        )
        if keyframe:
            print(f"[DEBUG] {self._name} keyframe {relative_pts}", flush=True)
        self._broadcaster.publish(sample)
        self.outputtimestamp(pts)


class HardwareVideoTrack(MediaStreamTrack):
    """MediaStreamTrack that yields encoded H.264 packets from a broadcaster subscription."""

    kind = "video"

    def __init__(self, broadcaster: EncodedStreamBroadcaster, label: str):
        super().__init__()
        self._subscription = broadcaster.subscribe()
        self.label = label

    async def recv(self) -> Packet:
        sample = await self._subscription.queue.get()
        if sample is None:
            raise MediaStreamError
        packet = Packet(sample.data)
        packet.pts = sample.pts
        packet.dts = sample.pts
        packet.time_base = sample.time_base
        try:
            packet.is_keyframe = sample.keyframe
        except Exception:
            pass
        return packet

    def stop(self) -> None:
        if self.readyState != "ended":
            super().stop()
        queue = self._subscription.queue
        try:
            while True:
                queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        except RuntimeError:
            pass
        self._subscription.close()


def _compute_crop(sensor_size: Tuple[int, int], profile: dict) -> Optional[Tuple[int, int, int, int]]:
    crop_values = profile.get("crop")
    if not crop_values or len(crop_values) != 2:
        return None

    crop_w = int(crop_values[0])
    crop_h = int(crop_values[1])
    if crop_w <= 0 or crop_h <= 0:
        return None

    sensor_w, sensor_h = sensor_size
    crop_w = min(crop_w, sensor_w)
    crop_h = min(crop_h, sensor_h)
    offset_x = int(profile.get("offset_x", 0))
    offset_y = int(profile.get("offset_y", 0))
    x = sensor_w // 2 - crop_w // 2 + offset_x
    y = sensor_h // 2 - crop_h // 2 + offset_y
    x = max(0, min(sensor_w - crop_w, x))
    y = max(0, min(sensor_h - crop_h, y))
    return (x, y, crop_w, crop_h)


class HardwareEncoderCamera:
    """Wraps a single Picamera2 instance with hardware H.264 encoding and fan-out broadcaster."""

    def __init__(
        self,
        index: int,
        profile: dict,
        framerate: int,
        loop: asyncio.AbstractEventLoop,
        description: str,
    ):
        self.index = index
        self.profile = copy.deepcopy(profile)
        self.loop = loop
        self.description = description
        self.broadcaster = EncodedStreamBroadcaster(loop)

        resolution = tuple(self.profile.get("resolution", [2304, 1296]))
        target_rate = float(self.profile.get("frame_rate", framerate))
        encoder_rate = Fraction(target_rate).limit_denominator(1000)
        if encoder_rate.denominator == 1:
            encoder_rate = Fraction(int(target_rate * 1000), 1000)

        self.camera = Picamera2(camera_num=index)
        transform = build_transform(
            int(self.profile.get("rotation", 0)),
            bool(self.profile.get("hflip", False)),
            bool(self.profile.get("vflip", False)),
        )

        config = self.camera.create_video_configuration(
            main={"size": resolution, "format": "YUV420"},
            transform=transform,
            controls={"FrameRate": target_rate},
            buffer_count=4,
        )
        self.camera.configure(config)

        crop_rect = _compute_crop(self.camera.sensor_resolution, self.profile)
        if crop_rect:
            try:
                self.camera.set_controls({"ScalerCrop": crop_rect})
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to apply crop for camera {index}: {exc}", file=sys.stderr)

        bitrate_mbps = self.profile.get("bitrate_mbps")
        bitrate_bps = None
        try:
            if bitrate_mbps is not None:
                bitrate_bps = int(float(bitrate_mbps) * 1_000_000)
        except Exception:  # noqa: BLE001
            bitrate_bps = None
        self.bitrate_bps = bitrate_bps

        if self.bitrate_bps:
            h264_codecs.MAX_BITRATE = max(h264_codecs.MAX_BITRATE, self.bitrate_bps)
            h264_codecs.DEFAULT_BITRATE = max(h264_codecs.DEFAULT_BITRATE, self.bitrate_bps)
            _ensure_h264_encoder_patch(self.bitrate_bps)

        iperiod_config = self.profile.get("gop_frames", max(1, int(target_rate)))
        try:
            iperiod = int(iperiod_config)
        except Exception:  # noqa: BLE001
            iperiod = max(1, int(target_rate))
        if iperiod <= 0:
            iperiod = max(1, int(target_rate))

        qp_value = self.profile.get("qp")
        qp: Optional[int]
        try:
            qp = int(qp_value) if qp_value is not None else None
        except Exception:  # noqa: BLE001
            qp = None
        repeat_headers = bool(self.profile.get("repeat_headers", True))
        profile_name = self.profile.get("h264_profile", "high")
        self.encoder = H264Encoder(
            bitrate=self.bitrate_bps,
            iperiod=iperiod,
            framerate=encoder_rate,
            profile=profile_name,
            qp=qp,
            repeat=repeat_headers,
        )
        self.output = HardwareEncoderOutput(loop, self.broadcaster, description)

        try:
            self.camera.start_recording(self.encoder, self.output)
        except Exception:
            try:
                self.output.stop()
            except Exception:
                pass
            try:
                self.camera.close()
            except Exception:
                pass
            raise
        self._apply_runtime_controls(target_rate)

    def _apply_runtime_controls(self, target_rate: float) -> None:
        controls: Dict[str, object] = {}
        controls["FrameRate"] = target_rate

        awb_enable = self.profile.get("awb_enable")
        gains = self.profile.get("colour_gains")
        awb_mode = resolve_awb_mode(self.profile.get("awb_mode"))
        if awb_enable is not None:
            controls["AwbEnable"] = bool(awb_enable)
            if awb_enable and awb_mode is not None:
                controls["AwbMode"] = awb_mode
        if awb_enable is not None and not bool(awb_enable) and gains and len(gains) == 2:
            controls["ColourGains"] = (float(gains[0]), float(gains[1]))

        exposure = self.profile.get("exposure_time")
        gain = self.profile.get("analogue_gain")
        if exposure is not None:
            controls["ExposureTime"] = int(exposure)
        if gain is not None:
            controls["AnalogueGain"] = float(gain)

        try:
            self.camera.set_controls(controls)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to apply controls for camera {self.index}: {exc}", file=sys.stderr)

    def create_track(self) -> HardwareVideoTrack:
        return HardwareVideoTrack(self.broadcaster, self.description)

    def stop(self) -> None:
        try:
            self.camera.stop_recording()
        except Exception:
            pass
        try:
            self.output.stop()
        except Exception:
            pass
        try:
            self.camera.close()
        except Exception:
            pass


class StereoHardwareCapture:
    """Owns the two hardware encoders and produces WebRTC-ready tracks."""

    def __init__(
        self,
        config_path: Path,
        framerate: int,
        loop: asyncio.AbstractEventLoop,
        resolution: Optional[Tuple[int, int]] = None,
    ):
        left_profile = copy.deepcopy(load_profile("cam0", config_path))
        right_profile = copy.deepcopy(load_profile("cam1", config_path))
        if resolution is not None:
            res_list = [int(resolution[0]), int(resolution[1])]
            for profile in (left_profile, right_profile):
                profile["resolution"] = res_list.copy()
                if list(profile.get("crop", res_list)) == res_list:
                    profile["crop"] = res_list.copy()
        left_profile["frame_rate"] = framerate
        right_profile["frame_rate"] = framerate

        self.left: HardwareEncoderCamera | None = None
        self.right: HardwareEncoderCamera | None = None

        try:
            self.left = HardwareEncoderCamera(0, left_profile, framerate, loop, description="left")
            self.right = HardwareEncoderCamera(1, right_profile, framerate, loop, description="right")
        except Exception:
            if self.left is not None:
                self.left.stop()
            if self.right is not None:
                self.right.stop()
            raise

    def create_tracks(self) -> Tuple[HardwareVideoTrack, HardwareVideoTrack]:
        return self.left.create_track(), self.right.create_track()

    def shutdown(self) -> None:
        if self.left:
            self.left.stop()
        if self.right:
            self.right.stop()


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
    def __init__(self, capture: StereoHardwareCapture, ca_cert: Path | None = None):
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

        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        try:
            await pc.setRemoteDescription(offer)
        except Exception as exc:  # noqa: BLE001
            await pc.close()
            self.pcs.discard(pc)
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"error": f"Invalid remote description: {exc}"}),
            )

        audio_track = SilenceAudioTrack()

        try:
            video_codecs = RTCRtpSender.getCapabilities("video").codecs  # type: ignore[attr-defined]
        except AttributeError:
            video_codecs = None

        h264_codecs = [c for c in video_codecs if c.mimeType.lower() == "video/h264"] if video_codecs else []

        if h264_codecs:
            print("[INFO] Preferring H.264 codecs for video stream", flush=True)
        else:
            print("[WARN] H.264 codec not advertised by client; using default preferences", flush=True)

        left_track, right_track = self.capture.create_tracks()
        stereo_tracks = [("left", left_track), ("right", right_track)]

        video_transceivers = [t for t in pc.getTransceivers() if t.kind == "video"]
        created_transceivers: list[tuple[str, RTCRtpSender, HardwareVideoTrack]] = []

        if len(video_transceivers) < len(stereo_tracks):
            print(
                f"[WARN] Client offered {len(video_transceivers)} video transceivers; expected "
                f"{len(stereo_tracks)}. Streaming available eyes only.",
                flush=True,
            )

        for (label, track), transceiver in zip(stereo_tracks, video_transceivers):
            try:
                transceiver.direction = "sendonly"
            except Exception:
                pass
            if h264_codecs:
                try:
                    transceiver.setCodecPreferences(h264_codecs)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Failed to set codec preferences for {label}: {exc}", file=sys.stderr)
            sender = transceiver.sender
            try:
                sender.replaceTrack(track)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Unable to attach {label} track: {exc}", file=sys.stderr)
                track.stop()
                await pc.close()
                self.pcs.discard(pc)
                return web.Response(
                    status=500,
                    content_type="application/json",
                    text=json.dumps({"error": f"Unable to attach {label} track."}),
                )
            created_transceivers.append((label, sender, track))

        # Stop any unused hardware tracks to release encoder resources.
        for label, track in stereo_tracks[len(video_transceivers) :]:
            print(f"[WARN] No remote slot for {label} track; stopping encoder.", flush=True)
            track.stop()

        audio_transceiver = next((t for t in pc.getTransceivers() if t.kind == "audio"), None)
        if audio_transceiver:
            try:
                audio_transceiver.direction = "sendonly"
            except Exception:
                pass
            if audio_transceiver.sender:
                try:
                    audio_transceiver.sender.replaceTrack(audio_track)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Failed to replace audio track: {exc}", file=sys.stderr)
        else:
            pc.addTrack(audio_track)

        async def _handle_state_change() -> None:
            if pc.connectionState in {"failed", "closed"}:
                for _, _, track in created_transceivers:
                    track.stop()
                audio_track.stop()
                await pc.close()
                self.pcs.discard(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            await _handle_state_change()

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
                    for report in stats.values():
                        if report.type == "outbound-rtp" and getattr(report, "kind", None) == "video":
                            print(
                                f"[STATS] mid={getattr(report, 'mid', '?')} "
                                f"bytes={getattr(report, 'bytesSent', 0)} "
                                f"frames={getattr(report, 'framesSent', 0)} "
                                f"keyFrames={getattr(report, 'keyFramesSent', 0)}",
                                flush=True,
                            )
                    await asyncio.sleep(1.0)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] stats logger stopped: {exc}", file=sys.stderr)

        asyncio.ensure_future(log_sender_stats())

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
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind (use 0.0.0.0 for all interfaces)")
    parser.add_argument("--port", type=int, default=8443, help="HTTP/WebSocket port for signaling")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to calibration profiles")
    parser.add_argument("--framerate", type=int, default=56, help="Capture framerate for both cameras")
    parser.add_argument(
        "--resolution",
        type=str,
        help="Override sensor resolution as WIDTHxHEIGHT (applies to both cameras)",
    )
    parser.add_argument(
        "--ice",
        nargs="*",
        default=["stun:stun.l.google.com:19302"],
        help="ICE servers (pass as stun:host:port or turn:user@pass:host:port)",
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
        "--cert-pass",
        default=None,
        help="Optional password for the TLS private key",
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

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        capture = StereoHardwareCapture(args.config, args.framerate, loop, resolution_override)
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

    semaphore = asyncio.Semaphore()

    async def shutdown(app_: web.Application) -> None:
        async with semaphore:
            await server.cleanup()
            capture.shutdown()

    app.on_shutdown.append(shutdown)

    def handle_signal() -> None:
        loop.create_task(app.shutdown())
        loop.create_task(app.cleanup())
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

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
                password=args.cert_pass,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to load TLS certificate/key: {exc}", file=sys.stderr)
            sys.exit(1)

    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
