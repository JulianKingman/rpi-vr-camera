#!/usr/bin/env python3
"""WebRTC/WebXR streamer that serves dual hardware-encoded stereo feeds."""

from __future__ import annotations

import argparse
import asyncio
import copy
import grp
import json
import os
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import yaml
from aiohttp import web
from aiortc import AudioStreamTrack, MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCRtpSender
from aiortc.mediastreams import MediaStreamError
from av.packet import Packet
from av import AudioFrame

# Try importing picamera2 for RPI (optional on Mac)
try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import Output

    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    Picamera2 = None  # type: ignore[assignment, misc]
    H264Encoder = None  # type: ignore[assignment, misc]
    Output = None  # type: ignore[assignment, misc]

from cam_utils import build_transform, resolve_awb_mode
from aiortc.codecs import h264 as h264_codecs

MICROSECOND_TIME_BASE = Fraction(1, 1_000_000)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "camera_profiles.yaml"
STATIC_DIR = REPO_ROOT / "web"
CERT_DIR = REPO_ROOT / "certs"

# Camera descriptions used in health-check diagnostics.
CAMERA_LABELS = {
    0: "Left, CSI port 0",
    1: "Right, CSI port 1",
}

BOOT_CONFIG_PATHS = (
    Path("/boot/firmware/config.txt"),
    Path("/boot/config.txt"),
)


# ---------------------------------------------------------------------------
# Pre-flight camera health checks
# ---------------------------------------------------------------------------

def _safe_char(char: str, fallback: str) -> str:
    """Return char if stdout can encode it, otherwise fallback."""
    try:
        char.encode(sys.stdout.encoding or "utf-8")
        return char
    except (UnicodeEncodeError, LookupError):
        return fallback

_OK = _safe_char("\u2713", "[OK]")
_FAIL = _safe_char("\u2717", "[FAIL]")
_ARROW = _safe_char("\u2192", "->")
_DASH = _safe_char("\u2014", "--")


def _print_ok(msg: str) -> None:
    print(f"  {_OK} {msg}", flush=True)


def _print_fail(msg: str) -> None:
    print(f"  {_FAIL} {msg}", flush=True)


def _print_hint(msg: str) -> None:
    print(f"    {_ARROW} {msg}", flush=True)


def _check_libcamera_available() -> bool:
    """Verify that libcamera can be reached (Python bindings + CLI tool)."""
    ok = True

    # Check Python bindings
    try:
        import libcamera  # noqa: F401
    except ImportError:
        _print_fail("libcamera Python bindings not found")
        _print_hint("Install with: sudo apt install -y python3-libcamera")
        ok = False

    # Check CLI tool (rpicam-hello on newer Pi OS, libcamera-hello on older)
    if shutil.which("rpicam-hello") is None and shutil.which("libcamera-hello") is None:
        _print_fail("rpicam-hello / libcamera-hello CLI tool not found on PATH")
        _print_hint("Install with: sudo apt install -y rpicam-apps (or libcamera-apps)")
        ok = False

    if ok:
        _print_ok("libcamera is available")

    return ok


def _check_video_group() -> bool:
    """Check that the current user belongs to the 'video' group."""
    try:
        video_gid = grp.getgrnam("video").gr_gid
    except KeyError:
        # No 'video' group on this system; skip the check.
        return True

    user_groups = os.getgroups()
    if video_gid in user_groups:
        _print_ok(f"User '{os.getenv('USER', 'unknown')}' is in the 'video' group")
        return True

    _print_fail(f"User '{os.getenv('USER', 'unknown')}' is NOT in the 'video' group")
    _print_hint("Add yourself with: sudo usermod -aG video $USER  (then log out and back in)")
    return False


def _check_boot_config() -> bool:
    """Look for a camera-related dtoverlay in the Raspberry Pi boot config."""
    config_path: Optional[Path] = None
    for candidate in BOOT_CONFIG_PATHS:
        if candidate.exists():
            config_path = candidate
            break

    if config_path is None:
        # Not a Raspberry Pi or config is elsewhere; skip silently.
        return True

    try:
        config_text = config_path.read_text()
    except PermissionError:
        _print_fail(f"Cannot read {config_path} (permission denied)")
        _print_hint(f"Run: sudo chmod +r {config_path}")
        return False

    # Look for camera-related overlays (dtoverlay=imx..., dtoverlay=ov..., camera_auto_detect, etc.)
    camera_indicators = [
        "camera_auto_detect=1",
        "dtoverlay=imx",
        "dtoverlay=ov",
        "start_x=1",
    ]
    found = False
    for indicator in camera_indicators:
        for line in config_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if indicator in stripped:
                found = True
                break
        if found:
            break

    if found:
        _print_ok(f"Camera overlay/auto-detect enabled in {config_path}")
    else:
        _print_fail(f"No camera overlay found in {config_path}")
        _print_hint(f"Ensure 'camera_auto_detect=1' or a camera dtoverlay is in {config_path}")
        _print_hint("Edit with: sudo nano /boot/firmware/config.txt  (then reboot)")

    return found


def _check_camera_devices() -> Tuple[bool, List[dict]]:
    """Use Picamera2.global_camera_info() to detect attached cameras."""
    if not PICAMERA2_AVAILABLE or Picamera2 is None:
        _print_fail("picamera2 is not installed")
        _print_hint("Install with: sudo apt install -y python3-picamera2")
        return False, []

    try:
        cam_info: List[dict] = Picamera2.global_camera_info()
    except Exception as exc:
        _print_fail(f"Picamera2 failed to enumerate cameras: {exc}")
        _print_hint("Check that libcamera is working: rpicam-hello --list-cameras")
        return False, []

    if not cam_info:
        _print_fail("No cameras detected by Picamera2")
        _print_hint("Check ribbon cables and ensure camera overlays are enabled")
        _print_hint("Run: rpicam-hello --list-cameras")
        return False, []

    return True, cam_info


def _check_camera_not_busy(index: int) -> bool:
    """Check if /dev/video* devices tied to a camera index might be held by another process.

    Returns True always (warning only) since PipeWire/WirePlumber commonly hold
    video devices open and picamera2 can usually still acquire the camera.
    """
    video_dev = Path(f"/dev/video{index}")
    if not video_dev.exists() or shutil.which("lsof") is None:
        return True

    try:
        result = subprocess.run(
            ["lsof", "-w", str(video_dev)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Extract process names holding the device
            procs = set()
            for line in result.stdout.strip().splitlines()[1:]:
                parts = line.split()
                if parts:
                    procs.add(parts[0])
            proc_list = ", ".join(sorted(procs))
            # PipeWire/WirePlumber commonly hold devices open; warn but don't fail
            print(f"  [WARN] /dev/video{index} is held by: {proc_list} (may be normal)", flush=True)
    except Exception:
        pass

    return True


def preflight_camera_check() -> bool:
    """Run all pre-flight camera health checks.

    Returns True if both cameras are ready. On failure, prints
    human-readable diagnostics and returns False.
    """
    print("[PREFLIGHT] Checking camera hardware...", flush=True)
    all_ok = True

    # 1. libcamera availability
    if not _check_libcamera_available():
        all_ok = False

    # 2. video group membership
    if not _check_video_group():
        all_ok = False

    # 3. Boot config overlay
    if not _check_boot_config():
        all_ok = False

    # 4. Enumerate cameras via Picamera2
    cameras_found, cam_info = _check_camera_devices()
    if not cameras_found:
        all_ok = False

    # 5. Verify we have at least 2 cameras (cam0 + cam1)
    if cameras_found:
        num_cameras = len(cam_info)
        if num_cameras < 2:
            all_ok = False
            _print_fail(f"Only {num_cameras} camera(s) detected; this stereo setup requires 2")
            for cam_idx in range(2):
                if cam_idx >= num_cameras:
                    label = CAMERA_LABELS.get(cam_idx, f"index {cam_idx}")
                    _print_fail(f"Camera {cam_idx} ({label}) not detected")
                    _print_hint("Check that the ribbon cable is fully seated")
                    _print_hint("Verify /boot/firmware/config.txt has the camera overlay enabled")
                    _print_hint(f"Run 'rpicam-hello --camera {cam_idx}' to test")
        else:
            for cam_idx in range(2):
                info = cam_info[cam_idx]
                model = info.get("Model", "unknown")
                label = CAMERA_LABELS.get(cam_idx, f"index {cam_idx}")
                _print_ok(f"Camera {cam_idx} ({label}) detected: {model}")

    # 6. Check that cameras are not held by other processes
    for cam_idx in range(2):
        if not _check_camera_not_busy(cam_idx):
            all_ok = False

    if all_ok:
        print("[PREFLIGHT] All camera checks passed.", flush=True)
    else:
        print("[PREFLIGHT] One or more camera checks failed. See messages above.", flush=True)

    return all_ok


# ---------------------------------------------------------------------------
# Network / startup helpers
# ---------------------------------------------------------------------------

def _get_local_ips() -> List[str]:
    """Return a list of non-loopback IPv4 addresses for this host."""
    ips: List[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            if not addr.startswith("127."):
                ips.append(addr)
    except Exception:  # noqa: BLE001
        pass
    # Fallback: connect to a public IP to discover the default route address
    if not ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            addr = s.getsockname()[0]
            s.close()
            if not addr.startswith("127."):
                ips.append(addr)
        except Exception:  # noqa: BLE001
            pass
    return sorted(set(ips))


def _get_hostname() -> str:
    """Return the machine hostname (e.g. 'rpi-vr-camera')."""
    return socket.gethostname()


def _render_qr_ascii(url: str) -> str:
    """Render a QR code as a compact ASCII block using the qrcode library.

    Falls back to a simple text notice if the library is unavailable.
    """
    try:
        import qrcode  # type: ignore[import-untyped]
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=1,
        )
        qr.add_data(url)
        qr.make(fit=True)
        matrix = qr.get_matrix()
        rows = len(matrix)
        cols = len(matrix[0]) if rows else 0
        lines: List[str] = []
        BOTH_BLACK = _safe_char("\u2588", "#")
        TOP_BLACK = _safe_char("\u2580", "^")
        BOT_BLACK = _safe_char("\u2584", "v")
        BOTH_WHITE = " "
        for r in range(0, rows, 2):
            line_chars: List[str] = []
            for c in range(cols):
                top = matrix[r][c]
                bot = matrix[r + 1][c] if r + 1 < rows else False
                if top and bot:
                    line_chars.append(BOTH_BLACK)
                elif top:
                    line_chars.append(TOP_BLACK)
                elif bot:
                    line_chars.append(BOT_BLACK)
                else:
                    line_chars.append(BOTH_WHITE)
            lines.append("  " + "".join(line_chars))
        return "\n".join(lines)
    except ImportError:
        return "  (install 'qrcode' package for QR code display: pip install qrcode)"
    except Exception:  # noqa: BLE001
        return "  (QR code generation failed)"


def print_camera_init(index: int, description: str, resolution: Tuple[int, int], framerate: float) -> None:
    """Print a camera initialization confirmation line."""
    w, h = resolution
    fps = int(framerate) if framerate == int(framerate) else framerate
    side = "Left" if description == "left" else "Right"
    print(f"{_OK} Camera {index} ({side}) initialized {_DASH} {w}x{h} @ {fps}fps", flush=True)


def print_tls_loaded(cert_path: Path, key_path: Path) -> None:
    """Print TLS certificate confirmation."""
    cert_dir = cert_path.parent
    print(f"{_OK} TLS certificates loaded from {cert_dir}/", flush=True)


def print_ready_banner(
    host: str,
    port: int,
    scheme: str,
    cert_dir: Optional[Path] = None,
) -> None:
    """Print the full ready banner with URLs, QR code, and status."""
    ips = _get_local_ips()
    hostname = _get_hostname()

    urls: List[str] = []
    for ip in ips:
        urls.append(f"{scheme}://{ip}:{port}/")

    mdns_url = f"{scheme}://{hostname}.local:{port}/"
    primary_url = urls[0] if urls else mdns_url

    separator = "=" * 58
    print("", flush=True)
    print(separator, flush=True)
    print("  Ready for connections", flush=True)
    print(separator, flush=True)
    print("", flush=True)

    if urls:
        print("  Network URLs:", flush=True)
        for url in urls:
            print(f"    {url}", flush=True)
    print("  mDNS URL:", flush=True)
    print(f"    {mdns_url}", flush=True)

    print("", flush=True)
    print(f"  QR code ({primary_url}):", flush=True)
    qr_output = _render_qr_ascii(primary_url)
    print(qr_output, flush=True)

    print("", flush=True)
    print(separator, flush=True)
    print("[INFO] Press Ctrl+C to stop the server.", flush=True)


# ---------------------------------------------------------------------------
# Settings classification
# ---------------------------------------------------------------------------

# Settings that can be applied at runtime without restarting cameras
RUNTIME_SETTINGS = {"awb_mode", "colour_gains", "bitrate_mbps", "awb_enable"}
# Settings that require a full server restart to take effect
RESTART_SETTINGS = {"resolution", "rotation", "hflip", "vflip", "crop", "offset_x", "offset_y"}


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


class SharedEpoch:
    """Shared PTS origin so both cameras' timestamps are aligned."""
    def __init__(self):
        self._epoch: Optional[int] = None
        self._lock = Lock()

    def relativize(self, pts: int) -> int:
        with self._lock:
            if self._epoch is None:
                self._epoch = pts
            return max(0, pts - self._epoch)


@dataclass
class EncodedStreamSubscription:
    identifier: int
    queue: asyncio.Queue[Optional[EncodedSample]]
    broadcaster: "EncodedStreamBroadcaster"

    def close(self) -> None:
        self.broadcaster.unsubscribe(self.identifier)


class EncodedStreamBroadcaster:
    """Fan-out helper so multiple WebRTC tracks can tap encoded frames."""

    def __init__(self, loop: asyncio.AbstractEventLoop, max_queue: int = 3):
        self._loop = loop
        self._max_queue = max_queue
        self._lock = Lock()
        self._next_identifier = 0
        self._subscribers: Dict[int, asyncio.Queue[Optional[EncodedSample]]] = {}
        self._drop_count = 0

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
                        queue.put_nowait(sample)
                    except asyncio.QueueFull:
                        try:
                            queue.get_nowait()  # drop oldest
                        except asyncio.QueueEmpty:
                            pass
                        self._drop_count += 1
                        print(f"[WARN] Broadcaster dropped frame (total: {self._drop_count})", flush=True)
                        try:
                            queue.put_nowait(sample)
                        except asyncio.QueueFull:
                            pass
                except RuntimeError:
                    dead_keys.append(key)
            if dead_keys:
                for key in dead_keys:
                    self._subscribers.pop(key, None)

        self._loop.call_soon_threadsafe(_dispatch)


class HardwareEncoderOutput(Output if Output is not None else object):
    """Picamera2 Output that forwards hardware encoded NAL units into asyncio queues."""

    def __init__(self, loop: asyncio.AbstractEventLoop, broadcaster: EncodedStreamBroadcaster, name: str, shared_epoch: SharedEpoch):
        if Output is not None:
            super().__init__()
        self._loop = loop
        self._broadcaster = broadcaster
        self._name = name
        self._shared_epoch = shared_epoch

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
            print(f"[ERROR] {self._name}: frame has no timestamp, skipping", file=sys.stderr, flush=True)
            return

        relative_pts = self._shared_epoch.relativize(pts)

        sample = EncodedSample(
            data=data_bytes,
            pts=relative_pts,
            time_base=time_base if isinstance(time_base, Fraction) else MICROSECOND_TIME_BASE,
            keyframe=bool(keyframe),
        )
        if keyframe:
            print(f"[DEBUG] {self._name} keyframe {relative_pts}", flush=True)
        self._broadcaster.publish(sample)
        if Output is not None and hasattr(self, 'outputtimestamp'):
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
        shared_epoch: SharedEpoch,
    ):
        if not PICAMERA2_AVAILABLE or Picamera2 is None:
            raise RuntimeError(
                "HardwareEncoderCamera requires picamera2 (RPI only). "
                "This script uses hardware H.264 encoding. For Mac testing, use test patterns or modify the script."
            )
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
        if not PICAMERA2_AVAILABLE or H264Encoder is None:
            raise RuntimeError(
                "Hardware encoding requires picamera2 (RPI only). "
                "Use camera adapter mode (CAMERA_MODE=test or CAMERA_MODE=opencv) for Mac testing."
            )

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
        self.output = HardwareEncoderOutput(loop, self.broadcaster, description, shared_epoch)

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

        # Print camera initialization confirmation
        print_camera_init(index, description, resolution, target_rate)

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

    def apply_runtime_settings(self, updates: dict) -> list[str]:
        """Apply runtime-changeable settings. Returns list of changes applied."""
        applied = []
        cam_controls: Dict[str, object] = {}

        if "awb_mode" in updates:
            mode = resolve_awb_mode(updates["awb_mode"])
            if mode is not None:
                self.profile["awb_mode"] = updates["awb_mode"]
                cam_controls["AwbMode"] = mode
                applied.append(f"awb_mode={updates['awb_mode']}")

        if "awb_enable" in updates:
            val = bool(updates["awb_enable"])
            self.profile["awb_enable"] = val
            cam_controls["AwbEnable"] = val
            applied.append(f"awb_enable={val}")

        if "colour_gains" in updates:
            gains = updates["colour_gains"]
            if isinstance(gains, (list, tuple)) and len(gains) == 2:
                self.profile["colour_gains"] = [float(gains[0]), float(gains[1])]
                cam_controls["ColourGains"] = (float(gains[0]), float(gains[1]))
                applied.append(f"colour_gains=[{gains[0]}, {gains[1]}]")

        if "bitrate_mbps" in updates:
            new_bitrate_mbps = float(updates["bitrate_mbps"])
            new_bitrate_bps = int(new_bitrate_mbps * 1_000_000)
            self.profile["bitrate_mbps"] = new_bitrate_mbps
            self.bitrate_bps = new_bitrate_bps
            h264_codecs.MAX_BITRATE = max(h264_codecs.MAX_BITRATE, new_bitrate_bps)
            h264_codecs.DEFAULT_BITRATE = max(h264_codecs.DEFAULT_BITRATE, new_bitrate_bps)
            try:
                self.encoder.bitrate = new_bitrate_bps
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to set encoder bitrate for camera {self.index}: {exc}", file=sys.stderr)
            applied.append(f"bitrate_mbps={new_bitrate_mbps}")

        if cam_controls:
            try:
                self.camera.set_controls(cam_controls)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to apply runtime controls for camera {self.index}: {exc}", file=sys.stderr)

        return applied

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

        shared_epoch = SharedEpoch()
        try:
            self.left = HardwareEncoderCamera(0, left_profile, framerate, loop, description="left", shared_epoch=shared_epoch)
            self.right = HardwareEncoderCamera(1, right_profile, framerate, loop, description="right", shared_epoch=shared_epoch)
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
    def __init__(self, capture: StereoHardwareCapture, ca_cert: Path | None = None, config_path: Path = CONFIG_PATH):
        self.capture = capture
        self.ca_cert = ca_cert
        self.config_path = config_path
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
                                drop_count = self.capture.left.broadcaster._drop_count + self.capture.right.broadcaster._drop_count
                                try:
                                    self._stats_channel.send(json.dumps({
                                        "type": "server_stats",
                                        "mid": mid,
                                        "bytesSent": bytes_sent,
                                        "framesSent": frames_sent,
                                        "keyFramesSent": key_frames_sent,
                                        "droppedFrames": drop_count,
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

    async def get_settings(self, _request: web.Request) -> web.Response:
        """GET /api/settings -- return current camera profile settings as JSON."""
        try:
            data = yaml.safe_load(self.config_path.read_text())
            profiles = data.get("profiles", {})
        except Exception as exc:  # noqa: BLE001
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to read config: {exc}"}),
            )
        result: Dict[str, dict] = {}
        for cam_name in ("cam0", "cam1"):
            profile = profiles.get(cam_name, {})
            result[cam_name] = {
                "description": profile.get("description", ""),
                "resolution": profile.get("resolution", [2304, 1296]),
                "rotation": profile.get("rotation", 0),
                "hflip": profile.get("hflip", False),
                "vflip": profile.get("vflip", False),
                "crop": profile.get("crop", [1000, 1000]),
                "offset_x": profile.get("offset_x", 0),
                "offset_y": profile.get("offset_y", 0),
                "awb_enable": profile.get("awb_enable", True),
                "awb_mode": profile.get("awb_mode", "auto"),
                "colour_gains": profile.get("colour_gains", [1.0, 1.0]),
                "bitrate_mbps": profile.get("bitrate_mbps", 10.0),
            }
        return web.Response(
            content_type="application/json",
            text=json.dumps(result),
        )

    async def post_settings(self, request: web.Request) -> web.Response:
        """POST /api/settings -- apply and/or save updated camera settings."""
        try:
            updates = await request.json()
        except Exception as exc:  # noqa: BLE001
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"error": f"Invalid JSON: {exc}"}),
            )

        if not isinstance(updates, dict):
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"error": "Expected JSON object with cam0/cam1 keys"}),
            )

        try:
            data = yaml.safe_load(self.config_path.read_text())
        except Exception as exc:  # noqa: BLE001
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to read config: {exc}"}),
            )

        profiles = data.get("profiles", {})
        runtime_applied: list[str] = []
        restart_needed = False
        cam_map = {"cam0": self.capture.left, "cam1": self.capture.right}

        for cam_name in ("cam0", "cam1"):
            cam_updates = updates.get(cam_name)
            if not cam_updates or not isinstance(cam_updates, dict):
                continue

            cam = cam_map.get(cam_name)
            profile = profiles.get(cam_name, {})

            runtime_changes: dict = {}
            restart_changes: dict = {}
            for key, value in cam_updates.items():
                if key in RUNTIME_SETTINGS:
                    runtime_changes[key] = value
                elif key in RESTART_SETTINGS:
                    restart_changes[key] = value

            if runtime_changes and cam:
                applied = cam.apply_runtime_settings(runtime_changes)
                runtime_applied.extend([f"{cam_name}.{a}" for a in applied])

            for key, value in cam_updates.items():
                if key in RUNTIME_SETTINGS or key in RESTART_SETTINGS:
                    profile[key] = value

            if restart_changes:
                restart_needed = True

            profiles[cam_name] = profile

        data["profiles"] = profiles
        try:
            self.config_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        except Exception as exc:  # noqa: BLE001
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to write config: {exc}"}),
            )

        result = {
            "ok": True,
            "runtime_applied": runtime_applied,
            "restart_needed": restart_needed,
        }
        if restart_needed:
            result["message"] = "Settings saved. Server restart required for resolution/rotation/crop changes."
        else:
            result["message"] = "Settings applied."

        print(f"[INFO] Settings updated: runtime={runtime_applied}, restart_needed={restart_needed}", flush=True)

        return web.Response(
            content_type="application/json",
            text=json.dumps(result),
        )


def _first_existing(patterns: Tuple[str, ...]) -> Optional[Path]:
    for pattern in patterns:
        for candidate in sorted(CERT_DIR.glob(pattern)):
            if candidate.is_file():
                return candidate
    return None


def discover_tls_assets() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Return (cert, key, ca_cert) from the certs/ directory if available."""
    cert = _first_existing(("*-server.crt", "*_server.crt", "server.crt"))
    key = _first_existing(("*-server.key", "*_server.key", "server.key"))
    ca = _first_existing(("*-ca.crt", "*_ca.crt", "ca.crt"))
    return cert, key, ca


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind (use 0.0.0.0 for all interfaces)")
    parser.add_argument("--port", type=int, default=8443, help="HTTP/WebSocket port for signaling")
    parser.add_argument(
        "--http-port",
        type=int,
        default=0,
        help="Optional HTTP fallback port for downloading certificates (0 disables; defaults to 8080 when TLS is on)",
    )
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
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        default=False,
        help="Skip the camera pre-flight health checks on startup",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    cert_path: Path | None = args.cert.expanduser() if args.cert else None
    key_path: Path | None = args.key.expanduser() if args.key else None
    ca_cert_path: Path | None = args.ca_cert.expanduser() if args.ca_cert else None
    resolution_override: Optional[Tuple[int, int]] = None

    if args.resolution:
        try:
            resolution_override = parse_resolution(args.resolution)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(1)

    auto_cert, auto_key, auto_ca = discover_tls_assets()

    if cert_path and not key_path:
        print("[ERROR] --cert provided without --key", file=sys.stderr)
        sys.exit(1)
    if key_path and not cert_path:
        print("[ERROR] --key provided without --cert", file=sys.stderr)
        sys.exit(1)

    if cert_path is None and key_path is None and auto_cert and auto_key:
        cert_path, key_path = auto_cert, auto_key
        print(f"[INFO] Using TLS certificate: {cert_path}", flush=True)
        print(f"[INFO] Using TLS private key: {key_path}", flush=True)

    if ca_cert_path is None and auto_ca:
        ca_cert_path = auto_ca
        print(f"[INFO] Exposing CA certificate at /ca.crt: {ca_cert_path}", flush=True)

    for label, path in (("TLS certificate", cert_path), ("TLS private key", key_path), ("CA certificate", ca_cert_path)):
        if path and not path.exists():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # --- Pre-flight camera health checks ---
    if args.skip_preflight:
        print("[INFO] Skipping camera pre-flight checks (--skip-preflight).", flush=True)
    elif PICAMERA2_AVAILABLE:
        if not preflight_camera_check():
            print(
                "\n[ERROR] Camera pre-flight checks failed. "
                "Resolve the issues above before starting the server.",
                file=sys.stderr,
            )
            sys.exit(2)
    else:
        print(
            "[WARN] picamera2 not available; skipping camera pre-flight checks "
            "(hardware encoding will not work without it).",
            flush=True,
        )

    exit_code = asyncio.run(
        run_streamer(
            args=args,
            cert_path=cert_path,
            key_path=key_path,
            ca_cert_path=ca_cert_path,
            resolution_override=resolution_override,
        )
    )
    sys.exit(exit_code)


async def run_streamer(
    args: argparse.Namespace,
    cert_path: Optional[Path],
    key_path: Optional[Path],
    ca_cert_path: Optional[Path],
    resolution_override: Optional[Tuple[int, int]],
) -> int:
    loop = asyncio.get_running_loop()
    try:
        capture = StereoHardwareCapture(args.config, args.framerate, loop, resolution_override)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Unable to start capture: {exc}", file=sys.stderr)
        return 1

    server = WebRTCServer(capture, ca_cert_path, config_path=args.config)
    app = web.Application()
    app["rtc_server"] = server
    app.router.add_get("/", server.index)
    app.router.add_post("/offer", server.offer)
    app.router.add_get("/api/settings", server.get_settings)
    app.router.add_post("/api/settings", server.post_settings)
    app.router.add_static("/static/", STATIC_DIR, show_index=True)
    if ca_cert_path:
        app.router.add_get("/ca.crt", server.serve_ca_certificate)

    ssl_context = None
    if cert_path and key_path:
        try:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(
                certfile=str(cert_path),
                keyfile=str(key_path),
                password=args.cert_pass,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to load TLS certificate/key: {exc}", file=sys.stderr)
            return 1
        print_tls_loaded(cert_path, key_path)
    elif cert_path or key_path:
        print("[ERROR] --cert and --key must be provided together for HTTPS", file=sys.stderr)
        return 1

    fallback_http_port = args.http_port
    if ssl_context and fallback_http_port == 0:
        fallback_http_port = 8080

    if ssl_context and fallback_http_port and fallback_http_port == args.port:
        print("[ERROR] HTTP fallback port cannot match HTTPS port", file=sys.stderr)
        capture.shutdown()
        return 1

    runner = web.AppRunner(app)
    try:
        await runner.setup()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to start web server: {exc}", file=sys.stderr)
        capture.shutdown()
        return 1

    async def start_site(port: int, ssl_ctx: Optional[ssl.SSLContext]) -> None:
        scheme = "https" if ssl_ctx else "http"
        bind_host = args.host
        if bind_host in {"0.0.0.0", "::"}:
            bind_host = "0.0.0.0"
        site = web.TCPSite(runner, host=args.host, port=port, ssl_context=ssl_ctx)
        await site.start()
        print(f"[INFO] Serving {scheme.upper()} on {scheme}://{bind_host}:{port}/", flush=True)

    try:
        if ssl_context:
            await start_site(args.port, ssl_context)
            if fallback_http_port:
                await start_site(fallback_http_port, None)
                print("[INFO] HTTP fallback exposes /ca.crt before trusting HTTPS.", flush=True)
        else:
            await start_site(args.port, None)
            if fallback_http_port and fallback_http_port != args.port:
                await start_site(fallback_http_port, None)

        stop_event = asyncio.Event()
        signals = (signal.SIGINT, signal.SIGTERM)
        for sig in signals:
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass

        scheme = "https" if ssl_context else "http"
        print_ready_banner(
            host=args.host,
            port=args.port,
            scheme=scheme,
            cert_dir=cert_path.parent if cert_path else None,
        )
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
                pass
        await server.cleanup()
        await runner.cleanup()
        capture.shutdown()

    return 0


if __name__ == "__main__":
    main()
