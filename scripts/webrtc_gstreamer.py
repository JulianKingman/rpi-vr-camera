#!/usr/bin/env python3
"""Stereo WebRTC streaming using a GStreamer pipeline (design option 3)."""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import logging
import signal
import ssl
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import contextlib
import os
import subprocess

import yaml
from aiohttp import web

try:
    import gi
except Exception:  # noqa: BLE001
    print(
        "[ERROR] Python GObject introspection bindings (python3-gi) are missing. "
        "Install them via 'sudo apt install python3-gi'.",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    gi.require_version("Gst", "1.0")
    gi.require_version("GstWebRTC", "1.0")
    gi.require_version("GstSdp", "1.0")
    from gi.repository import Gst, GstSdp, GstWebRTC  # type: ignore[attr-defined]
except (ImportError, ValueError) as exc:
    print(
        "[ERROR] GStreamer introspection modules not available. "
        "Install 'gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 "
        "gir1.2-gst-plugins-bad-1.0 gstreamer1.0-gl' or run 'make system-deps'.",
        file=sys.stderr,
    )
    sys.exit(1)

Gst.init(None)

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
    width_str, height_str = value.lower().split("x", maxsplit=1)
    width = int(width_str)
    height = int(height_str)
    if width <= 0 or height <= 0:
        raise ValueError("Resolution values must be positive integers.")
    return width, height


def _create_queue(name: str) -> Gst.Element:
    queue = Gst.ElementFactory.make("queue", name)
    if queue is None:
        raise RuntimeError("Unable to create queue element")
    queue.set_property("max-size-buffers", 1)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)
    queue.set_property("leaky", 2)  # drop oldest when downstream is slow
    return queue


def _has_property(element: Gst.Element, prop_name: str) -> bool:
    return any(spec.name == prop_name for spec in element.list_properties())


def _attempt_camera_open() -> bool:
    env = os.environ.copy()
    env.setdefault("LIBCAMERA_LOG_LEVELS", "3")
    script = (
        "from picamera2 import Picamera2\n"
        "picam = Picamera2()\n"
        "picam.close()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        if message:
            logging.debug("Camera access check failed: %s", message)
    return result.returncode == 0


def _wait_for_camera_release(timeout: float = 3.0, interval: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _attempt_camera_open():
            return True
        time.sleep(interval)
    return _attempt_camera_open()


def _list_camera_processes() -> list[tuple[int, str]]:
    processes: list[tuple[int, str]] = []
    try:
        result = subprocess.run(
            ["bash", "-lc", "lsof -w /dev/video*"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return processes
    if result.returncode != 0:
        return processes
    lines = [line.strip() for line in result.stdout.splitlines()[1:]]
    for line in lines:
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        cmd = parts[0]
        processes.append((pid, cmd))
    if processes:
        logging.debug("Camera devices currently held by: %s", processes)
    return processes


def _kill_process_with_escalation(pid: int, sig: int = signal.SIGTERM) -> bool:
    """Kill process with automatic privilege escalation if needed."""
    # Try direct kill first
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return True  # Process already gone
    except PermissionError:
        pass  # Fall through to subprocess-based kill
    # Fallback: use kill command (may work for same-user processes)
    result = subprocess.run(
        ["kill", f"-{sig}", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    # Final fallback: try sudo kill
    sudo_result = subprocess.run(
        ["sudo", "kill", f"-{sig}", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    if sudo_result.returncode == 0:
        return True
    return False


def _prompt_kill_process(pid: int, cmd: str) -> bool:
    prompt = f"Terminate PID {pid} ({cmd}) holding camera device? [y/N]: "
    response: str | None = None
    try:
        if sys.stdin and sys.stdin.isatty():
            response = input(prompt)
        else:
            with open("/dev/tty", "r", encoding="utf-8", errors="ignore") as tty_in, open(
                "/dev/tty", "w", encoding="utf-8", errors="ignore"
            ) as tty_out:
                print(prompt, end="", flush=True, file=tty_out)
                response = tty_in.readline()
    except EOFError:
        return False
    except OSError:
        return False
    if response is None:
        return False
    if response.strip().lower() not in {"y", "yes"}:
        return False
    if not _kill_process_with_escalation(pid, signal.SIGTERM):
        logging.error("Failed to kill PID %s (%s) with SIGTERM", pid, cmd)
        return False
    # Wait briefly, then try SIGKILL if still alive
    time.sleep(0.5)
    try:
        os.kill(pid, 0)  # Check if process still exists
        logging.warning("PID %s still alive after SIGTERM; sending SIGKILL", pid)
        if not _kill_process_with_escalation(pid, signal.SIGKILL):
            logging.error("Failed to kill PID %s (%s) with SIGKILL", pid, cmd)
            return False
    except ProcessLookupError:
        pass  # Process is gone
    except PermissionError:
        # Try once more with SIGKILL via escalation
        if not _kill_process_with_escalation(pid, signal.SIGKILL):
            logging.error("Failed to kill PID %s (%s) with SIGKILL after escalation", pid, cmd)
            return False
    return True


def _kill_process(pid: int, cmd: str) -> bool:
    if not _kill_process_with_escalation(pid, signal.SIGTERM):
        logging.warning("Failed to kill PID %s (%s) with SIGTERM, trying SIGKILL", pid, cmd)
        if not _kill_process_with_escalation(pid, signal.SIGKILL):
            logging.error("Failed to kill PID %s (%s) even with SIGKILL", pid, cmd)
            return False
        return True
    # Wait briefly, then verify process is gone or send SIGKILL
    time.sleep(0.5)
    try:
        os.kill(pid, 0)  # Check if process still exists
        logging.debug("PID %s still alive after SIGTERM; sending SIGKILL", pid)
        if not _kill_process_with_escalation(pid, signal.SIGKILL):
            return False
    except (ProcessLookupError, PermissionError):
        pass  # Process is gone or we can't check it
    return True


def _stop_conflicting_services(kill_mode: str = "prompt") -> list[str]:
    units = [
        "pipewire-pulse.socket",
        "pipewire.socket",
        "pipewire-pulse.service",
        "pipewire.service",
        "wireplumber.service",
    ]
    stopped: list[str] = []
    for unit in units:
        try:
            completed = subprocess.run(
                ["systemctl", "--user", "stop", unit],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                stopped.append(unit)
        except FileNotFoundError:
            break
    for pattern in ("pipewire-pulse", "pipewire", "wireplumber"):
        subprocess.run(
            ["pkill", "-u", str(os.getuid()), pattern],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if kill_mode in {"prompt", "auto"}:
        for pid, cmd in _list_camera_processes():
            if kill_mode == "prompt":
                result = _prompt_kill_process(pid, cmd)
            else:
                result = _kill_process(pid, cmd)
            if result:
                logging.info("Terminated PID %s (%s)", pid, cmd)
    return stopped


def _restart_services(units: list[str]) -> None:
    for unit in units:
        try:
            subprocess.run(
                ["systemctl", "--user", "start", unit],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            break


def _ensure_camera_ready(stop_services: bool, kill_mode: str) -> list[str]:
    if _attempt_camera_open():
        return []
    if not stop_services:
        raise RuntimeError(
            "Cameras appear busy (PipeWire/WirePlumber or another capture tool is running). "
            "Stop those services or rerun without --keep-pipewire (or use --auto-kill-conflicts)."
        )
    logging.warning(
        "Cameras busy; attempting to stop PipeWire/WirePlumber to claim the sensors."
    )
    stopped = _stop_conflicting_services(kill_mode)
    if not _wait_for_camera_release():
        raise RuntimeError(
            "Unable to acquire the cameras even after stopping PipeWire/WirePlumber. "
            "Release other camera applications and retry."
        )
    return stopped


@dataclass
class BranchGeometry:
    width: int
    height: int
    crop_left: int
    crop_right: int
    crop_top: int
    crop_bottom: int


def _compute_geometry(profile: dict) -> BranchGeometry:
    res = profile.get("resolution") or [2304, 1296]
    res_w, res_h = int(res[0]), int(res[1])

    crop_values = profile.get("crop") or [res_w, res_h]
    crop_w = int(min(res_w, max(1, crop_values[0])))
    crop_h = int(min(res_h, max(1, crop_values[1])))

    offset_x = int(profile.get("offset_x", 0))
    offset_y = int(profile.get("offset_y", 0))

    left = max(0, (res_w - crop_w) // 2 - offset_x)
    right = max(0, res_w - crop_w - left)
    top = max(0, (res_h - crop_h) // 2 - offset_y)
    bottom = max(0, res_h - crop_h - top)
    return BranchGeometry(
        width=crop_w,
        height=crop_h,
        crop_left=left,
        crop_right=right,
        crop_top=top,
        crop_bottom=bottom,
    )


class GStreamerStereoPipeline:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        config_path: Path,
        left_profile_name: str,
        right_profile_name: str,
        framerate: int,
        ice_server: Optional[str],
        bitrate_mbps: Optional[float],
    ):
        self.loop = loop
        self.config_path = config_path
        self.left_profile_name = left_profile_name
        self.right_profile_name = right_profile_name
        self.framerate = framerate
        self.ice_server = ice_server
        self.default_bitrate_mbps = bitrate_mbps

        self.pipeline = Gst.Pipeline.new("stereo-gst")
        if self.pipeline is None:
            raise RuntimeError("Failed to create GStreamer pipeline.")

        self.webrtcbin: Gst.Element | None = None
        self.encoder: Gst.Element | None = None
        self.bus_task: Optional[asyncio.Task[None]] = None
        self.effective_bitrate_mbps: Optional[float] = None
        self.camera_names: list[str] = self._probe_camera_ids()

        self._session_active = False
        self._answer_future: Optional[asyncio.Future[str]] = None
        self._gather_future: Optional[asyncio.Future[None]] = None
        self._shutdown_event = asyncio.Event()

        self._build_pipeline()

    def _build_pipeline(self) -> None:
        left_profile = load_profile(self.left_profile_name, self.config_path)
        right_profile = load_profile(self.right_profile_name, self.config_path)

        left_geom = _compute_geometry(left_profile)
        right_geom = _compute_geometry(right_profile)
        composite_height = min(left_geom.height, right_geom.height)
        composite_width = left_geom.width + right_geom.width

        effective_bitrate_mbps = self.default_bitrate_mbps
        if effective_bitrate_mbps is None:
            def _rate(profile: dict) -> float:
                try:
                    value = profile.get("bitrate_mbps")
                    return float(value) if value is not None else 0.0
                except Exception:  # noqa: BLE001
                    return 0.0

            combined = _rate(left_profile) + _rate(right_profile)
            effective_bitrate_mbps = combined if combined > 0 else None
        self.effective_bitrate_mbps = effective_bitrate_mbps

        use_gl = Gst.ElementFactory.find("glvideomixer") is not None
        mixer_factory = "glvideomixer" if use_gl else "compositor"

        mixer = Gst.ElementFactory.make(mixer_factory, "stitcher")
        if mixer is None:
            raise RuntimeError(f"Missing GStreamer element '{mixer_factory}'.")
        if use_gl:
            mixer.set_property("latency", 0)
        else:
            mixer.set_property("background", 1)  # black
            mixer.set_property("latency", 0)
        self.pipeline.add(mixer)

        left_last = self._build_camera_branch(
            name="left",
            camera_id=0,
            profile=left_profile,
            geometry=left_geom,
            use_gl=use_gl,
        )
        right_last = self._build_camera_branch(
            name="right",
            camera_id=1,
            profile=right_profile,
            geometry=right_geom,
            use_gl=use_gl,
        )

        # Link branch outputs to mixer sink pads
        for last_element, geom, xpos, sink_idx in (
            (left_last, left_geom, 0, 0),
            (right_last, right_geom, left_geom.width, 1),
        ):
            src_pad = last_element.get_static_pad("src")
            if src_pad is None:
                raise RuntimeError(f"{last_element.get_name()} missing src pad.")
            sink_pad = mixer.request_pad_simple(f"sink_{sink_idx}")
            if sink_pad is None:
                raise RuntimeError("Failed to obtain mixer sink pad.")
            sink_pad.set_property("xpos", int(xpos))
            sink_pad.set_property("ypos", 0)
            sink_pad.set_property("width", int(geom.width))
            sink_pad.set_property("height", int(geom.height))
            sink_pad.set_property("alpha", 1.0)
            if src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError("Unable to link branch into mixer.")

        # Mixer output path
        post_queue = _create_queue("stitcher_queue")
        self.pipeline.add(post_queue)
        if not mixer.link(post_queue):
            raise RuntimeError("Failed to link mixer to queue.")

        elements_chain: list[Gst.Element] = [post_queue]

        if use_gl:
            gldownload = Gst.ElementFactory.make("gldownload", "stitcher_download")
            if gldownload is None:
                raise RuntimeError("Missing gldownload element.")
            elements_chain.append(gldownload)
            self.pipeline.add(gldownload)

        convert = Gst.ElementFactory.make("videoconvert", "stitcher_convert")
        if convert is None:
            raise RuntimeError("Missing videoconvert element.")
        self.pipeline.add(convert)
        elements_chain.append(convert)

        scale = Gst.ElementFactory.make("videoscale", "stitcher_scale")
        if scale is None:
            raise RuntimeError("Missing videoscale element.")
        self.pipeline.add(scale)
        elements_chain.append(scale)

        caps = Gst.ElementFactory.make("capsfilter", "stitcher_caps")
        if caps is None:
            raise RuntimeError("Missing capsfilter element.")
        caps_string = (
            f"video/x-raw,format=I420,width={composite_width},height={composite_height},"
            f"framerate={self.framerate}/1"
        )
        caps.set_property("caps", Gst.Caps.from_string(caps_string))
        self.pipeline.add(caps)
        elements_chain.append(caps)

        pre_encoder_queue = _create_queue("encoder_queue")
        self.pipeline.add(pre_encoder_queue)
        elements_chain.append(pre_encoder_queue)

        target_bps = None
        if effective_bitrate_mbps:
            target_bps = int(max(1.0, effective_bitrate_mbps) * 1_000_000)

        encoder, encoder_name = self._create_encoder(target_bps)
        self.encoder = encoder
        self.pipeline.add(encoder)
        logging.info("Using %s for H.264 encoding", encoder_name)

        h264parse = Gst.ElementFactory.make("h264parse", "stereo_h264parse")
        if h264parse is None:
            raise RuntimeError("Missing h264parse element.")
        h264parse.set_property("config-interval", 1)
        self.pipeline.add(h264parse)

        pay = Gst.ElementFactory.make("rtph264pay", "stereo_pay")
        if pay is None:
            raise RuntimeError("Missing rtph264pay element.")
        pay.set_property("config-interval", 1)
        pay.set_property("pt", 96)
        self.pipeline.add(pay)

        rtp_caps = Gst.ElementFactory.make("capsfilter", "stereo_rtp_caps")
        if rtp_caps is None:
            raise RuntimeError("Missing RTP capsfilter.")
        rtp_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                "application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000"
            ),
        )
        self.pipeline.add(rtp_caps)

        webrtc_queue = _create_queue("webrtc_queue")
        self.pipeline.add(webrtc_queue)

        webrtcbin = Gst.ElementFactory.make("webrtcbin", "stereo_webrtc")
        if webrtcbin is None:
            raise RuntimeError("Missing webrtcbin element.")
        webrtcbin.set_property("bundle-policy", GstWebRTC.WebRTCBundlePolicy.MAX_BUNDLE)
        webrtcbin.set_property("latency", 0)
        if self.ice_server:
            webrtcbin.set_property("stun-server", self.ice_server)
        self.pipeline.add(webrtcbin)
        self.webrtcbin = webrtcbin
        try:
            webrtcbin.emit(
                "add-transceiver",
                GstWebRTC.WebRTCRTPTransceiverDirection.SENDONLY,
                None,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to pre-create WebRTC transceiver: %s", exc)

        for upstream, downstream in zip(elements_chain, elements_chain[1:]):
            if not upstream.link(downstream):
                raise RuntimeError(f"Failed to link {upstream.get_name()} -> {downstream.get_name()}.")

        for upstream, downstream in (
            (pre_encoder_queue, encoder),
            (encoder, h264parse),
            (h264parse, pay),
            (pay, rtp_caps),
            (rtp_caps, webrtc_queue),
        ):
            if not upstream.link(downstream):
                raise RuntimeError(f"Failed to link {upstream.get_name()} -> {downstream.get_name()}.")

        src_pad = webrtc_queue.get_static_pad("src")
        sink_pad = webrtcbin.request_pad_simple("sink_0") or webrtcbin.request_pad_simple("sink_%u")
        if src_pad is None or sink_pad is None:
            raise RuntimeError("Missing WebRTC pads for linking.")
        if src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link RTP payload into WebRTC bin.")

        webrtcbin.connect("on-ice-candidate", self._on_ice_candidate)
        webrtcbin.connect("notify::ice-gathering-state", self._on_ice_gathering_state)
        webrtcbin.connect("notify::connection-state", self._on_connection_state)

    def _build_camera_branch(
        self,
        name: str,
        camera_id: int,
        profile: dict,
        geometry: BranchGeometry,
        use_gl: bool,
    ) -> Gst.Element:
        elements: list[Gst.Element] = []

        src = Gst.ElementFactory.make("libcamerasrc", f"{name}_src")
        if src is None:
            raise RuntimeError("Missing libcamerasrc element.")
        if self.camera_names and camera_id < len(self.camera_names):
            camera_name = self.camera_names[camera_id]
            if camera_name and _has_property(src, "camera-name"):
                try:
                    src.set_property("camera-name", camera_name)
                except Exception as exc:  # noqa: BLE001
                    logging.warning("Failed to bind %s branch to camera '%s': %s", name, camera_name, exc)
        elif _has_property(src, "camera"):
            try:
                src.set_property("camera", camera_id)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to select camera index %s for %s: %s", camera_id, name, exc)
        elements.append(src)

        capsfilter = Gst.ElementFactory.make("capsfilter", f"{name}_raw_caps")
        if capsfilter is None:
            raise RuntimeError("Missing capsfilter for camera branch.")
        res = profile.get("resolution") or [geometry.width, geometry.height]
        caps_str = (
            f"video/x-raw,width={int(res[0])},height={int(res[1])},format=NV12,framerate={self.framerate}/1"
        )
        capsfilter.set_property("caps", Gst.Caps.from_string(caps_str))
        elements.append(capsfilter)

        elements.append(_create_queue(f"{name}_sensor_queue"))

        rotation = int(profile.get("rotation", 0)) % 360
        if rotation:
            rotate = Gst.ElementFactory.make("videoflip", f"{name}_rotate")
            if rotate is None:
                raise RuntimeError("Missing videoflip element.")
            method = {90: 1, 180: 2, 270: 3}.get(rotation, 0)
            rotate.set_property("method", method)
            elements.append(rotate)

        if bool(profile.get("hflip", False)):
            hflip = Gst.ElementFactory.make("videoflip", f"{name}_hflip")
            if hflip is None:
                raise RuntimeError("Missing videoflip element for hflip.")
            hflip.set_property("method", 4)
            elements.append(hflip)

        if bool(profile.get("vflip", False)):
            vflip = Gst.ElementFactory.make("videoflip", f"{name}_vflip")
            if vflip is None:
                raise RuntimeError("Missing videoflip element for vflip.")
            vflip.set_property("method", 5)
            elements.append(vflip)

        if any([geometry.crop_left, geometry.crop_right, geometry.crop_top, geometry.crop_bottom]):
            crop = Gst.ElementFactory.make("videocrop", f"{name}_crop")
            if crop is None:
                raise RuntimeError("Missing videocrop element.")
            crop.set_property("left", geometry.crop_left)
            crop.set_property("right", geometry.crop_right)
            crop.set_property("top", geometry.crop_top)
            crop.set_property("bottom", geometry.crop_bottom)
            elements.append(crop)

        elements.append(_create_queue(f"{name}_preprocess_queue"))

        if use_gl:
            glupload = Gst.ElementFactory.make("glupload", f"{name}_glupload")
            glcolor = Gst.ElementFactory.make("glcolorconvert", f"{name}_glcolor")
            if glupload is None or glcolor is None:
                raise RuntimeError("Missing GL upload/colorconvert elements.")
            elements.extend([glupload, glcolor])
            elements.append(_create_queue(f"{name}_premix_queue"))
        else:
            convert = Gst.ElementFactory.make("videoconvert", f"{name}_convert")
            scale = Gst.ElementFactory.make("videoscale", f"{name}_scale")
            if convert is None or scale is None:
                raise RuntimeError("Missing branching vconvert/vscale elements.")
            elements.extend([convert, scale])
            eye_caps = Gst.ElementFactory.make("capsfilter", f"{name}_eye_caps")
            if eye_caps is None:
                raise RuntimeError("Missing branch capsfilter.")
            eye_caps.set_property(
                "caps",
                Gst.Caps.from_string(
                    f"video/x-raw,width={geometry.width},height={geometry.height},format=I420"
                ),
            )
            elements.append(eye_caps)
            elements.append(_create_queue(f"{name}_premix_queue"))

        for element in elements:
            self.pipeline.add(element)

        for upstream, downstream in zip(elements, elements[1:]):
            if not upstream.link(downstream):
                raise RuntimeError(
                    f"Failed to link {upstream.get_name()} -> {downstream.get_name()} in {name} branch."
                )

        return elements[-1]

    async def start(self) -> None:
        state_change = self.pipeline.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Failed to start GStreamer pipeline.")
        self._shutdown_event.clear()
        self.bus_task = self.loop.create_task(self._bus_watch_loop())

    async def stop(self) -> None:
        self._shutdown_event.set()
        if self.bus_task:
            self.bus_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.bus_task
            self.bus_task = None
        self.pipeline.set_state(Gst.State.NULL)
        self._reset_session()

    async def _bus_watch_loop(self) -> None:
        bus = self.pipeline.get_bus()
        while not self._shutdown_event.is_set():
            message = bus.timed_pop_filtered(
                Gst.SECOND // 5,
                Gst.MessageType.ERROR
                | Gst.MessageType.EOS
                | Gst.MessageType.STATE_CHANGED
                | Gst.MessageType.WARNING,
            )
            if message is None:
                continue
            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                logging.error("Pipeline error: %s (%s)", err, debug)
                self._reset_session()
            elif message.type == Gst.MessageType.EOS:
                logging.info("Pipeline reached EOS.")
                self._reset_session()

    def _reset_session(self) -> None:
        if self._answer_future and not self._answer_future.done():
            self.loop.call_soon_threadsafe(
                self._answer_future.set_exception,
                RuntimeError("Pipeline reset during negotiation."),
            )
        if self._gather_future and not self._gather_future.done():
            self.loop.call_soon_threadsafe(self._gather_future.set_result, None)
        self._answer_future = None
        self._gather_future = None
        self._session_active = False

    async def create_answer(self, sdp: str, type_: str) -> Dict[str, str]:
        if type_.lower() != "offer":
            raise ValueError("Expected SDP offer from remote peer.")
        if self._session_active:
            raise RuntimeError("Existing peer connection active; please disconnect first.")

        if self.webrtcbin is None:
            raise RuntimeError("WebRTC component not initialised.")

        res, sdp_message = GstSdp.SDPMessage.new()
        if res != GstSdp.SDPResult.OK:
            raise RuntimeError("Failed to allocate SDP message.")
        parse_result = GstSdp.sdp_message_parse_buffer(sdp.encode("utf-8"), sdp_message)
        if parse_result != GstSdp.SDPResult.OK:
            raise ValueError("Invalid SDP offer payload.")

        offer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.OFFER, sdp_message)

        self._session_active = True
        self._answer_future = self.loop.create_future()
        self._gather_future = self.loop.create_future()

        promise = Gst.Promise.new()
        self.webrtcbin.emit("set-remote-description", offer, promise)
        promise.interrupt()

        answer_promise = Gst.Promise.new_with_change_func(self._on_answer_created, None)
        self.webrtcbin.emit("create-answer", None, answer_promise)

        try:
            answer_sdp = await asyncio.wait_for(self._answer_future, timeout=5)
        except asyncio.TimeoutError as exc:
            self._reset_session()
            raise RuntimeError("Timed out creating SDP answer.") from exc

        try:
            await asyncio.wait_for(self._gather_future, timeout=5)
        except asyncio.TimeoutError:
            logging.warning("ICE gathering incomplete; continuing with partial candidates.")

        return {"type": "answer", "sdp": answer_sdp}

    def _on_answer_created(self, promise: Gst.Promise, _data: object | None) -> None:
        if self.webrtcbin is None:
            return
        if promise.wait() != Gst.PromiseResult.REPLIED:
            self.loop.call_soon_threadsafe(
                self._answer_future.set_exception, RuntimeError("Failed to craft SDP answer.")
            )
            return
        reply = promise.get_reply()
        answer = reply.get_value("answer")
        local_promise = Gst.Promise.new()
        self.webrtcbin.emit("set-local-description", answer, local_promise)
        local_promise.interrupt()
        sdp_text = answer.sdp.as_text()
        if self._answer_future and not self._answer_future.done():
            self.loop.call_soon_threadsafe(self._answer_future.set_result, sdp_text)

    def _on_ice_candidate(self, _element: Gst.Element, mlineindex: int, candidate: str) -> None:
        logging.debug("Generated ICE candidate (m=%s): %s", mlineindex, candidate)

    def _on_ice_gathering_state(self, element: Gst.Element, _param: object) -> None:
        state = element.get_property("ice-gathering-state")
        if (
            state == GstWebRTC.WebRTCICEGatheringState.COMPLETE
            and self._gather_future
            and not self._gather_future.done()
        ):
            self.loop.call_soon_threadsafe(self._gather_future.set_result, None)

    def _on_connection_state(self, element: Gst.Element, _param: object) -> None:
        raw_state = element.get_property("connection-state")
        try:
            state_enum = GstWebRTC.WebRTCPeerConnectionState(raw_state)
        except Exception:  # noqa: BLE001
            state_enum = raw_state
        name = getattr(state_enum, "value_nick", str(state_enum))
        logging.info("WebRTC connection state: %s", name)
        if state_enum in (
            GstWebRTC.WebRTCPeerConnectionState.CLOSED,
            GstWebRTC.WebRTCPeerConnectionState.DISCONNECTED,
            GstWebRTC.WebRTCPeerConnectionState.FAILED,
        ):
            self._reset_session()

    def _create_encoder(self, bitrate_bps: Optional[int]) -> tuple[Gst.Element, str]:
        attempts: list[str] = []
        for name in (
            "v4l2h264enc",
            "v4l2slh264enc",
            "omxh264enc",
            "openh264enc",
            "x264enc",
        ):
            element = Gst.ElementFactory.make(name, f"stereo_{name}")
            if element is None:
                attempts.append(f"{name} (missing)")
                continue
            try:
                self._configure_encoder(element, name, bitrate_bps)
                return element, name
            except Exception as exc:  # noqa: BLE001
                attempts.append(f"{name} (error: {exc})")
        raise RuntimeError(
            "No usable H.264 encoder element found. Checked: " + ", ".join(attempts or ["none"])
        )

    def _configure_encoder(self, element: Gst.Element, name: str, bitrate_bps: Optional[int]) -> None:
        if name.startswith("v4l2") or name.startswith("omx"):
            if bitrate_bps:
                if _has_property(element, "bitrate"):
                    element.set_property("bitrate", bitrate_bps)
                elif _has_property(element, "extra-controls"):
                    element.set_property("extra-controls", f"controls,video_bitrate={bitrate_bps}")
            if _has_property(element, "iframe-period"):
                element.set_property("iframe-period", max(self.framerate, 1))
            if _has_property(element, "insert-sps-pps"):
                element.set_property("insert-sps-pps", True)
            if _has_property(element, "low-latency"):
                element.set_property("low-latency", True)
        elif name == "openh264enc":
            if bitrate_bps:
                if _has_property(element, "bitrate"):
                    element.set_property("bitrate", bitrate_bps)
                if _has_property(element, "max-bitrate"):
                    element.set_property("max-bitrate", bitrate_bps)
            if _has_property(element, "gop-size"):
                element.set_property("gop-size", max(1, int(self.framerate)))
            if _has_property(element, "rate-control"):
                element.set_property("rate-control", 1)
            if _has_property(element, "usage-type"):
                element.set_property("usage-type", 0)
        elif name == "x264enc":
            if bitrate_bps:
                if _has_property(element, "bitrate"):
                    element.set_property("bitrate", max(1, bitrate_bps // 1000))
            if _has_property(element, "speed-preset"):
                element.set_property("speed-preset", "ultrafast")
            if _has_property(element, "tune"):
                element.set_property("tune", "zerolatency")
            if _has_property(element, "key-int-max"):
                element.set_property("key-int-max", max(1, int(self.framerate)))
        else:
            raise RuntimeError(f"Unsupported encoder element {name}")

    @staticmethod
    def _probe_camera_ids() -> list[str]:
        try:
            env = os.environ.copy()
            env.setdefault("LIBCAMERA_LOG_LEVELS", "3")
            script = (
                "from picamera2 import Picamera2; import json;"
                " print(json.dumps(Picamera2.global_camera_info()))"
            )
            completed = subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            output = completed.stdout.strip().splitlines()
            if not output:
                return []
            payload = output[-1]
            infos = json.loads(payload)
            ids: list[str] = []
            for info in infos:
                cam_id = (
                    info.get("Id")
                    or info.get("CameraId")
                    or info.get("CameraID")
                    or info.get("Path")
                )
                if cam_id is None:
                    cam_id = str(info.get("Num", ""))
                ids.append(str(cam_id))
            return [cid for cid in ids if cid]
        except Exception as exc:  # noqa: BLE001
            logging.warning("Unable to probe camera IDs via Picamera2: %s", exc)
            return []


class Option3Server:
    def __init__(self, pipeline: GStreamerStereoPipeline, ca_cert: Optional[Path] = None):
        self.pipeline = pipeline
        self.ca_cert = ca_cert

    async def index(self, _request: web.Request) -> web.StreamResponse:
        html_path = STATIC_DIR / "index.html"
        if html_path.exists():
            return web.FileResponse(html_path)
        return web.Response(text="Client not found.", status=404)

    async def offer(self, request: web.Request) -> web.Response:
        payload = await request.json()
        try:
            answer = await self.pipeline.create_answer(
                payload["sdp"],
                payload.get("type", "offer"),
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception("Offer handling failed")
            status = 409 if isinstance(exc, RuntimeError) and "Existing peer connection active" in str(exc) else 400
            return web.Response(
                status=status,
                text=json.dumps({"error": str(exc)}),
                content_type="application/json",
            )
        return web.json_response(answer)

    async def serve_ca_certificate(self, _request: web.Request) -> web.Response:
        if not self.ca_cert:
            return web.Response(status=404, text="CA certificate not configured.")
        response = web.FileResponse(self.ca_cert)
        response.headers["Content-Disposition"] = f'attachment; filename="{self.ca_cert.name}"'
        response.headers["Cache-Control"] = "no-store"
        return response


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP for signalling server.")
    parser.add_argument("--port", type=int, default=8443, help="HTTP(s) port for signalling.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Calibration profile YAML.")
    parser.add_argument("--left-profile", default="cam0", help="Profile name for left camera.")
    parser.add_argument("--right-profile", default="cam1", help="Profile name for right camera.")
    parser.add_argument("--framerate", type=int, default=72, help="Target framerate for both cameras.")
    parser.add_argument(
        "--bitrate-mbps",
        type=float,
        help="Override total encoder bitrate (per composite stream) in Mbps.",
    )
    parser.add_argument(
        "--stun",
        help="Optional stun server URL (e.g. stun://stun.l.google.com:19302).",
    )
    parser.add_argument(
        "--cert",
        type=Path,
        help="Path to TLS certificate (PEM) for HTTPS/WebRTC (requires --key).",
    )
    parser.add_argument(
        "--key",
        type=Path,
        help="Path to TLS private key (PEM) for HTTPS/WebRTC (requires --cert).",
    )
    parser.add_argument("--cert-pass", help="Optional password for TLS private key.")
    parser.add_argument(
        "--ca-cert",
        type=Path,
        help="Path to CA certificate to expose at /ca.crt for headset import.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--keep-pipewire",
        action="store_true",
        help="Do not stop PipeWire/WirePlumber even if they hold the cameras.",
    )
    parser.add_argument(
        "--auto-kill-conflicts",
        action="store_true",
        help="Automatically terminate processes holding /dev/video* when claiming the cameras.",
    )
    parser.add_argument(
        "--force-prompt",
        action="store_true",
        help="Force interactive prompts even if stdin is not a TTY.",
    )
    parser.add_argument(
        "--no-restart-pipewire",
        action="store_true",
        help="Do not restart PipeWire/WirePlumber services on exit if they were stopped.",
    )
    return parser


async def _shutdown(app: web.Application) -> None:
    pipeline: GStreamerStereoPipeline = app["pipeline"]
    await pipeline.stop()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    if args.cert or args.key:
        if not (args.cert and args.key):
            print("[ERROR] --cert and --key must be provided together.", file=sys.stderr)
            sys.exit(1)
        if not args.cert.exists():
            print(f"[ERROR] TLS certificate not found: {args.cert}", file=sys.stderr)
            sys.exit(1)
        if not args.key.exists():
            print(f"[ERROR] TLS private key not found: {args.key}", file=sys.stderr)
            sys.exit(1)

    if not args.config.exists():
        print(f"[ERROR] Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    ssl_context = None
    if args.cert and args.key:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            ssl_context.load_cert_chain(
                certfile=str(args.cert),
                keyfile=str(args.key),
                password=args.cert_pass,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to load TLS material: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.stun and not args.stun.startswith(("stun://", "turn://")):
        stun_uri = f"stun://{args.stun}"
    else:
        stun_uri = args.stun

    stopped_services: list[str] = []
    try:
        if args.keep_pipewire:
            kill_mode = "none"
        elif args.auto_kill_conflicts:
            kill_mode = "auto"
        elif args.force_prompt:
            kill_mode = "prompt"
        elif not sys.stdin.isatty():
            kill_mode = "auto"
        else:
            kill_mode = "prompt"
        logging.debug(
            "Camera acquisition guard using kill_mode=%s (stdin isatty=%s)",
            kill_mode,
            sys.stdin.isatty(),
        )
        stopped_services = _ensure_camera_ready(not args.keep_pipewire, kill_mode)
    except RuntimeError as exc:
        logging.error("%s", exc)
        if not args.keep_pipewire:
            logging.error(
                "Cameras remain busy even after attempting to stop PipeWire/WirePlumber. "
                "Try --auto-kill-conflicts or stop the offending process manually."
            )
        sys.exit(1)
    if stopped_services and args.no_restart_pipewire:
        logging.warning(
            "PipeWire/WirePlumber services stopped and left inactive (--no-restart-pipewire set)."
        )
    elif stopped_services:
        atexit.register(_restart_services, stopped_services)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    pipeline = GStreamerStereoPipeline(
        loop=loop,
        config_path=args.config,
        left_profile_name=args.left_profile,
        right_profile_name=args.right_profile,
        framerate=args.framerate,
        ice_server=stun_uri,
        bitrate_mbps=args.bitrate_mbps,
    )

    loop.run_until_complete(pipeline.start())

    ca_cert_path = args.ca_cert.expanduser() if args.ca_cert else None
    if ca_cert_path and not ca_cert_path.exists():
        print(f"[ERROR] CA certificate not found: {ca_cert_path}", file=sys.stderr)
        sys.exit(1)

    server = Option3Server(pipeline, ca_cert=ca_cert_path)
    app = web.Application()
    app["pipeline"] = pipeline
    app.router.add_get("/", server.index)
    app.router.add_post("/offer", server.offer)
    app.router.add_static("/static/", STATIC_DIR, show_index=True)
    if ca_cert_path:
        app.router.add_get("/ca.crt", server.serve_ca_certificate)

    app.on_shutdown.append(_shutdown)

    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            signal.signal(sig, lambda *_args: stop_event.set())

    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, host=args.host, port=args.port, ssl_context=ssl_context)
    loop.run_until_complete(site.start())

    logging.info("GStreamer Option 3 pipeline running on %s:%s", args.host, args.port)
    try:
        loop.run_until_complete(stop_event.wait())
    finally:
        logging.info("Shutting down...")
        loop.run_until_complete(app.shutdown())
        loop.run_until_complete(runner.cleanup())
        loop.run_until_complete(app.cleanup())
        loop.run_until_complete(pipeline.stop())
        if stopped_services and not args.no_restart_pipewire:
            _restart_services(stopped_services)
        loop.close()


if __name__ == "__main__":
    main()
