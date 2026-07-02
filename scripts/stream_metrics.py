"""Shared measurement helpers for the WebRTC streaming servers.

Provides the server half of the end-to-end latency instrumentation:

- ``boottime_us()``: the server-side clock all timestamps are expressed in.
  CLOCK_BOOTTIME on Linux (the domain libcamera SensorTimestamps use),
  CLOCK_MONOTONIC elsewhere (the domain the Mac camera adapter uses).
- ``FrameTimestampLog``: per-eye ring buffer mapping each encoded frame's
  RTP-clock PTS (90 kHz) to its capture / encode-done wall points, plus
  authoritative frames/keyframes counters. The datachannel ``frame_ts``
  sender drains this; the client matches entries against
  requestVideoFrameCallback's rtpTimestamp to compute capture->render
  latency.
- ``PtsClockCheck``: one-shot startup verification that raw camera PTS
  values actually live in the boottime clock domain (they must, or every
  capture_us the client receives is garbage).

The 90 kHz conversion reuses aiortc's own ``convert_timebase`` so the
values here are bit-identical (same truncation) to what RTCRtpSender puts
on the wire, offset only by aiortc's random per-track timestamp origin —
which the client recovers by modal voting.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple

from aiortc.mediastreams import VIDEO_TIME_BASE, convert_timebase

MICROSECOND_TIME_BASE = Fraction(1, 1_000_000)

_BOOTTIME_CLOCK = getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)


def boottime_us() -> int:
    """Server timestamp in microseconds on the platform's frame-timestamp clock."""
    return time.clock_gettime_ns(_BOOTTIME_CLOCK) // 1000


def pts_to_90k(pts: int, time_base: Fraction) -> int:
    """Convert a track PTS to the 90 kHz RTP clock exactly as aiortc's sender does."""
    return convert_timebase(pts, time_base, VIDEO_TIME_BASE)


class FrameTimestampLog:
    """Ring buffer of per-frame timing for one eye, drained incrementally per client."""

    def __init__(self, maxlen: int = 512):
        self._lock = Lock()
        self._entries: deque[Tuple[int, int, int, int]] = deque(maxlen=maxlen)
        self._seq = 0
        self.frames = 0
        self.keyframes = 0

    def record(
        self,
        pts: int,
        time_base: Fraction,
        capture_us: int,
        enc_done_us: int,
        keyframe: bool,
    ) -> None:
        pts90k = pts_to_90k(pts, time_base)
        with self._lock:
            self._seq += 1
            self.frames += 1
            if keyframe:
                self.keyframes += 1
            self._entries.append((self._seq, pts90k, capture_us, enc_done_us))

    def since(self, cursor: int) -> Tuple[List[Tuple[int, int, int]], int]:
        """Entries newer than ``cursor`` as (pts90k, capture_us, enc_done_us), plus new cursor."""
        with self._lock:
            fresh = [e for e in self._entries if e[0] > cursor]
            if fresh:
                cursor = fresh[-1][0]
        return [(pts90k, cap, enc) for _, pts90k, cap, enc in fresh], cursor


def save_report(payload: dict, reports_dir: Path) -> Path:
    """Persist a client metrics report; returns the written path."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = reports_dir / f"report-{stamp}.json"
    counter = 1
    while path.exists():
        path = reports_dir / f"report-{stamp}-{counter}.json"
        counter += 1
    path.write_text(json.dumps(payload, indent=2))
    return path


class PtsClockCheck:
    """Verify raw frame PTS values are in the boottime clock domain.

    Samples ``boottime_us() - raw_pts_us`` for the first ``sample_count``
    frames and prints one summary line. The delta should be a small stable
    positive value (exposure + readout + encode). A huge or negative delta
    means the PTS domain assumption is wrong and capture_us cannot be
    trusted — the E2E metric must not be believed until this passes.
    """

    def __init__(self, name: str, sample_count: int = 100):
        self._name = name
        self._sample_count = sample_count
        self._deltas: Optional[List[int]] = []
        self._lock = Lock()

    def sample(self, raw_pts_us: int) -> None:
        with self._lock:
            if self._deltas is None:
                return
            self._deltas.append(boottime_us() - raw_pts_us)
            if len(self._deltas) < self._sample_count:
                return
            deltas_ms = [d / 1000 for d in self._deltas]
            self._deltas = None
        median = statistics.median(deltas_ms)
        print(
            f"[CLOCK] {self._name}: boottime - pts over {self._sample_count} frames: "
            f"median={median:.1f}ms min={min(deltas_ms):.1f}ms max={max(deltas_ms):.1f}ms "
            f"{'OK' if 0 <= median <= 100 else 'SUSPECT — pts not in boottime domain, e2e metric invalid'}",
            flush=True,
        )
