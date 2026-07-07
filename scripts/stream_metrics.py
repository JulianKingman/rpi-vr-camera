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
        # A correct reconstruction lands capture_us a little BEFORE encode-output, so the
        # delta is a small positive value (readout + encode). A negative or huge (seconds)
        # delta means the domain is wrong. A positive-but-large delta (>100ms) means the
        # domain is right but the encode pipeline is deep — a latency problem, not a metric
        # bug — so we distinguish the two rather than lumping both under "SUSPECT".
        if median < 0 or median > 5000:
            verdict = "SUSPECT — pts not in boottime domain, e2e metric invalid"
        elif median > 100:
            verdict = f"HIGH — domain OK but capture->encode latency is {median:.0f}ms (see [PILAT])"
        else:
            verdict = "OK"
        print(
            f"[CLOCK] {self._name}: boottime - pts over {self._sample_count} frames: "
            f"median={median:.1f}ms min={min(deltas_ms):.1f}ms max={max(deltas_ms):.1f}ms "
            f"{verdict}",
            flush=True,
        )


class PiLatencyMonitor:
    """Rolling steady-state report of capture -> encode-done latency for one eye.

    The one-shot ``PtsClockCheck`` samples only the first frames, which are
    contaminated by camera/encoder warm-up. This keeps sampling and prints a
    median/p95 over a sliding window every ``report_period_s`` seconds, after a
    warm-up grace period, so we can tell transient startup latency apart from a
    genuinely deep encode pipeline.
    """

    def __init__(self, name: str, warmup_frames: int = 120, window: int = 240, report_period_s: float = 3.0):
        self._name = name
        self._warmup_frames = warmup_frames
        self._samples: deque[float] = deque(maxlen=window)
        self._report_period_us = int(report_period_s * 1_000_000)
        self._count = 0
        self._last_report_us = 0
        self._lock = Lock()

    def sample(self, capture_us: int, enc_done_us: int) -> None:
        with self._lock:
            self._count += 1
            if self._count <= self._warmup_frames:
                return
            self._samples.append((enc_done_us - capture_us) / 1000.0)
            if self._last_report_us == 0:
                self._last_report_us = enc_done_us
                return
            if enc_done_us - self._last_report_us < self._report_period_us:
                return
            self._last_report_us = enc_done_us
            ordered = sorted(self._samples)
            snapshot = ordered
        if not snapshot:
            return
        median = snapshot[len(snapshot) // 2]
        p95 = snapshot[min(len(snapshot) - 1, int(len(snapshot) * 0.95))]
        print(
            f"[PILAT] {self._name}: capture->encode-done over {len(snapshot)} frames: "
            f"median={median:.1f}ms p95={p95:.1f}ms",
            flush=True,
        )
