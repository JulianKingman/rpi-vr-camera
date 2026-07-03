#!/usr/bin/env python3
"""Measure glass-to-glass latency from slo-mo footage of the blink test.

Point the camera rig at web/blink.html, film the source monitor and the
stream preview together in one high-framerate video, then:

    python scripts/measure_blink.py path/to/slomo.mov

The script finds the two flashing regions automatically (highest temporal
luminance variance), extracts each region's black<->white flip times, pairs
source flips with their delayed copies in the stream, and prints per-flip
lag plus median/IQR. Pass --fps if the container metadata is wrong.
"""

from __future__ import annotations

import argparse
import statistics
import sys

import cv2
import numpy as np


def load_luma_frames(path: str, max_width: int = 320) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = max_width / gray.shape[1]
        if scale < 1:
            gray = cv2.resize(gray, (max_width, int(gray.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        frames.append(gray.astype(np.float32))
    cap.release()
    if len(frames) < 60:
        raise SystemExit(f"Only {len(frames)} frames decoded — need a longer clip.")
    return np.stack(frames), fps


def find_flash_regions(stack: np.ndarray, n_regions: int = 2) -> list[np.ndarray]:
    """Return boolean masks for the N strongest independently-flashing regions."""
    std = stack.std(axis=0)
    thresh = np.percentile(std, 99)
    candidate = (std >= max(thresh, 10)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels, labels = cv2.connectedComponents(candidate)
    components = [(labels == i).sum() for i in range(1, n_labels)]
    order = np.argsort(components)[::-1][: n_regions * 3]  # extra candidates

    # Merge components whose signals are near-identical (same physical screen
    # split by glare), then keep the two least-correlated groups.
    masks, signals = [], []
    for idx in order:
        mask = labels == (idx + 1)
        if mask.sum() < 20:
            continue
        signal = stack[:, mask].mean(axis=1)
        merged = False
        for i, existing in enumerate(signals):
            r = np.corrcoef(signal, existing)[0, 1]
            if r > 0.9:
                masks[i] = masks[i] | mask
                signals[i] = stack[:, masks[i]].mean(axis=1)
                merged = True
                break
        if not merged:
            masks.append(mask)
            signals.append(signal)
    if len(masks) < n_regions:
        raise SystemExit(
            f"Found only {len(masks)} flashing region(s); make sure both the source "
            "monitor and the stream preview are visible and flashing in frame."
        )
    return masks[:n_regions]


def flip_times(stack: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frame indices where the region crosses its mid-luminance, with direction."""
    signal = stack[:, mask].mean(axis=1)
    mid = (signal.max() + signal.min()) / 2
    binary = signal > mid
    changes = np.nonzero(binary[1:] != binary[:-1])[0] + 1
    directions = binary[changes]  # True = went white
    return changes.astype(float), directions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Slo-mo video file (AirDrop it from the phone)")
    parser.add_argument("--fps", type=float, default=None, help="Override recording fps if metadata is wrong")
    args = parser.parse_args()

    stack, meta_fps = load_luma_frames(args.video)
    fps = args.fps or meta_fps
    if not fps or fps < 30:
        raise SystemExit(f"Suspicious fps ({fps}); pass --fps (slo-mo is usually 120 or 240).")
    print(f"{stack.shape[0]} frames @ {fps:.0f}fps ({stack.shape[0] / fps:.1f}s of footage)")

    mask_a, mask_b = find_flash_regions(stack)
    flips_a, dir_a = flip_times(stack, mask_a)
    flips_b, dir_b = flip_times(stack, mask_b)
    print(f"Region A: {mask_a.sum()} px, {len(flips_a)} flips; Region B: {mask_b.sum()} px, {len(flips_b)} flips")

    def pair_lags(src_flips, src_dirs, dst_flips, dst_dirs):
        lags = []
        for t, d in zip(src_flips, src_dirs):
            later = dst_flips[(dst_flips > t) & (dst_dirs == d)]
            if later.size:
                lag = later[0] - t
                if lag < fps:  # ignore pairings >1s apart (missed flip)
                    lags.append(lag)
        return lags

    # The source region is whichever ordering yields the smaller positive median lag.
    lags_ab = pair_lags(flips_a, dir_a, flips_b, dir_b)
    lags_ba = pair_lags(flips_b, dir_b, flips_a, dir_a)
    if not lags_ab and not lags_ba:
        raise SystemExit("Could not pair any flips between the two regions.")
    lags = min((l for l in (lags_ab, lags_ba) if l), key=statistics.median)
    which = "A->B" if lags is lags_ab else "B->A"

    ms = sorted(lag * 1000 / fps for lag in lags)
    median = statistics.median(ms)
    q1 = ms[len(ms) // 4]
    q3 = ms[(3 * len(ms)) // 4]
    print(f"\nDirection {which}, {len(ms)} paired flips")
    print("Per-flip latency (ms):", ", ".join(f"{v:.1f}" for v in ms))
    print(f"\nGlass-to-glass: median {median:.1f}ms, IQR {q1:.1f}-{q3:.1f}ms, resolution ±{1000 / fps:.1f}ms")


if __name__ == "__main__":
    main()
