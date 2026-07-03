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


def find_source_region(stack: np.ndarray) -> np.ndarray:
    """The source is the strongest-flashing region (the blink.html window)."""
    std = stack.std(axis=0)
    thresh = np.percentile(std, 99)
    candidate = (std >= max(thresh, 10)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels, labels = cv2.connectedComponents(candidate)
    if n_labels < 2:
        raise SystemExit("No flashing region found — is blink.html visible and flipping?")
    sizes = [(labels == i).sum() for i in range(1, n_labels)]
    return labels == (1 + int(np.argmax(sizes)))


def normalized(signal: np.ndarray) -> np.ndarray:
    signal = signal - signal.mean()
    norm = np.linalg.norm(signal)
    return signal / norm if norm else signal


def find_delayed_echo_mask(
    stack: np.ndarray, source_mask: np.ndarray, max_lag: int, min_lag: int = 10
) -> np.ndarray:
    """Mask of pixels that echo the source flash at a genuinely positive lag.

    Region-level detection fails here because instant glare (surfaces lit by
    the flash at ~0 lag, including the screen's own glass) spatially merges
    with the true stream echo. Instead, classify per pixel: FFT
    cross-correlate each candidate pixel's time series with the source signal
    and keep pixels whose BEST lag is >= min_lag frames.
    """
    # Downsample time 2x and average 8x8 tiles: single pixels are too noisy to
    # classify, tiles recover SNR while keeping spatial separation from glare.
    TILE = 8
    ds = stack[::2]
    source_signal = normalized(ds[:, source_mask].mean(axis=1))
    exclude = cv2.dilate(source_mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0

    n, height, width = ds.shape
    th, tw = height // TILE, width // TILE
    tiles = ds[:, : th * TILE, : tw * TILE].reshape(n, th, TILE, tw, TILE).mean(axis=(2, 4))
    tile_excluded = (
        exclude[: th * TILE, : tw * TILE].reshape(th, TILE, tw, TILE).max(axis=(1, 3)) > 0
    )

    signals = tiles.reshape(n, -1)
    signals = signals - signals.mean(axis=0)
    norms = np.linalg.norm(signals, axis=0)
    norms[norms == 0] = 1
    signals = signals / norms

    nfft = 1 << (2 * n - 1).bit_length()
    spec_src = np.fft.rfft(source_signal, nfft)
    spec_tiles = np.fft.rfft(signals, nfft, axis=0)
    # corr[k] = sum_t src[t] * tile[t+k]
    corr = np.fft.irfft(np.conj(spec_src)[:, None] * spec_tiles, nfft, axis=0)[: max_lag // 2]

    best_lag = corr.argmax(axis=0).reshape(th, tw) * 2  # undo time downsampling
    best_corr = corr.max(axis=0).reshape(th, tw)
    corr_at_zero = corr[0].reshape(th, tw)
    delayed = (
        (best_corr >= 0.3)
        & (best_lag >= min_lag)
        & (best_corr > corr_at_zero + 0.05)
        & ~tile_excluded
    )
    print(
        f"Tile classification: {th * tw} tiles, {int(delayed.sum())} delayed"
        + (f", median best lag {int(np.median(best_lag[delayed]))} frames" if delayed.any() else "")
    )
    if delayed.sum() < 2:
        raise SystemExit(
            "No delayed echo tiles found — the stream preview's flash isn't "
            "registering. Make the preview bigger/brighter or dim the room."
        )

    n_labels, labels = cv2.connectedComponents(delayed.astype(np.uint8))
    sizes = [(labels == i).sum() for i in range(1, n_labels)]
    best_tiles = labels == (1 + int(np.argmax(sizes)))
    mask = np.zeros(stack.shape[1:], bool)
    mask[: th * TILE, : tw * TILE] = np.kron(best_tiles, np.ones((TILE, TILE), bool))
    return mask


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
    parser.add_argument(
        "--flip-period-ms",
        type=float,
        default=500.0,
        help="blink.html flip period; used to self-calibrate real time per frame "
        "(handles rendered slo-mo exports where frame rate varies across the file). "
        "Pass 0 to trust the container fps instead.",
    )
    args = parser.parse_args()

    stack, meta_fps = load_luma_frames(args.video)
    fps = args.fps or meta_fps
    if not fps or fps < 30:
        raise SystemExit(f"Suspicious fps ({fps}); pass --fps (slo-mo is usually 120 or 240).")
    print(f"{stack.shape[0]} frames @ {fps:.0f}fps ({stack.shape[0] / fps:.1f}s of footage)")

    source_mask = find_source_region(stack)
    src_flips, src_dirs = flip_times(stack, source_mask)
    print(f"Source: {source_mask.sum()} px, {len(src_flips)} flips")
    # Echo can lag by up to ~1 flip period; in a slowed section one 500ms
    # period spans 0.5s*240fps frames, so search generously.
    echo_mask = find_delayed_echo_mask(stack, source_mask, max_lag=min(stack.shape[0] // 2, 240))

    def pair_lags(dst_flips, dst_dirs):
        """Per-flip lag in ms.

        Real time per frame varies across rendered slo-mo exports (the slowed
        section is 240fps-real inside a 60fps container), so when the flip
        period is known, the spacing between consecutive SOURCE flips — a
        known flip_period_ms of real time — calibrates each lag locally.
        """
        lags = []
        for i, (t, d) in enumerate(zip(src_flips, src_dirs)):
            later = dst_flips[(dst_flips > t) & (dst_dirs == d)]
            if not later.size:
                continue
            lag_frames = later[0] - t
            if args.flip_period_ms:
                if i + 1 >= len(src_flips):
                    continue
                period_frames = src_flips[i + 1] - src_flips[i]
                # A lag longer than the flip period means this flip's copy was
                # missed (paired with a later flip) — drop it.
                if period_frames <= 0 or lag_frames > period_frames:
                    continue
                lag_ms = lag_frames * args.flip_period_ms / period_frames
            else:
                lag_ms = lag_frames * 1000 / fps
            if 0 < lag_ms < 1000:
                lags.append(lag_ms)
        return lags

    dst_flips, dst_dirs = flip_times(stack, echo_mask)
    lags = pair_lags(dst_flips, dst_dirs)
    print(f"Echo: {echo_mask.sum()} px, {len(dst_flips)} flips, {len(lags)} paired with source")
    if len(lags) < 3:
        raise SystemExit("Too few paired flips — film a longer clip (~20s of slo-mo).")
    ms = sorted(lags)

    median = statistics.median(ms)
    q1 = ms[len(ms) // 4]
    q3 = ms[(3 * len(ms)) // 4]
    print("Per-flip latency (ms):", ", ".join(f"{v:.1f}" for v in ms))
    calib = "flip-period self-calibrated" if args.flip_period_ms else f"container fps {fps:.0f}"
    print(f"\nGlass-to-glass: median {median:.1f}ms, IQR {q1:.1f}-{q3:.1f}ms ({calib})")


if __name__ == "__main__":
    main()
