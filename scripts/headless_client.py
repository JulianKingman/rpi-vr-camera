#!/usr/bin/env python3
"""Headless WebRTC client that connects to the streaming server and logs all stats.

aiortc's inbound stats are limited (no bytesReceived/framesDecoded/frameWidth),
so we rely on:
  - Client-side: packetsReceived, packetsLost, jitter, datachannel RTT
  - Server-side: bytesSent, framesSent, keyFramesSent, droppedFrames (via datachannel)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

SERVER_URL = "http://127.0.0.1:8443"
STATS_INTERVAL = 1.0


async def drain_track(track):
    """Consume frames from a track so the receive pipeline stays healthy."""
    try:
        while True:
            await track.recv()
    except MediaStreamError:
        pass
    except Exception as exc:
        print(f"[DRAIN] {track.kind} track ended: {exc}", flush=True)


async def run_client():
    pc = RTCPeerConnection()

    prev: dict = {}
    server_stats: dict = {}  # mid -> latest server_stats msg
    prev_server: dict = {}   # mid -> previous server_stats msg

    dc = pc.createDataChannel("latency")
    ping_seq = 0
    pending_pings: dict[int, float] = {}
    ewma_rtt: Optional[float] = None

    @dc.on("open")
    def on_open():
        print("[DC] DataChannel open", flush=True)

    @dc.on("message")
    def on_message(message):
        nonlocal ewma_rtt
        if not isinstance(message, str):
            return
        try:
            msg = json.loads(message)
        except Exception:
            return

        if msg.get("type") == "pong" and isinstance(msg.get("seq"), int):
            sent_t = pending_pings.pop(msg["seq"], None)
            if sent_t is not None:
                rtt = (time.monotonic() - sent_t) * 1000
                ewma_rtt = rtt if ewma_rtt is None else ewma_rtt * 0.8 + rtt * 0.2

        if msg.get("type") == "server_stats":
            mid = msg.get("mid", "?")
            prev_server[mid] = server_stats.get(mid)
            server_stats[mid] = {**msg, "_t": time.monotonic()}

    drain_tasks = []

    @pc.on("track")
    def on_track(track):
        print(f"[TRACK] Received {track.kind} track: {track.id}", flush=True)
        drain_tasks.append(asyncio.ensure_future(drain_track(track)))

    @pc.on("connectionstatechange")
    async def on_state():
        print(f"[CONN] State: {pc.connectionState}", flush=True)

    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    print(f"[SIG] Sending offer to {SERVER_URL}/offer ...", flush=True)
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SERVER_URL}/offer",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"[ERROR] Offer failed ({resp.status}): {text}", flush=True)
                return
            answer = await resp.json()

    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    print("[SIG] Remote description set, waiting for connection...", flush=True)

    for _ in range(50):
        if pc.connectionState == "connected":
            break
        await asyncio.sleep(0.1)

    if pc.connectionState != "connected":
        print(f"[WARN] Connection state is {pc.connectionState} after 5s", flush=True)

    print("[STATS] Starting stats loop (Ctrl+C to stop)\n", flush=True)

    try:
        while pc.connectionState not in ("failed", "closed"):
            # Send ping
            if dc.readyState == "open":
                t = time.monotonic()
                pending_pings[ping_seq] = t
                try:
                    dc.send(json.dumps({"type": "ping", "seq": ping_seq, "t": t}))
                except Exception:
                    pass
                ping_seq += 1

            stats = await pc.getStats()
            now = time.monotonic()

            elapsed = now - prev.get("t", now)
            if elapsed <= 0:
                prev["t"] = now
                await asyncio.sleep(STATS_INTERVAL)
                continue

            lines = []

            if ewma_rtt is not None:
                lines.append(f"  RTT(dc): {ewma_rtt:.1f}ms")

            for report in stats.values():
                if report.type == "inbound-rtp" and getattr(report, "kind", None) == "video":
                    mid = getattr(report, "mid", "?")
                    packets_lost = getattr(report, "packetsLost", 0)
                    packets_recv = getattr(report, "packetsReceived", 0)
                    jitter = getattr(report, "jitter", None)

                    prev_key = f"inbound_{mid}"
                    p = prev.get(prev_key, {})

                    d_lost = packets_lost - p.get("packetsLost", 0)
                    d_recv = packets_recv - p.get("packetsReceived", 0)
                    loss_pct = (d_lost / (d_lost + d_recv) * 100) if (d_lost + d_recv) > 0 else 0

                    parts = [
                        f"mid={mid}",
                        f"pkts={d_recv}",
                        f"loss={loss_pct:.2f}%",
                    ]
                    if jitter is not None:
                        parts.append(f"jitter={jitter:.1f}ms")

                    lines.append(f"  [RX] {' | '.join(parts)}")

                    prev[prev_key] = {
                        "packetsLost": packets_lost,
                        "packetsReceived": packets_recv,
                    }

            # Server-side stats (from datachannel)
            for mid in sorted(server_stats.keys()):
                cur = server_stats[mid]
                prv = prev_server.get(mid)
                if not prv or "_t" not in prv:
                    lines.append(
                        f"  [TX] mid={mid} bytesSent={cur.get('bytesSent',0)} "
                        f"frames={cur.get('framesSent',0)} "
                        f"keyFrames={cur.get('keyFramesSent',0)} "
                        f"drops={cur.get('droppedFrames',0)}"
                    )
                    continue

                dt = cur["_t"] - prv["_t"]
                if dt <= 0:
                    continue
                d_bytes = cur.get("bytesSent", 0) - prv.get("bytesSent", 0)
                d_frames = cur.get("framesSent", 0) - prv.get("framesSent", 0)
                bitrate_mbps = (d_bytes * 8) / dt / 1_000_000
                fps = d_frames / dt

                lines.append(
                    f"  [TX] mid={mid} bitrate={bitrate_mbps:.2f}Mbps "
                    f"fps={fps:.1f} "
                    f"keyFrames={cur.get('keyFramesSent',0)} "
                    f"drops={cur.get('droppedFrames',0)}"
                )

            if lines:
                print(f"--- Stats @ +{now - prev.get('t0', now):.0f}s ---", flush=True)
                if "t0" not in prev:
                    prev["t0"] = now
                for line in lines:
                    print(line, flush=True)
                print(flush=True)

            prev["t"] = now
            await asyncio.sleep(STATS_INTERVAL)

    except KeyboardInterrupt:
        pass
    finally:
        for task in drain_tasks:
            task.cancel()
        await pc.close()
        print("[DONE] Client closed.", flush=True)


if __name__ == "__main__":
    asyncio.run(run_client())
