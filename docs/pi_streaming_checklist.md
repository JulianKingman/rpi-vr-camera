# Raspberry Pi Streaming Checklist

Quick-reference for running the WebRTC stereo stream on the Pi and verifying the performance optimizations.

## Pre-flight

```bash
# SSH into the Pi
ssh pi@<pi-ip>

# Pull latest changes
cd ~/rpi-vr-camera
git pull

# Ensure dependencies are installed
make python-deps
```

## Verify Config

Check that `config/camera_profiles.yaml` has the encoder settings you intend for both cam0 and cam1:

```yaml
bitrate_mbps: 40.0
h264_profile: baseline    # no B-frames = no reordering latency (code default is "high"!)
gop_frames: 2             # optional; without it the GOP defaults to 1 second
repeat_headers: true      # SPS/PPS on every keyframe for stream resilience
```

If `h264_profile`/`gop_frames` are missing, the defaults are high profile and a 1-second
GOP — with a 1s GOP a single lost keyframe freezes the stream for up to a second, and
PLI-triggered keyframes do NOT work on this server (aiortc ignores keyframe requests for
pre-encoded tracks).

> **Note (Pi 5):** despite the class name, `picamera2.encoders.H264Encoder` is
> SOFTWARE x264 on the Pi 5 — the BCM2712 has no H.264 hardware encoder. Encoder
> settings here are x264 settings, and encode cost is CPU, not ISP.

## Start Streaming

```bash
# Basic (default 56fps, 2304x1296)
make stream-webrtc

# With custom framerate/resolution
make stream-webrtc ARGS="--framerate 56 --resolution 2048x1296"

# With TLS for Quest (required for WebXR)
make stream-webrtc ARGS="--cert certs/server.pem --key certs/server-key.pem --ca-cert certs/ca.pem"
```

The server binds to `0.0.0.0:8443` by default. Open `https://<pi-ip>:8443` in a browser.

## TLS Certs (for Quest)

If you haven't generated certs yet:

```bash
make tls-certs
```

Then install the CA cert on the Quest: download from `https://<pi-ip>:8443/ca.crt` and import under Settings > Security > Install certificates.

## HUD Verification Checklist

After clicking "Start Stream" in the browser, the HUD should show 6 rows. Here's what to look for:

### Bitrate
- **Expected**: L and R both near 40 Mbps (green)
- **Red (<20 Mbps)**: Encoder struggling or bitrate config wrong
- **Check**: `grep bitrate_mbps config/camera_profiles.yaml`

### FPS
- **Expected**: Both eyes near the target framerate (e.g., 56 fps), green
- **Yellow (>5% off)**: Minor scheduling jitter, usually OK
- **Red (>15% off)**: Camera or encoder can't keep up
- **Check**: Try lowering resolution or framerate

### Resolution
- **Expected**: Matches your config (e.g., 2048x1296 or 1296x1296 after crop/rotate)
- **0x0**: Track not decoding yet, wait a few seconds

### Latency
- **rtt**: Network round-trip (labelled `dc` when measured over the datachannel, `ice` as fallback). Green <10ms on LAN, yellow 10-30ms, red >30ms
- **jb (L/R)**: Jitter buffer per eye. Green <20ms, yellow 20-60ms, red >60ms
  - High JB = bursty frame delivery or network jitter
  - The JB target HUD input is applied in milliseconds (default 0)
- **dec**: Decode time per frame. Should be <5ms on modern hardware

### E2E (capture → render, the headline latency metric)
- Shows per-eye p50/p95 of true capture-to-render latency, measured per frame by
  matching `requestVideoFrameCallback` RTP timestamps against server capture
  timestamps (clock-synced over the datachannel). Green ≤25ms, yellow ≤35ms.
- `pi/net/buf` breakdown: Pi capture+encode / network+pacing / client buffer+decode+present.
- `sync±x` is the clock-offset confidence — treat E2E as suspect if it exceeds ~2ms.
- `locking...` for the first ~1s is normal (RTP origin recovery). `stale` means
  frames stopped presenting; `n/a` means the browser lacks rVFC.
- On the Pi, watch the server's startup `[CLOCK]` line: if it says SUSPECT, the
  camera PTS clock domain is wrong and E2E values cannot be trusted.

### Network
- **jit**: Packet jitter. <10ms is good on LAN
- **loss**: Packet loss percentage. Green <0.1%, red >1%
  - If loss is high, check WiFi signal or switch to wired ethernet
- **nack/pli**: Retransmission requests. Should be 0 normally
  - Frequent NACKs = packet loss; frequent PLIs = keyframe requests (decoder lost sync)

### Server
- **L/R bitrate**: Server-side encoding bitrate (should match client-side Bitrate row)
- **fps / kf**: Frames and keyframes counted at the encoder output (authoritative —
  aiortc's own outbound stats carry no frame counts). kf rate should match your GOP:
  `framerate / gop_frames` per second.
- **drop**: Broadcaster dropped frames per eye. Should be 0. If >0, encoder or network can't keep up

## Verifying Optimizations

### 1. Baseline Profile (no B-frames)
Look at the Pi's terminal output for:
```
H264Encoder(..., profile='baseline', ...)
```
Or check that the encoder isn't producing B-frames (no reordering latency visible in JB stats).

### 2. GOP taking effect
In the HUD's Server row, `kf=` should increment at `framerate / gop_frames` per second.
If kf stays at 0 or increments once a second when you configured a short GOP, the
setting isn't taking effect.

### 3. Broadcaster Queue Fix
If the Pi is under load and dropping frames, you'll see:
```
[WARN] Broadcaster dropped frame (total: N)
```
The drop count also appears in the HUD Server row. Queue is now bounded at 3 (was 8).

### 4. SharedEpoch (timestamp alignment)
Both cameras now share a common PTS origin. You should NOT see large timestamp jumps between L/R tracks. If decode timing or JB differs wildly between L and R, timestamps may still be misaligned (file an issue).

### 5. Server Stats via Datachannel
The Server row in the HUD should populate with per-eye bitrate, keyframe count, and drop count. If it stays empty, the datachannel isn't connecting (check browser console for errors).

## Troubleshooting

### No video, just black
- Check Pi terminal for `[ERROR]` lines
- Verify cameras are connected: `libcamera-hello --camera 0` and `--camera 1`
- Check that both cameras are detected: `libcamera-hello --list-cameras`

### Video freezes after a few seconds
- Likely keyframe loss + long GOP. Verify `gop_frames: 2` in config
- Check for packet loss in the HUD Network row
- Try wired ethernet instead of WiFi

### One eye has much worse stats than the other
- Camera hardware issue or one CSI cable is loose
- SharedEpoch bug (check timestamp logs for misalignment)
- Crop/rotation mismatch in config (cam0 is rotated 270, cam1 is 90)

### High jitter buffer (>50ms)
- Network issue: check WiFi signal, try ethernet
- Frame pacing issue on encoder side: check for `[WARN] Broadcaster dropped frame` messages
- Lower JB target to 0 in the HUD and see if it helps (browser may override)

### HUD shows "waiting for data..."
- Stats haven't arrived yet, wait a few seconds after Start Stream
- If persistent, the WebRTC connection may not have completed (check browser console)

### Red FPS even though video looks smooth
- The FPS target comes from the server's config message (falls back to inference only
  if that hasn't arrived). If it disagrees with your configured framerate, check the
  server terminal for datachannel errors.

## Recording a Baseline (before/after any optimization)

1. Fixed scene and lighting, same network, 30s warm-up after connect.
2. Click **Rec 60s** in the HUD. The report (every HUD metric per 250ms tick plus
   every raw per-frame E2E sample) is POSTed to the server and saved under `reports/`.
3. Repeat 3×, compare medians:
   ```bash
   python scripts/compare_reports.py reports/report-<before>.json reports/report-<after>.json
   ```
Never compare runs recorded with different scenes, networks, or configs.

## Physical Glass-to-Glass Measurement (ground truth, no software trust)

The software E2E metric must be certified against physics once per setup:

1. Open `https://<pi-ip>:8443/static/blink.html` on a monitor. Point the Pi camera at it
   (fill ~¼ of the frame). The panel flips black↔white every 500ms.
2. Start the stream. Put the Quest lens-side-up next to the monitor so a phone on a
   stand can film **both** the monitor and one Quest eyepiece in a single shot.
3. Record ~20s of 240fps slo-mo on the phone.
4. Scrub frame-by-frame: for ≥10 flips, count phone frames between the monitor flip and
   the same flip appearing in the Quest lens. Latency = frames × 4.17ms.
5. Report median and IQR (resolution ±4.2ms).
6. **Calibration gate**: physical median should exceed the HUD E2E p50 by roughly one
   display frame (~7–14ms, compositor + persistence). If they disagree by more than
   ~15ms beyond that, the software metric is broken (clock sync or PTS domain) — fix it
   before trusting any optimization numbers.

## Performance Targets (Pi on LAN, 72fps / <25ms goal)

| Metric | Good | Acceptable | Problem |
|--------|------|------------|---------|
| E2E p50 (software) | ≤18ms | 18-28ms | >28ms |
| Glass-to-glass (blink test) | ≤25ms | 25-35ms | >35ms |
| FPS | ≥0.97× target | 0.9-0.97× | <0.9× |
| Bitrate | 0.7-1.3× config | 0.4-0.7× | <0.4× |
| RTT | <5ms | 5-15ms | >30ms |
| JB | <15ms | 15-40ms | >60ms |
| Decode | <3ms | 3-8ms | >15ms |
| Loss | 0% | <0.1% | >1% |
| Drops | 0 | 1-3/interval | >3 |
