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

Check that `config/camera_profiles.yaml` has the optimized encoder settings for both cam0 and cam1:

```yaml
bitrate_mbps: 40.0
h264_profile: baseline    # no B-frames = no reordering latency
gop_frames: 2             # IDR every 2 frames = ~36ms recovery on loss
repeat_headers: true       # SPS/PPS on every keyframe for stream resilience
```

If any of these are missing, the defaults will be suboptimal (high profile with B-frames, 1-second GOP).

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
- **rtt**: Network round-trip. Green <10ms on LAN, yellow 10-30ms, red >30ms
- **jb (L/R)**: Jitter buffer per eye. Green <20ms, yellow 20-60ms, red >60ms
  - High JB = bursty frame delivery or network jitter
  - Try lowering JB target in the HUD input (default 10ms, try 0)
- **dec**: Decode time per frame. Should be <5ms on modern hardware

### Network
- **jit**: Packet jitter. <10ms is good on LAN
- **loss**: Packet loss percentage. Green <0.1%, red >1%
  - If loss is high, check WiFi signal or switch to wired ethernet
- **nack/pli**: Retransmission requests. Should be 0 normally
  - Frequent NACKs = packet loss; frequent PLIs = keyframe requests (decoder lost sync)

### Server
- **L/R bitrate**: Server-side encoding bitrate (should match client-side Bitrate row)
- **kf**: Cumulative keyframes sent. With `gop_frames: 2`, this should increment rapidly (~28/sec at 56fps)
- **drop**: Broadcaster dropped frames. Should be 0. If >0, encoder or network can't keep up

## Verifying Optimizations

### 1. Baseline Profile (no B-frames)
Look at the Pi's terminal output for:
```
H264Encoder(..., profile='baseline', ...)
```
Or check that the encoder isn't producing B-frames (no reordering latency visible in JB stats).

### 2. GOP = 2 (fast keyframe recovery)
In the HUD's Server row, `kf=` should increment rapidly. At 56fps with gop_frames=2, expect ~28 keyframes/sec. If kf stays at 0 or increments slowly, the gop setting isn't taking effect.

You can also watch the Pi's terminal:
```
[DEBUG] left keyframe <pts>
[DEBUG] right keyframe <pts>
```
These should appear every 2 frames.

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
- The FPS target auto-detects from observed framerate. If one eye briefly dips, the target may snap to a lower value. This is cosmetic.

## Performance Targets (Pi on LAN)

| Metric | Good | Acceptable | Problem |
|--------|------|------------|---------|
| FPS | 54-58 | 45-54 | <45 |
| Bitrate | 35-45 Mbps | 20-35 Mbps | <20 Mbps |
| RTT | <5ms | 5-15ms | >30ms |
| JB | <15ms | 15-40ms | >60ms |
| Decode | <3ms | 3-8ms | >15ms |
| Loss | 0% | <0.1% | >1% |
| Drops | 0 | 1-3/interval | >3 |
