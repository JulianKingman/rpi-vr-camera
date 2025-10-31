# Low-Latency Stereo Streaming – Option 2 vs Option 3

## 1. Background & Goals
- Current `scripts/webrtc_stream.py` path uses Picamera2 `capture_array()` per camera, processes frames in Python/OpenCV, and relies on aiortc’s software H.264 encoder.
- Observed issues: high glass-to-glass latency (>60 ms), choppy perceived framerate despite nominal 56 fps capture, and poor bitrate/quality under load.
- Target: deliver <25 ms motion-to-photon at >72 Hz with consistent 30–50 Mbit quality to Oculus/Quest clients, while keeping calibration logic in YAML and HTTPS/WebRTC signalling in Python.

This document details two redesign options:
1. **Option 2 – Dual hardware encode + twin WebRTC tracks** (retain Python control surface).
2. **Option 3 – Full GStreamer pipeline** (GPU-heavy, declarative pipeline).

## 2. Shared Requirements & Constraints
- Raspberry Pi 5 dual IMX708 sensors, Pi OS Bookworm 64-bit.
- Calibration, crop, flip, and alignment data should remain editable via `config/camera_profiles.yaml`.
- TLS + signalling remains via aiohttp (or equivalent) and must serve Quest browser clients.
- Solution must expose metrics (bitrate, dropped frames, latency markers) for tuning.

## 3. Option 2 – Dual Hardware Encode + Two WebRTC Tracks

### 3.1 Summary
Use Picamera2’s hardware encoder path (or GStreamer wrapper) per eye to produce Annex-B H.264 elementary streams in hardware. Wrap each encoded stream in a `MediaStreamTrack` and expose two independent WebRTC video tracks (left/right). Update the WebXR client to render each track per eye.

### 3.2 Architecture
- **Capture**: Two `Picamera2` instances configured with `create_video_configuration` + `H264Encoder`.
- **Encoding**: `picamera2.encoders.H264Encoder` or `libcamera` request API with hardware `v4l2` encode. Configure with slice mode and target bitrate pulled from YAML (`bitrate_mbps` per camera).
- **Transport**: Custom aiortc `VideoStreamTrack` subclass that feeds pre-encoded NAL units via `av.packet.Packet` to the RTP sender. Force RTP payload format `video/H264` with correct SPS/PPS.
- **Signalling**: Reuse existing aiohttp `/offer` endpoint; negotiate two `RTCRtpTransceiver` instances.
- **Client**: Amend `web/index.html` (WebXR) so it enumerates both receivers, binds left track to left eye layer and right track to right eye layer.

```
Picamera2(cam0) ──> H264Encoder ──> NAL queue ──┐
                                                ├─> aiortc sender (track "left") ──> WebRTC
Picamera2(cam1) ──> H264Encoder ──> NAL queue ──┘
```

### 3.3 Data Flow & Threading
- Capture loop replaced with encoder callbacks (`CircularOutput` or custom `EncodedBufferOutput`).
- Each callback enqueues timestamped NAL batches into asyncio-safe queues; aiortc pulls and pushes them on demand.
- Use monotonic timestamps to populate RTP `pts` and `dts`, enabling accurate sync.
- Optional: leverage `asyncio.create_task` to multiplex both streams and maintain constant encode buffers (3–4 frames max).

### 3.4 Configuration Touchpoints
- Extend `camera_profiles.yaml`:
  - `bitrate_mbps` per camera (already present).
  - `gop_frames`, `idr_interval`, `sps_pps_mode` for fine control.
  - `output_size` to allow optional downscale prior to encode (handled via Picamera2 ISP).
- CLI flags for `--max-latency-ms`, `--slice-bytes`, `--qp-min/max`.

### 3.5 WebRTC/Aiortc Changes
- Add helper to advertise H.264 High/L3.1 with stereo-friendly parameter sets.
- Set `RTCRtpSender._encoder` to a passthrough mode or subclass aiortc encoder to avoid re-encoding.
- Monitor `RTCRtpSender.getStats()` for `framesEncoded`, `framesSent`, `nackCount`.
- Add ICE candidate priority tweaks for low-latency UDP; allow override for TURN.

### 3.6 Client/WebXR Updates
- Modify WebXR session setup:
  - Request two video tracks via SDP offer.
  - Bind tracks to `XRWebGLLayer` textures.
  - Provide fallback side-by-side render if only one track arrives (legacy clients).
- Add latency instrumentation overlay (timestamp decode vs. capture).

### 3.7 Expected Performance
- Hardware encode path reduces CPU load by >50%.
- With tuned IDR (1 s) and slice size, expect stable 35–45 Mbit at 72–90 fps.
- Latency budget: ~8 ms sensor → ISP, 4 ms encode, <4 ms jitter buffer, <6 ms decode/display on Quest.
- Glass-to-glass target ≈20–22 ms.

### 3.8 Risks & Mitigations
- **Encoder passthrough in aiortc**: aiortc API is optimized for raw frames; ensure compatibility with pre-encoded streams. Mitigate with integration tests and fallback to minimal decoding/re-encoding only if needed.
- **Synchronization drift**: Two separate encoders could drift; align frame timestamps using shared clock and periodic SPS alignment.
- **Client support**: Quest browser must accept two video tracks. Validate with WebXR sample; provide fallback side-by-side.
- **Thermals**: Dual encoders at 40 Mbit each can heat the Pi; require active cooling + monitor with `vcgencmd`.

### 3.9 Milestones
1. Prototype single camera encoded track with aiortc passthrough.
2. Generalize to dual tracks; confirm near-zero rebuffering on wired LAN.
3. Update WebXR client + QA on Quest 2/3.
4. Add telemetry (Prometheus or simple JSON endpoint).
5. Harden error handling (encoder resets, camera disconnect).

## 4. Option 3 – Full GStreamer Pipeline

### 4.1 Summary
Move capture, processing, encode, and WebRTC transport into a GStreamer graph on the Pi. Use `libcamerasrc` (or `rpicamsrc`) to ingest both sensors, perform GPU composites with `glvideomixer` or shaders, encode via `v4l2h264enc`, and publish through `webrtcbin`. Python (or a minimal C++ helper) only orchestrates pipeline startup and signalling.

### 4.2 Architecture
- **Pipeline Core** (one-line schematic):
  ```
  libcamerasrc device=/base/soc/i2c0mux/... ! video/x-raw,format=NV12 ! queue !
    glupload ! glshader (calibration rectify) ! queue name=left_q
  libcamerasrc device=/base/soc/i2c1mux/... ! video/x-raw,format=NV12 ! queue !
    glupload ! glshader (calibration rectify) ! queue name=right_q
  left_q. ! compositor name=stitcher … ! queue ! v4l2h264enc extra-controls=... !
    h264parse config-interval=1 ! rtph264pay pt=96 ! queue ! webrtcbin sendonly=true
  ```
- Optionally maintain two parallel encode branches (one per eye) if we want to keep dual-track output.
- Calibration offsets implemented via GLSL uniform parameters or `videocrop` + `videoflip`.
- `webrtcbin` handles DTLS/SRTP, trickle ICE, and can use QUIC data channels.

### 4.3 Control Plane & Signalling
- Python service (aiohttp or tornado) exposes `/offer` that forwards SDP to GStreamer via `webrtcbin` signals (`on-negotiation-needed`, `on-ice-candidate`).
- Configuration YAML consumed by the Python wrapper to build pipeline string with dynamic parameters (resolution, crop, framerate).
- Provide CLI to regenerate the `gst-launch-1.0` style string for debugging.

### 4.4 Calibration & Processing
- Use GPU path (`glshader`) for lens correction/crop to maintain throughput.
- For rectification, store coefficient matrices in YAML; upload as shader uniforms.
- Provide offline tool to compile GLSL fragments and validate on development machine.

### 4.5 Performance Expectations
- GStreamer leverages VideoCore VII GPU + hardware encoder; CPU usage minimal.
- End-to-end latency can drop to ≈12–15 ms with appropriate queue sizing and `latency=0` on `webrtcbin`.
- Easy to enable FEC/RTX, adapt bitrate dynamically, and integrate `Remb` feedback.

### 4.6 Risks & Complexity
- **Pipeline complexity**: Debugging GStreamer graphs on Pi is non-trivial; requires deep toolkit knowledge.
- **Shader maintenance**: Calibration via GLSL requires custom tooling/tests.
- **Threading**: Need to ensure `queue max-size-buffers=1 max-size-bytes=0 max-size-time=0` on critical links to prevent buffering.
- **Error handling**: Camera disconnects or encoder stalls need bus watch + automatic pipeline restart logic.
- **Tooling dependencies**: Requires `gstreamer1.0-gl`, `gst-plugins-bad` (already in prerequisites) and possibly custom builds for latest `webrtcbin` fixes.

### 4.7 Implementation Track
1. Build minimal `libcamerasrc ! v4l2h264enc ! filesink` pipeline per camera to confirm encode stability.
2. Introduce `webrtcbin` with test pattern to validate signalling path.
3. Integrate dual camera capture + stitcher; start with CPU `videomixer` before migrating to GPU shaders.
4. Port calibration parameters into GLSL/uniform pipelines.
5. Harden: add health checking, runtime reconfiguration (bitrate/framerate), metrics via `gst-shark` or `gst-stats`.
6. Optimize: explore `rtpsession` tuning, QUIC, or SRT for alternative transports.

### 4.8 Decision Points
- Whether to maintain dual-track vs. composite output inside GStreamer (client impact).
- Investment in shader pipeline vs. simpler crop/flip elements.
- Long-term maintainability: can the team support GStreamer expertise? If not, Option 2 may carry better cost/benefit.

## 5. Recommendation Snapshot
- **Option 2**: Moderate engineering effort, aligns with existing Python tooling, hits latency/quality targets with manageable risk. Recommended as next milestone.
- **Option 3**: Highest performance ceiling and feature richness but requires significant GStreamer + shader expertise; consider after Option 2 proves insufficient or if future roadmap includes advanced rendering/foveation.

## 6. Current Progress
- **Option 2 prototype online**: `scripts/webrtc_stream.py` now runs Picamera2’s hardware H.264 encoders for each sensor, fans out encoded NAL units via an asyncio broadcaster, and pushes dual aiortc tracks without re-encoding. The WebXR shell in `web/index.html` negotiates both tracks, exposes VR entry, and surfaces lightweight FPS/`getStats()` telemetry for live latency spotting.
- **Operational tooling**: New camera-process helpers (`scripts/find_camera_processes.sh`, `scripts/check_cameras.sh`, `scripts/kill_camera_processes.sh`) cleanly reclaim `/dev/video*` before streaming, while bitrate overrides and Picamera2 control patches let us sweep encoder targets straight from the YAML profiles.
- **Option 3 scaffolding**: `scripts/webrtc_gstreamer.py` builds the end-to-end GStreamer graph from the design (dual libcamerasrc branches → compositor → hardware encoder → webrtcbin), complete with PipeWire/WirePlumber suppression, ICE negotiation, and dynamic bitrate loading from the same configuration file.
- **Next validation loop**: Measure glass-to-glass latency on hardware for the Option 2 path, exercise the GStreamer pipeline on a cooled Pi 5, and compare stability/maintainability before locking the long-term direction; both redesign avenues stay active while we collect data.
