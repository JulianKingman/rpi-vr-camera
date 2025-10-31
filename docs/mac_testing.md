# Testing on macOS (Development Without RPI Hardware)

This guide explains how to test the VR camera streaming pipeline on macOS before deploying to Raspberry Pi.

## Options for Camera Testing

### 1. Use Mac Camera (Recommended for Real-World Testing)

Test with your Mac's built-in camera or external USB cameras:

```bash
# Use Mac camera (auto-detected)
export CAMERA_MODE=opencv
make stream-webrtc ARGS="--host 0.0.0.0 --port 8443"

# Or use test patterns (no hardware needed)
export CAMERA_MODE=test
export TEST_PATTERN=moving_bars  # or: timestamp, color_cycle, checkerboard
make stream-webrtc ARGS="--host 0.0.0.0 --port 8443"
```

The adapter automatically detects when `picamera2` is unavailable and falls back to OpenCV.

### 2. Test Patterns (Best for Latency Measurement)

Test patterns generate synthetic video frames at configurable frame rates, perfect for latency testing:

- `moving_bars`: Moving vertical bars (good for motion detection)
- `timestamp`: Frame with visible timestamp (best for latency measurement)
- `color_cycle`: Color cycling pattern
- `checkerboard`: Static checkerboard

```bash
export CAMERA_MODE=test
export TEST_PATTERN=timestamp
make stream-webrtc ARGS="--framerate 90 --resolution 1920x1080"
```

### 3. Dual Camera Simulation

For stereo testing on Mac with a single camera, the test pattern mode can simulate two cameras:

```bash
# Simulate dual cameras using test patterns
export CAMERA_MODE=test
python scripts/webrtc_stream.py --framerate 56
```

Both "cameras" will use the same test pattern (you can modify `camera_adapter.py` to generate different patterns per index).

## Quick Start

1. **Install dependencies** (OpenCV is the main requirement):
```bash
pip install opencv-python numpy aiortc aiohttp pyyaml
```

2. **Test with Mac camera**:
```bash
export CAMERA_MODE=opencv
python scripts/webrtc_stream.py --host 0.0.0.0 --port 8443
```

3. **Test with patterns** (no camera needed):
```bash
export CAMERA_MODE=test
export TEST_PATTERN=timestamp
python scripts/webrtc_stream.py --framerate 90
```

4. **Open in browser**: Navigate to `http://localhost:8443` (or `https://` if using TLS)

## Latency Testing Workflow

1. **Start with test patterns** to measure baseline streaming latency:
   ```bash
   export CAMERA_MODE=test
   export TEST_PATTERN=timestamp
   python scripts/webrtc_stream.py --framerate 90
   ```
   The timestamp pattern shows `T:<timestamp> F:<frame>` on each frame, making latency visible.

2. **Measure end-to-end latency** in the browser by:
   - Capturing a screenshot showing the timestamp
   - Comparing browser timestamp to system clock
   - Or add HUD overlay in `web/index.html` that displays latency

3. **Tune encoder settings** in `webrtc_stream.py`:
   - Bitrate
   - Keyframe interval
   - Buffer sizes

4. **Test with real camera** to verify pipeline:
   ```bash
   export CAMERA_MODE=opencv
   python scripts/webrtc_stream.py
   ```

## Limitations on Mac vs RPI

- **Hardware acceleration**: Mac uses software encoding (slower than RPI hardware encoder)
- **Camera controls**: OpenCV has limited control over exposure/gain vs libcamera
- **Frame synchronization**: No hardware sync line, so frames may drift
- **Performance**: Mac is much faster, so latency may be better than RPI; use for algorithm testing

## Environment Variables

- `CAMERA_MODE`: `auto` (default), `rpi`, `opencv`, or `test`
- `TEST_PATTERN`: `moving_bars` (default), `timestamp`, `color_cycle`, or `checkerboard`

## Next Steps

Once latency is optimized on Mac:

1. Profile CPU/GPU usage during streaming
2. Test at target frame rates (56/90 fps)
3. Measure encoder bitrate vs quality
4. Deploy to RPI and compare actual latency
5. Tune RPI-specific settings (hardware encoder, thermal limits)


