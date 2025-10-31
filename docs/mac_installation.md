# macOS Installation Guide

Installation steps for developing on macOS (without Raspberry Pi hardware).

## Prerequisites

1. **Python 3.11+** (check with `python3 --version`)
   - macOS usually includes Python 3, or install via Homebrew: `brew install python3`

2. **Homebrew** (recommended for optional tools)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

## Installation Steps

### 1. Clone/Setup Project
```bash
cd /path/to/rpi-vr-camera
```

### 2. Create Python Virtual Environment
```bash
python3 -m venv .venv

# Activate (bash/zsh)
source .venv/bin/activate

# Or for Fish shell
source .venv/bin/activate.fish
```

### 3. Install Python Dependencies

**Essential packages** (required for streaming):
```bash
pip install --upgrade pip wheel
pip install aiortc aiohttp opencv-python numpy pyyaml
```

**Optional packages**:
```bash
# For GUI calibration tool (Qt)
pip install PySide6

# Note: picamera2 is RPI-only and will fail on Mac - that's expected!
```

### 4. Optional: Install ffmpeg (for video conversion/testing)

Via Homebrew:
```bash
brew install ffmpeg
```

Or download from [ffmpeg.org](https://ffmpeg.org/download.html)

### 5. Verify Installation

Test the camera adapter:
```bash
python scripts/test_mac_camera.py
```

If successful, you should see:
- Available cameras listed
- OpenCV camera test (if camera available)
- Test pattern generator working

## Quick Test

Test WebRTC streaming with test patterns:
```bash
export CAMERA_MODE=test
export TEST_PATTERN=timestamp
python scripts/webrtc_stream.py --host 127.0.0.1 --port 8443
```

Then open `http://localhost:8443` in your browser.

## Troubleshooting

### `opencv-python` installation fails
- Ensure you have Python 3.11+ (not 3.12+ may have issues)
- Try: `pip install --upgrade pip setuptools wheel`
- On Apple Silicon: ensure you're using native Python (not Rosetta)

### Camera not detected
- Grant camera permissions: System Settings → Privacy & Security → Camera
- Check available cameras: `python -c "from camera_adapter import list_available_cameras; print(list_available_cameras())"`

### `aiortc` build fails
- Install OpenSSL: `brew install openssl`
- May need: `export LDFLAGS="-L$(brew --prefix openssl)/lib"` and `export CPPFLAGS="-I$(brew --prefix openssl)/include"`

### Import errors for picamera2/libcamera
- These are RPI-only modules - expected to fail on Mac
- The adapter automatically uses OpenCV/test patterns instead

## What's Different on Mac?

- **No hardware encoder**: Uses software encoding (slower but fine for testing)
- **No libcamera**: Uses OpenCV VideoCapture instead
- **No picamera2**: Uses camera adapter with OpenCV backend
- **No GStreamer**: Not needed for basic streaming (aiortc handles encoding)

## Next Steps

See [mac_testing.md](./mac_testing.md) for how to test the streaming pipeline.

