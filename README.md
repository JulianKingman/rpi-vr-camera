# rpi-vr-camera

## Goal
Low-latency stereo camera capture and VR streaming pipeline tuned for Raspberry Pi 5 hardware.

## Hardware & OS Prep
- Raspberry Pi 5, dual IMX708 camera modules, active cooling, low-latency network (wired preferred).
- Raspberry Pi OS Bookworm 64-bit with latest firmware (`sudo rpi-update` on a test SD card first).
- Edit `/boot/firmware/config.txt`:
  - `camera_auto_detect=0`
  - `dtoverlay=imx708,cam0`
  - `dtoverlay=imx708,cam1`
- Enable libcamera stack (`sudo raspi-config nonint do_camera 0`) and reboot.

## Development Dependencies
- Build essentials, git, cmake/meson as needed.
- `libcamera`, `libcamera-apps`, `gstreamer1.0-*` (including `gstreamer1.0-libav`, `gstreamer1.0-tools`, `gstreamer1.0-plugins-{good,bad}`).
- Python 3.11+ with `aiortc`, `opencv-python`, `numpy`, `picamera2` for rapid prototyping.
- Optional: `nv12` aware viewers (e.g. `ffplay`, `gst-launch-1.0` on a development laptop) for debugging streams.
## Installing Dependencies
All commands assume Raspberry Pi OS Bookworm 64-bit on a Pi 5.

```bash
# System toolchain, libcamera, and GStreamer stack
sudo apt update
sudo apt install -y \
  build-essential git cmake meson ninja-build pkg-config \
  libcap-dev \
  libcamera-dev libcamera-apps python3-libcamera python3-kms++ python3-prctl \
  gstreamer1.0-tools gstreamer1.0-libav \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  gstreamer1.0-alsa gstreamer1.0-plugins-base \
  ffmpeg

# Python environment (use a venv to avoid polluting system packages)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install aiortc opencv-python numpy picamera2 pyyaml PySide6
```

- The `make python-deps` helper writes the `.pth` link automatically; if you ever recreate the venv manually, run `echo '/usr/lib/python3/dist-packages' >> .venv/lib/python3.*/site-packages/rpi_libcamera.pth`.
- If `aiortc` build fails, install the extra headers: `sudo apt install libssl-dev libffi-dev`.
- If `picamera2` fails while building `python-prctl`, ensure `libcap-dev` is present (`sudo apt install libcap-dev`).
- If the legacy `libcamera-hello` binary is missing, use the renamed `rpicam-hello` tools (`sudo apt install rpicam-apps`) and verify `/usr/bin` is on your `PATH`.
- To leave the virtual environment when you are done, run `deactivate`.

## Quick Start Commands
Run everything from the project root:

```bash
sudo make system-deps    # apt packages (libcamera, gstreamer, etc.)
make python-deps         # create/update .venv and Python requirements
make preview CAMERA=0 METADATA=1 PREVIEW=null
make capture-left        # writes capture-output/left/<timestamp>/left.h264 (+ .pts) + left.mp4
make capture-right       # writes capture-output/right/<timestamp>/right.h264 (+ .pts) + right.mp4
make capture-stereo      # launches left/right in parallel with shared timestamp label
make calibration-ui      # live Qt alignment UI for tweaking rotation/crop/offsets
make stream-preview      # real-time side-by-side preview using calibration settings
OUT_ROOT=test-output make capture-left  # store captures under test-output/...
```

- `make help` lists all available targets and tunable variables (override defaults like `FRAMERATE=56` or `OUTPUT=myclip.h264`).
- Use `make capture CAMERA=0 OUTPUT=capture-output/custom/clip.mp4 PTS=capture-output/custom/clip.pts` for ad-hoc names (override the base folder with `OUT_ROOT=...`).
- `make capture-stereo` records left then right under a shared timestamp label (sequential start for now).
- Default `TIMEOUT=5000` yields a 5 s clip; try `TIMEOUT=30000 make capture-left` for half a minute, or `TIMEOUT=0 KEYPRESS=1` to keep recording until Enter.
- Add `PREVIEW=drm` to the preview target when running on a Wayland desktop to see the live feed.
- Run `make stream-preview` for a live side-by-side window honouring the current calibration; use `--headless` if no GUI is available.
- Run `make calibration-ui` for a live Qt preview where you can tweak rotation, flips, crops, and offsets (writes back to `config/camera_profiles.yaml`).
- Captures land under `capture-output/<left|right>/<YYYYMMDD_HHMMSS>` with raw `.h264`, matching `.mp4`, and `.pts`; override the base folder via `OUT_ROOT=...`.

## Calibration UI (Qt)
Launch the desktop calibration tool to visualise both camera streams side by side and interactively tweak their transforms:

```bash
make calibration-ui
```

- Rotation, horizontal/vertical flips, crop size, and XY offsets update the preview immediately.
- Press **Save Calibration** to write the values back into `config/camera_profiles.yaml`; subsequent scripts (capture, streaming) will pick up the new defaults.
- Close the window to release both cameras before running other capture commands.

## Prototype 120 FPS Capture
Aim for the 1536×864@120 mode first to validate dual-camera throughput.

```bash
# Left camera (camera index 0)
rpicam-vid --camera 0 --framerate 120 --width 1536 --height 864 \
  --codec h264 --inline --nopreview --timeout 5000 \
  --output left.mp4 --save-pts left.pts

# Right camera (camera index 1)
rpicam-vid --camera 1 --framerate 120 --width 1536 --height 864 \
  --codec h264 --inline --nopreview --timeout 5000 \
  --output right.mp4 --save-pts right.pts
```

- Pair `--timeout 0` with `--keypress` to stream until you hit Enter (`TIMEOUT=0 KEYPRESS=1 make capture-left` mirrors this); tweaking `TIMEOUT=<ms>` sets clip length, while per-run folders under `capture-output/<left|right>/` keep each run separated.
- Only add `--segment <seconds>` if you truly need rolling chunk files—otherwise you’ll end up with hundreds of tiny `.h264` segments.
- The `.pts` sidecar stores per-frame timestamps (needed for muxing or latency analysis). Skip it with `SAVE_PTS=0 make capture-left` if you just need video frames.
- Output `.mp4` duration honours the `FRAMERATE` value; adjust it before capture if you switch modes (the converter re-synthesises timestamps internally).
- Files are stored under `capture-output/<left|right>/<timestamp>/`; you can override the root folder with `OUT_ROOT`.
- To re-enable a live window, run from the Pi’s Wayland console and use `--preview drm,0,0,640,360`. On older stacks try `QT_QPA_PLATFORM=wayland rpicam-vid ... --preview 0,0,640,360`. Avoid X11 forwarding/headless X, which triggers the “failed to import fd” crash at 120 fps. Add `--info-text "%frame%"` for an on-screen fps counter.
- Monitor dropped frames in the terminal output; any sustained underrun suggests power, thermal, or I/O contention.
- Use `--listen` on one camera and `rpicam-vid --listen --port 8888 --framerate 120 ...` on another Pi to stream over the network if you need remote inspection.
- For Oculus Quest 2 testing, plan to feed the encoded stream into a WebRTC/WebXR client; target <15 ms glass-to-glass by minimizing buffering in the transport stage.


## Implementation Roadmap
1. **Dual camera capture**
   - Verify both modules enumerate via `rpicam-hello --list-cameras` (or `libcamera-hello` on older images).
   - Use `scripts/list_cameras.py` (or `make preview CAMERA=0 METADATA=1 PREVIEW=null`) to inspect per-sensor modes and controls.
   - `make preview CAMERA=0 METADATA=1` (and `CAMERA=1`) starts a 1536×864@120 preview with optional rotation, crop, and live metadata output for debugging.
   - `make stream-preview` (Picamera2) shows both calibrated feeds in real time for quick stereo verification.
   - `make capture-stereo` records both sensors in parallel, storing synchronized outputs under a shared timestamp label.
   - Next: extend preview into a dual-feed capture node that pipes synchronized frames into shared queues while logging exposure/gain/frame time.
2. **Orientation, crop, and calibration**
   - Apply rotation and crop in GPU-friendly pipeline (`libcamera` transform flags or `GStreamer` `videoflip`, `videocrop`).
   - Maintain per-camera defaults in `config/camera_profiles.yaml` (resolution, flips, crop).
   - Run `make calibration-ui` (`scripts/calibration_gui.py`) to interactively tune rotation, flips, crops, and offsets and save back to the profiles file.
   - Use `scripts/capture_calibration.py --camera 0 --profile cam0` (and `--camera 1`) to collect calibration frames into `data/calibration/`; feed captures into OpenCV `stereoCalibrate` and persist results to YAML for runtime use.
   - Implement lens shading / color balance harmonization if mismatched.
3. **Framerate and exposure controls**
   - Surface controls for `FrameDurationLimits`, `AnalogueGain`, `ExposureTime` via libcamera ControlList (task 3).
   - Use `scripts/set_controls.py --camera 0 --fps 120 --exposure 8000 --gain 1.0 --disable-ae` to lock in capture parameters; extend to named presets in `config/camera_profiles.yaml`.
   - Monitor dropped frames and thermal throttling; add warning overlays/logging.
4. **Stereo stitching & VR formatting**
   - Prototype GPU-accelerated stereo compositor: rectify -> stitch (side-by-side or equirectangular depending on headset target).
   - Evaluate `OpenCV CUDA` vs `Vulkan` vs shader-based pipeline for Pi 5 VideoCore VII.
   - Output VR-ready frame packing standard (initially side-by-side 1:1, expand to OpenXR-compliant format).
5. **WebRTC transport**
   - Use `GStreamer webrtcbin` or `aiortc` with hardware encode (H.265/AVC via `v4l2h264enc`/`v4l2hevcenc`) with a clear <15 ms glass-to-glass target (150 ms is too high and triggers discomfort).
   - Implement signaling server (simple WebSocket) and browser-based VR viewer (WebXR) or native OpenXR client.
   - Add adaptive bitrate / dynamic framerate controls.
6. **Performance tuning & validation**
   - Target <11 ms pipeline latency at 90 Hz; profile CPU/GPU usage (perf, `libcamera-vid --save-pts`).
   - Stress-test thermal limits; enable fan curve.
   - Automate integration tests that capture, encode, and loopback decode to ensure synchronization.

- `make preview CAMERA=0 METADATA=1` – run the Picamera2 preview without retyping the full flag list.
- `make stream-preview` – live calibrated side-by-side preview (Picamera2/OpenCV) for stereo verification.
- `make calibration-ui` – Qt interface for per-camera rotation/flip/crop tuning with live video.
- `make convert-left` / `make convert-right` – regenerate MP4s from stored raw streams if you rerun the converter.
- `make capture-left` / `make capture-right` – grab one camera stream until you hit Enter.
- `scripts/list_cameras.py --show-controls` – enumerate sensors, supported modes, and control metadata.
- `scripts/set-controls.py --camera 0 --fps 120 --exposure 10000` – tweak capture timing on the fly.
- `config/camera_profiles.yaml` – central place for per-camera transforms, crops, and future presets.

## Running (Design Phase)
- Implementation in progress; no runnable pipeline yet.
- Once capture node exists, document commands here (`./scripts/preview.py --camera left`, etc.).

## Open Questions
- Confirm hardware sync line availability for perfect frame alignment.
- Decide on final VR target (e.g., Quest via WebXR vs PC VR via WebRTC/OpenXR).
- Determine stitching approach (rectilinear vs equirectangular) based on lens FOV and headset expectations.
- Validate that Raspberry Pi hardware encoder sustains 90 Hz at chosen resolution; otherwise consider foveated rendering or resolution reduction.

## Next Actions
- Prototype dual capture with `picamera2` and confirm spec framerates.
- Draft calibration data pipeline and YAML schema.
- Spike WebRTC transport with test pattern feed to measure end-to-end latency.
