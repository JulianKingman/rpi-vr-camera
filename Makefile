# Helper targets for rpi-vr-camera development workflow

PYTHON ?= python3
VENV_PY := .venv/bin/python3
CAMERA ?= 0
RESOLUTION ?= 1536x864
FRAMERATE ?= 120
WIDTH ?= 1536
HEIGHT ?= 864
OUTPUT ?= capture.h264
PTS ?= capture.pts
PREVIEW ?= null
METADATA ?= 0
SAVE_PTS ?= 1
FFMPEG ?= ffmpeg
SOURCE ?= $(OUTPUT)
TARGET ?= $(basename $(SOURCE)).mp4
TIMEOUT ?= 5000
KEYPRESS ?= 1
OUT_ROOT ?= capture-output
LABEL ?= $(shell date +%Y%m%d_%H%M%S)

CAPTURE_TIMEOUT := --timeout $(TIMEOUT)
CAPTURE_KEYPRESS := $(if $(filter 0,$(KEYPRESS)),,--keypress)
SETPTS_EXPR := N/($(FRAMERATE)*TB)

.PHONY: help system-deps python-deps preview capture capture-left capture-right convert convert-left convert-right capture-stereo calibration-ui stream-preview stream-cast stream-webrtc

help:
	@echo "Available targets:"
	@echo "  make system-deps      # install apt packages (run with sudo)"
	@echo "  make python-deps      # create/update Python venv"
	@echo "  make preview          # preview camera feed"
	@echo "  make capture          # capture from camera (5s default; tweak TIMEOUT/KEYPRESS)"
	@echo "  make calibration-ui   # launch Qt calibration interface"
	@echo "  make stream-preview   # live side-by-side stereo preview (Picamera2)"
	@echo "  make stream-cast      # preview + network cast via ffmpeg"
	@echo "  make stream-webrtc    # launch WebRTC/WebXR server (requires browser client)"
	@echo "Variables: CAMERA, RESOLUTION, PREVIEW, METADATA, FRAMERATE, WIDTH, HEIGHT, OUTPUT, PTS"

system-deps:
	@echo "Installing system dependencies (requires sudo)..."
	sudo ./scripts/install_system_deps.sh

python-deps:
	@echo "Bootstrapping Python environment..."
	PYTHON=$(PYTHON) ./scripts/install_python_env.sh

preview:
	@if [ ! -x .venv/bin/python3 ]; then \
		echo "Virtualenv missing. Run 'make python-deps' first."; \
		exit 1; \
	fi
	source .venv/bin/activate && ./scripts/preview.py --camera $(CAMERA) --preview $(PREVIEW) --resolution $(RESOLUTION)$(if $(filter 0,$(METADATA)),, --metadata)

capture:
	@echo "Capturing camera $(CAMERA) at $(WIDTH)x$(HEIGHT)@$(FRAMERATE) -> $(OUTPUT)"
	@mkdir -p $(dir $(OUTPUT))
	@mp4=$(basename $(OUTPUT)).mp4; \
	rpicam-vid --camera $(CAMERA) --framerate $(FRAMERATE) --width $(WIDTH) --height $(HEIGHT) \
	  --codec h264 --inline --nopreview $(CAPTURE_TIMEOUT) $(CAPTURE_KEYPRESS) \
	  --output $(OUTPUT)$(if $(filter 0,$(SAVE_PTS)),, --save-pts $(PTS)); \
	echo "Converting $(OUTPUT) -> $$mp4"; \
	$(FFMPEG) -y -f h264 -i $(OUTPUT) -vf "setpts=$(SETPTS_EXPR)" \
	  -r $(FRAMERATE) -c:v libx264 -preset ultrafast -pix_fmt yuv420p -movflags +faststart $$mp4 >/dev/null

capture-left: LABEL ?= $(shell date +%Y%m%d_%H%M%S)_left
capture-left: OUTPUT = $(OUT_ROOT)/left/$(if $(STEREO_LABEL),$(STEREO_LABEL),$(LABEL))/left.h264
capture-left: PTS = $(OUT_ROOT)/left/$(if $(STEREO_LABEL),$(STEREO_LABEL),$(LABEL))/left.pts
capture-left: CAMERA := 0
capture-left: capture

capture-right: LABEL ?= $(shell date +%Y%m%d_%H%M%S)_right
capture-right: OUTPUT = $(OUT_ROOT)/right/$(if $(STEREO_LABEL),$(STEREO_LABEL),$(LABEL))/right.h264
capture-right: PTS = $(OUT_ROOT)/right/$(if $(STEREO_LABEL),$(STEREO_LABEL),$(LABEL))/right.pts
capture-right: CAMERA := 1
capture-right: capture

capture-stereo:
	@label=$$(date +%Y%m%d_%H%M%S); \
	echo "Stereo capture label=$$label"; \
	$(MAKE) capture-left LABEL="$$label" STEREO_LABEL="$$label"; \
	$(MAKE) capture-right LABEL="$$label" STEREO_LABEL="$$label"; \
	echo "Stereo capture completed: label=$$label"

convert:
	@if [ ! -f "$(SOURCE)" ]; then \
		echo "Missing input stream: $(SOURCE)"; \
		exit 1; \
	fi
	@mkdir -p $(dir $(TARGET))
	$(FFMPEG) -y -f h264 -i $(SOURCE) -vf "setpts=$(SETPTS_EXPR)" \
	  -r $(FRAMERATE) -c:v libx264 -preset ultrafast -pix_fmt yuv420p -movflags +faststart $(TARGET)

convert-left:
	@latest=$$(ls -1dt $(OUT_ROOT)/left/* 2>/dev/null | head -n 1); \
	if [ -z "$$latest" ]; then \
		echo "No left captures found under $(OUT_ROOT)/left/"; \
		exit 1; \
	fi; \
	src="$$latest/left.h264"; \
	if [ ! -f "$$src" ]; then \
		echo "Missing $$src"; \
		exit 1; \
	fi; \
	target="$$latest/left.mp4"; \
	$(MAKE) convert SOURCE="$$src" TARGET="$$target"; \
	echo "Wrote $$target"

convert-right:
	@latest=$$(ls -1dt $(OUT_ROOT)/right/* 2>/dev/null | head -n 1); \
	if [ -z "$$latest" ]; then \
		echo "No right captures found under $(OUT_ROOT)/right/"; \
		exit 1; \
	fi; \
	src="$$latest/right.h264"; \
	if [ ! -f "$$src" ]; then \
		echo "Missing $$src"; \
		exit 1; \
	fi; \
	target="$$latest/right.mp4"; \
	$(MAKE) convert SOURCE="$$src" TARGET="$$target"; \
	echo "Wrote $$target"

calibration-ui:
	@if [ ! -x .venv/bin/python3 ]; then \
		echo "Virtualenv missing. Run 'make python-deps' first."; \
		exit 1; \
	fi
	. .venv/bin/activate && python scripts/calibration_gui.py

stream-preview:
	@if [ ! -x .venv/bin/python3 ]; then \
		echo "Virtualenv missing. Run 'make python-deps' first."; \
		exit 1; \
	fi
	. .venv/bin/activate && python scripts/stream_stereo.py

stream-cast:
	@if [ ! -x .venv/bin/python3 ]; then \
		echo "Virtualenv missing. Run 'make python-deps' first."; \
		exit 1; \
	fi
	. .venv/bin/activate && python scripts/stream_cast.py $(ARGS)

stream-webrtc:
	@if [ ! -x .venv/bin/python3 ]; then \
		echo "Virtualenv missing. Run 'make python-deps' first."; \
		exit 1; \
	fi
	. .venv/bin/activate && python scripts/webrtc_stream.py $(ARGS)
