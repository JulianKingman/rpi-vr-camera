# Helper targets for rpi-vr-camera development workflow

PYTHON ?= python3
VENV_PY := .venv/bin/python3
CAMERA ?= 0
RESOLUTION ?= 2304x1296
FRAMERATE ?= 56
WIDTH ?= 2304
HEIGHT ?= 1296
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

.PHONY: help system-deps python-deps preview capture capture-left capture-right convert convert-left convert-right capture-stereo calibration-ui stream-preview stream-cast stream-webrtc stream-webrtc-gst prompt-test tls-certs install-service uninstall-service service-logs service-status install-mdns uninstall-mdns

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
	@echo "  make stream-webrtc-gst # launch GStreamer-based WebRTC pipeline (option 3 prototype)"
	@echo "  make tls-certs        # interactive helper for HTTPS certificates"
	@echo "  make install-service  # install, enable, and start the systemd service"
	@echo "  make uninstall-service # stop, disable, and remove the systemd service"
	@echo "  make service-logs     # view live journal logs for the service"
	@echo "  make service-status   # show systemd service status"
	@echo "  make install-mdns     # advertise as rpi-vr-camera.local via Avahi/mDNS"
	@echo "  make uninstall-mdns   # remove mDNS advertisement"
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

DEFAULT_GST_HOST ?= 0.0.0.0
DEFAULT_GST_PORT ?= 8443
DEFAULT_GST_FRAMERATE ?= 72
DEFAULT_GST_STUN ?= stun://stun.l.google.com:19302

GST_ARGS = --host $(DEFAULT_GST_HOST) --port $(DEFAULT_GST_PORT) --framerate $(DEFAULT_GST_FRAMERATE) --stun $(DEFAULT_GST_STUN)

stream-webrtc-gst:
	@scripts/check_cameras.sh || exit $$?
	@if [ ! -x .venv/bin/python3 ]; then \
		echo "Virtualenv missing. Run 'make python-deps' first."; \
		exit 1; \
	fi
	. .venv/bin/activate && python scripts/webrtc_gstreamer.py $(GST_ARGS)

prompt-test:
	@echo "Running prompt test Makefile target..."
	@scripts/prompt_test.sh

tls-certs:
	$(PYTHON) scripts/generate_tls_certs.py

# --- systemd service targets ---

SERVICE_NAME := rpi-vr-camera
SERVICE_FILE := config/rpi-vr-camera.service
SYSTEMD_DIR := /etc/systemd/system

install-service:
	@if [ ! -f $(SERVICE_FILE) ]; then \
		echo "Service file not found: $(SERVICE_FILE)"; \
		exit 1; \
	fi
	@if [ ! -x .venv/bin/python3 ]; then \
		echo "Virtualenv missing. Run 'make python-deps' first."; \
		exit 1; \
	fi
	@echo "Installing systemd service $(SERVICE_NAME)..."
	sudo cp $(SERVICE_FILE) $(SYSTEMD_DIR)/$(SERVICE_NAME).service
	sudo systemctl daemon-reload
	sudo systemctl enable $(SERVICE_NAME).service
	sudo systemctl start $(SERVICE_NAME).service
	@echo "Service installed and started. Check status with: make service-status"

uninstall-service:
	@echo "Stopping and removing systemd service $(SERVICE_NAME)..."
	-sudo systemctl stop $(SERVICE_NAME).service
	-sudo systemctl disable $(SERVICE_NAME).service
	-sudo rm -f $(SYSTEMD_DIR)/$(SERVICE_NAME).service
	sudo systemctl daemon-reload
	@echo "Service removed."

service-logs:
	sudo journalctl -u $(SERVICE_NAME).service -f

service-status:
	sudo systemctl status $(SERVICE_NAME).service

# --- mDNS/Avahi targets ---

MDNS_SERVICE_SRC := config/rpi-vr-camera-mdns.service
MDNS_SERVICE_DST := /etc/avahi/services/rpi-vr-camera.service
MDNS_HOSTNAME    := rpi-vr-camera

install-mdns:
	@echo "==> Installing mDNS/Avahi service advertisement..."
	@if ! dpkg -s avahi-daemon >/dev/null 2>&1; then \
		echo "    Installing avahi-daemon..."; \
		sudo apt-get update -qq && sudo apt-get install -y -qq avahi-daemon; \
	else \
		echo "    avahi-daemon already installed."; \
	fi
	@echo "    Copying service file to $(MDNS_SERVICE_DST)"
	sudo cp $(MDNS_SERVICE_SRC) $(MDNS_SERVICE_DST)
	@current=$$(hostname); \
	if [ "$$current" != "$(MDNS_HOSTNAME)" ]; then \
		echo "    Setting hostname to $(MDNS_HOSTNAME) (was $$current)..."; \
		sudo hostnamectl set-hostname $(MDNS_HOSTNAME); \
	else \
		echo "    Hostname already set to $(MDNS_HOSTNAME)."; \
	fi
	sudo systemctl enable avahi-daemon
	sudo systemctl restart avahi-daemon
	@echo "==> Done. This Pi is now reachable at https://$(MDNS_HOSTNAME).local:8443"

uninstall-mdns:
	@echo "==> Removing mDNS/Avahi service advertisement..."
	@if [ -f $(MDNS_SERVICE_DST) ]; then \
		sudo rm -f $(MDNS_SERVICE_DST); \
		echo "    Removed $(MDNS_SERVICE_DST)"; \
	else \
		echo "    $(MDNS_SERVICE_DST) not found (already removed)."; \
	fi
	sudo systemctl restart avahi-daemon
	@echo "==> Done. Service advertisement removed."
	@echo "    Note: hostname was NOT reverted. Use 'sudo hostnamectl set-hostname <name>' to change it."
