#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "[prompt-test] Checking for camera holders..."
"$SCRIPT_DIR/check_cameras.sh"

echo "[prompt-test] Cameras cleared. (No pipeline launched in this test.)"
