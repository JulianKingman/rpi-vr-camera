#!/usr/bin/env bash
set -euo pipefail

# List processes using /dev/video*.
if ! command -v lsof >/dev/null 2>&1; then
  echo "lsof not found" >&2
  exit 1
fi

lsof -w /dev/video* 2>/dev/null
