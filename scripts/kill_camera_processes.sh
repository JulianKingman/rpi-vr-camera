#!/usr/bin/env bash
# Terminates the provided PIDs, escalating to SIGKILL if necessary.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 PID [PID ...]" >&2
  exit 1
fi

for pid in "$@"; do
  if [[ "$pid" =~ ^[0-9]+$ ]]; then
    if kill -0 "$pid" 2>/dev/null; then
      echo "Terminating PID $pid"
      if ! kill "$pid" 2>/dev/null; then
        if command -v sudo >/dev/null 2>&1; then
          sudo kill "$pid" 2>/dev/null || true
        fi
      fi
      sleep 0.5
      if kill -0 "$pid" 2>/dev/null; then
        echo "PID $pid still alive; sending SIGKILL"
        if ! kill -9 "$pid" 2>/dev/null; then
          if command -v sudo >/dev/null 2>&1; then
            sudo kill -9 "$pid" 2>/dev/null || true
          fi
        fi
        sleep 0.2
      fi
    fi
  fi
done
