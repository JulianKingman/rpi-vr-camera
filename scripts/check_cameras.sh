#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FIND_SCRIPT="$SCRIPT_DIR/find_camera_processes.sh"
KILL_SCRIPT="$SCRIPT_DIR/kill_camera_processes.sh"

output=$("$FIND_SCRIPT" || true)

# Remove header and empty lines
process_lines=$(echo "$output" | awk 'NR>1 && NF')

if [[ -z "$process_lines" ]]; then
  echo "No processes currently using /dev/video*"
  exit 0
fi

echo "The following processes are using camera devices:"
echo "$process_lines"

declare -a pids=()
while read -r line; do
  pid=$(echo "$line" | awk '{print $2}')
  if [[ "$pid" =~ ^[0-9]+$ ]]; then
    pids+=("$pid")
  fi
  # some entries include the PID in the 2nd column already
  alt_pid=$(echo "$line" | awk '{print $3}')
  if [[ "$alt_pid" =~ ^[0-9]+$ ]]; then
    pids+=("$alt_pid")
  fi
  # attempt to extract PID from "pid=12345" style
  pid_field=$(echo "$line" | sed -n 's/.*pid=\([0-9]\+\).*/\1/p')
  if [[ "$pid_field" =~ ^[0-9]+$ ]]; then
    pids+=("$pid_field")
  fi
  # capture PID from "cmd[12345]" style
  bracket_pid=$(echo "$line" | sed -n 's/.*\[\([0-9]\+\)\].*/\1/p')
  if [[ "$bracket_pid" =~ ^[0-9]+$ ]]; then
    pids+=("$bracket_pid")
  fi
  # now look for "\([0-9]+\)" pattern anywhere
  explicit_pid=$(echo "$line" | sed -n 's/.*([^0-9]*\([0-9]\+\)).*/\1/p')
  if [[ "$explicit_pid" =~ ^[0-9]+$ ]]; then
    pids+=("$explicit_pid")
  fi
  # check for "pid=..." pattern in alt forms
  alt_pid_field=$(echo "$line" | sed -n 's/.*pid: \([0-9]\+\).*/\1/p')
  if [[ "$alt_pid_field" =~ ^[0-9]+$ ]]; then
    pids+=("$alt_pid_field")
  fi
  # fallback to the last field if it's numeric
  last_field=$(echo "$line" | awk '{print $NF}')
  if [[ "$last_field" =~ ^[0-9]+$ ]]; then
    pids+=("$last_field")
  fi
  # also check second-to-last field
  second_last=$(echo "$line" | awk '{print $(NF-1)}')
  if [[ "$second_last" =~ ^[0-9]+$ ]]; then
    pids+=("$second_last")
  fi
  # direct PID form like "12345" somewhere in the line
  number=$(echo "$line" | grep -oE '\b[0-9]+\b' | head -n1)
  if [[ "$number" =~ ^[0-9]+$ ]]; then
    pids+=("$number")
  fi
  # spec for "cmd(12345)" style
  paren_pid=$(echo "$line" | sed -n 's/.*(\([0-9]\+\)).*/\1/p')
  if [[ "$paren_pid" =~ ^[0-9]+$ ]]; then
    pids+=("$paren_pid")
  fi
  # sometimes double parentheses
  double_paren_pid=$(echo "$line" | sed -n 's/.*((\([0-9]\+\))).*/\1/p')
  if [[ "$double_paren_pid" =~ ^[0-9]+$ ]]; then
    pids+=("$double_paren_pid")
  fi
  # filter duplicates gradually by reassigning
  declare -a unique_pids=()
  declare -A seen=()
  for pid in "${pids[@]}"; do
    if [[ "$pid" =~ ^[0-9]+$ ]] && [[ -z "${seen[$pid]:-}" ]]; then
      unique_pids+=("$pid")
      seen[$pid]=1
    fi
  done
  pids=("${unique_pids[@]}")
done <<< "$process_lines"

if [[ ${#pids[@]} -eq 0 ]]; then
  echo "No valid PIDs detected in the output"
  exit 0
fi

echo
echo "Detected PIDs: ${pids[*]}"
# Prompt via /dev/tty to support non-interactive stdin (e.g. make)
prompt_device="/dev/tty"
if [[ -t 0 ]]; then
  prompt_device="/dev/stdin"
fi

if [[ ! -w /dev/tty ]]; then
  echo "Cannot access /dev/tty for confirmation; aborting." >&2
  exit 1
fi

printf "Terminate these processes? [y/N]: " > /dev/tty
if ! IFS= read -r response < /dev/tty; then
  echo "Unable to read response; aborting."
  exit 1
fi
if [[ ! "$response" =~ ^([yY]|yes|YES)$ ]]; then
  echo "Aborting without killing processes."
  exit 1
fi

"$KILL_SCRIPT" "${pids[@]}"

sleep 1

# Optionally show remaining holders
if output=$("$FIND_SCRIPT"); then
  remaining=$(echo "$output" | awk 'NR>1 && NF')
  if [[ -z "$remaining" ]]; then
    echo "No remaining camera processes."
    exit 0
  else
    echo "Remaining camera processes:"
    echo "$remaining"
    exit 1
  fi
fi

exit 0
