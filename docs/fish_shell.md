# Fish Shell Support

This project works with Fish shell. Use the `.fish` activation script.

## Activating the Virtual Environment

For Fish shell, use:
```fish
source .venv/bin/activate.fish
```

Not:
```fish
source .venv/bin/activate  # This is for bash/sh and won't work in Fish
```

## Quick Reference

```fish
# Activate venv
source .venv/bin/activate.fish

# Deactivate
deactivate

# Install dependencies
make python-deps  # Works in any shell

# Run scripts
python scripts/webrtc_stream.py --host 127.0.0.1 --port 8443
```

## Using Make Commands

The Makefile uses bash for activation internally, so `make` commands work regardless of your shell:
```fish
make python-deps
make stream-webrtc ARGS="--host 127.0.0.1"
```

You don't need to activate the venv manually when using `make` targets.

