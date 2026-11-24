#!/usr/bin/env bash
# Thin foreground launcher; tmux can keep the whole collector alive across SSH disconnects.
set -euo pipefail
VOX2PIX_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$VOX2PIX_ROOT"
exec "${VOXWM_PYTHON:-.venv/bin/python}" - "$@" <<'PY'
import json
import os
from pathlib import Path
import sys

environment_file = os.environ.get("VOXWM_ENV_CONFIG")
if environment_file:
    environment = json.loads(Path(environment_file).read_text())["environment"]
    os.environ.update(environment)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.execv(sys.executable, [
    sys.executable, "-u", "-m", "vox2pix_data", "collect", "--mode", "production",
    "--dataset", os.environ.get("VOX2PIX_DATASET", "outputs/vox2pix_50k_rgb32"),
    "--scenes", "10000", "--workers", os.environ.get("VOX2PIX_WORKERS", "8"),
    "--renderer", "software",
    "--craftium-dir", os.environ.get("VOX2PIX_CRAFTIUM", "third_party/craftium_vox2pix_fog48"),
    *sys.argv[1:],
])
PY
