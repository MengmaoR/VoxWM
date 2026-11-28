#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${1:?Usage: monitor_vae_curated.sh OUTPUT_DIR [--port PORT]}"
output=$1
shift
exec "${VAE_MONITOR_PYTHON:-.venv/bin/python}" -m voxel_vae.training_monitor \
    --output-dir "$output" "$@"
