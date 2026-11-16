#!/usr/bin/env bash
# User-run incremental engine build. No model dependencies or system installation.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CRAFTIUM_DIR=${1:-"$ROOT/third_party/craftium"}
python3 "$ROOT/scripts/apply_fixed24.py" --craftium-dir "$CRAFTIUM_DIR"
cmake --build "$CRAFTIUM_DIR" --target luanti -j "${BUILD_THREADS:-4}"
