#!/usr/bin/env bash
# Build the Vox2Pix client in an already configured, isolated Craftium checkout.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CRAFTIUM_DIR=${1:-"$ROOT/third_party/craftium_vox2pix_floor"}
PATCH="$ROOT/voxwm_gt/patches/lighting_floor.patch"

if [[ ! -f "$CRAFTIUM_DIR/CMakeCache.txt" ]]; then
  printf '%s\n' "Configure CMake in the isolated Craftium checkout first: $CRAFTIUM_DIR" >&2
  exit 2
fi
python3 "$ROOT/scripts/apply_fixed24.py" --craftium-dir "$CRAFTIUM_DIR"
if ! git -C "$CRAFTIUM_DIR" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  git -C "$CRAFTIUM_DIR" apply --check "$PATCH"
  git -C "$CRAFTIUM_DIR" apply "$PATCH"
fi
cmake --build "$CRAFTIUM_DIR" --target luanti -j "${BUILD_THREADS:-4}"
