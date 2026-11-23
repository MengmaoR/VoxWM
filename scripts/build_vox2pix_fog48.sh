#!/usr/bin/env bash
# Build independent native draw/fog ranges in an isolated configured checkout.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CRAFTIUM_DIR=${1:-"$ROOT/third_party/craftium_vox2pix_fog48"}
ABSOLUTE_PATCH="$ROOT/voxwm_gt/patches/fog_absolute.patch"
FOG_PATCH="$ROOT/voxwm_gt/patches/fog_range.patch"

if [[ ! -f "$CRAFTIUM_DIR/CMakeCache.txt" ]]; then
  printf 'Configure CMake in the isolated Craftium checkout first: %s\n' "$CRAFTIUM_DIR" >&2
  exit 2
fi

if ! git -C "$CRAFTIUM_DIR" apply --reverse --check "$ABSOLUTE_PATCH" >/dev/null 2>&1; then
  if ! git -C "$CRAFTIUM_DIR" apply --reverse --check "$FOG_PATCH" >/dev/null 2>&1; then
    # Builds the existing capture/material/native-fog layers first.
    bash "$ROOT/scripts/build_vox2pix_fog.sh" "$CRAFTIUM_DIR"
  fi
  git -C "$CRAFTIUM_DIR" apply --check "$ABSOLUTE_PATCH"
  git -C "$CRAFTIUM_DIR" apply "$ABSOLUTE_PATCH"
fi
cmake --build "$CRAFTIUM_DIR" --target luanti -j "${BUILD_THREADS:-4}"
