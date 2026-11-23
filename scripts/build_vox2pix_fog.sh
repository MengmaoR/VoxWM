#!/usr/bin/env bash
# Build opt-in native fog range controls in an isolated configured checkout.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CRAFTIUM_DIR=${1:-"$ROOT/third_party/craftium_vox2pix_fog"}
FOG_PATCH="$ROOT/voxwm_gt/patches/fog_range.patch"
MATERIAL_PATCH="$ROOT/voxwm_gt/patches/material_ambient.patch"
LEGACY_PATCH="$ROOT/voxwm_gt/patches/lighting_floor.patch"

if [[ ! -f "$CRAFTIUM_DIR/CMakeCache.txt" ]]; then
  printf 'Configure CMake in the isolated Craftium checkout first: %s\n' "$CRAFTIUM_DIR" >&2
  exit 2
fi

if ! git -C "$CRAFTIUM_DIR" apply --reverse --check "$FOG_PATCH" >/dev/null 2>&1; then
  python3 "$ROOT/scripts/apply_fixed24.py" --craftium-dir "$CRAFTIUM_DIR"
  if ! git -C "$CRAFTIUM_DIR" apply --reverse --check "$MATERIAL_PATCH" >/dev/null 2>&1; then
    if ! git -C "$CRAFTIUM_DIR" apply --reverse --check "$LEGACY_PATCH" >/dev/null 2>&1; then
      git -C "$CRAFTIUM_DIR" apply --check "$LEGACY_PATCH"
      git -C "$CRAFTIUM_DIR" apply "$LEGACY_PATCH"
    fi
    git -C "$CRAFTIUM_DIR" apply --check "$MATERIAL_PATCH"
    git -C "$CRAFTIUM_DIR" apply "$MATERIAL_PATCH"
  fi
  git -C "$CRAFTIUM_DIR" apply --check "$FOG_PATCH"
  git -C "$CRAFTIUM_DIR" apply "$FOG_PATCH"
fi
cmake --build "$CRAFTIUM_DIR" --target luanti -j "${BUILD_THREADS:-4}"
