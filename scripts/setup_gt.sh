#!/usr/bin/env bash
# User-run local setup. Installs only the independent native Craftium GT environment.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CRAFTIUM_DIR="$ROOT/third_party/craftium"
GT_ENV="$ROOT/.venv"
PIN=ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49
while [[ $# -gt 0 ]]; do
    case "$1" in
        --craftium-dir) CRAFTIUM_DIR=${2:?missing path}; shift 2 ;;
        -h|--help)
            printf 'Usage: bash scripts/setup_gt.sh [--craftium-dir PATH]\nRequires uv, git, CMake, make/C++ and native Craftium libraries. Creates a local Python 3.11 CPU venv only.\n'
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
for tool in uv git cmake make c++; do
    command -v "$tool" >/dev/null || { printf 'Missing %s; see README.md for local native dependencies.\n' "$tool" >&2; exit 1; }
done
if [[ ! -e "$CRAFTIUM_DIR" ]]; then
    mkdir -p "$(dirname "$CRAFTIUM_DIR")"
    git clone --no-checkout https://github.com/francelico/craftium.git "$CRAFTIUM_DIR"
    git -C "$CRAFTIUM_DIR" checkout --detach "$PIN"
fi
[[ $(git -C "$CRAFTIUM_DIR" rev-parse HEAD) == "$PIN" ]] || { printf 'Craftium source must remain at %s; existing checkout was not changed.\n' "$PIN" >&2; exit 1; }
git -C "$CRAFTIUM_DIR" submodule update --init --recursive
python3 "$ROOT/scripts/apply_fixed24.py" --craftium-dir "$CRAFTIUM_DIR"
# Install only the native GT requirements; no model dependencies.
if [[ ! -x "$GT_ENV/bin/python" ]]; then
    uv venv --python 3.11 "$GT_ENV" --seed
fi
uv pip install --python "$GT_ENV/bin/python" -r "$ROOT/requirements-gt.txt"
# Let CMake find user-installed Homebrew libraries on macOS. No brew/sudo is run here.
if [[ $(uname -s) == Darwin ]] && command -v brew >/dev/null; then
    export CMAKE_PREFIX_PATH="$(brew --prefix)${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    # Mono.framework ships libpng 1.4 headers. CMake previously mixed those with Homebrew
    # libpng 1.6, so Luanti failed to load PNG/fonts and Craftium timed out.
    PNG_PREFIX=$(brew --prefix libpng)
    JPEG_PREFIX=$(brew --prefix jpeg-turbo)
    GL_FW=/System/Library/Frameworks/OpenGL.framework
    CMAKE_BIN=$(command -v cmake)
    CMAKE_WRAP=$(mktemp -d)
    # Ignore Mono headers and XQuartz libGL. Do not set CMAKE_FIND_FRAMEWORK=LAST:
    # that previously selected /usr/X11R6 OpenGL 1.x instead of OpenGL.framework.
    cat > "$CMAKE_WRAP/macos-gl-png.cmake" <<'EOF'
list(APPEND CMAKE_IGNORE_PATH
    "/Library/Frameworks/Mono.framework"
    "/opt/X11"
    "/usr/X11R6"
    "/opt/X11/include"
    "/opt/X11/lib"
    "/usr/X11R6/include"
    "/usr/X11R6/lib")
list(APPEND CMAKE_IGNORE_PREFIX_PATH
    "/Library/Frameworks/Mono.framework"
    "/opt/X11"
    "/usr/X11R6")
EOF
    cat > "$CMAKE_WRAP/cmake" <<EOF
#!/bin/sh
exec "$CMAKE_BIN" "\$@" \\
  -DCMAKE_TOOLCHAIN_FILE=$CMAKE_WRAP/macos-gl-png.cmake \\
  -DPNG_PNG_INCLUDE_DIR=$PNG_PREFIX/include \\
  -DPNG_LIBRARY=$PNG_PREFIX/lib/libpng.dylib \\
  -DJPEG_INCLUDE_DIR=$JPEG_PREFIX/include \\
  -DJPEG_LIBRARY=$JPEG_PREFIX/lib/libjpeg.dylib \\
  -DOPENGL_INCLUDE_DIR=$GL_FW/Headers \\
  -DOPENGL_gl_LIBRARY=$GL_FW \\
  -DOPENGL_glu_LIBRARY=$GL_FW
EOF
    chmod +x "$CMAKE_WRAP/cmake"
    export PATH="$CMAKE_WRAP:$PATH"
fi
CRAFTIUM_SRC="$CRAFTIUM_DIR"
# In-source CMake leaves object files under src/CMakeFiles and irr/CMakeFiles; only
# removing the top-level cache relinks the old Mono-header PNG objects.
find "$CRAFTIUM_SRC" -name CMakeCache.txt -delete
find "$CRAFTIUM_SRC" -type d -name CMakeFiles -prune -exec rm -rf {} +
# Official setup.py copytree skips existing craftium/bin, so a rebuild can still
# package the previous luanti binary.
for staged in bin builtin fonts locale textures client craftium-envs; do
    rm -rf "$CRAFTIUM_SRC/craftium/$staged"
done
BUILD_THREADS=${BUILD_THREADS:-4} uv pip install --python "$GT_ENV/bin/python" --no-build-isolation --no-deps --reinstall --no-cache "$CRAFTIUM_SRC"
uv pip install --python "$GT_ENV/bin/python" -r "$ROOT/requirements-gt.txt"
if grep -q '/Library/Frameworks/Mono.framework/Headers' "$CRAFTIUM_SRC"/irr/src/CMakeFiles/IRRVIDEOOBJ.dir/flags.make 2>/dev/null; then
    printf 'Irrlicht still compiles with Mono.framework headers; PNG/JPEG includes were not overridden.\n' >&2
    exit 1
fi
if grep -q '/Library/Frameworks/Mono.framework/Headers/png.h' "$CRAFTIUM_SRC"/irr/src/CMakeFiles/IRRVIDEOOBJ.dir/CImageLoaderPNG.cpp.o.d 2>/dev/null; then
    printf 'CImageLoaderPNG still includes Mono png.h 1.4.12.\n' >&2
    exit 1
fi
LUANTI_BIN="$GT_ENV/lib/python3.11/site-packages/craftium/bin/luanti"
if [[ $(uname -s) == Darwin && -x "$LUANTI_BIN" ]]; then
    libs=$(otool -L "$LUANTI_BIN")
    printf '%s\n' "$libs" | grep -q '/opt/X11/lib/libGL' && {
        printf 'Luanti still links XQuartz libGL; OpenGL.framework was not selected.\n' >&2
        exit 1
    }
    printf '%s\n' "$libs" | grep -q 'OpenGL.framework' || {
        printf 'Luanti is not linked to OpenGL.framework.\n' >&2
        exit 1
    }
fi
"$GT_ENV/bin/python" - <<'PY'
import importlib.util
import sys
import av, craftium, gymnasium, numpy
assert sys.version_info[:2] == (3, 11)
print('Local CPU GT environment ready:', sys.executable)
print('No model packages are required by the GT environment.')
print('Native engine has not been started. Ready for user-run generation/play.')
PY
printf 'Test A: %s scripts/test_gt.py --case test_a\n' "$GT_ENV/bin/python"
