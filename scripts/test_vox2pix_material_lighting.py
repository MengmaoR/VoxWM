"""Compile the pinned native material policy and ambient setting bounds."""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
from tempfile import TemporaryDirectory

from test_vox2pix_lighting import PATCH as LEGACY_PATCH, PIN, ROOT


PATCH = ROOT / "voxwm_gt/patches/material_ambient.patch"

TEST = """
#include "src/client/voxwm_material_ambient.h"
#include "src/voxwm_lighting_floor.h"
#include <cassert>
#include <limits>

int main()
{
    static_assert(voxwmSolidAmbientMaterial(TILE_MATERIAL_OPAQUE, NDT_NORMAL));
    static_assert(voxwmSolidAmbientMaterial(TILE_MATERIAL_BASIC, NDT_NODEBOX));
    static_assert(voxwmSolidAmbientMaterial(TILE_MATERIAL_WAVING_LEAVES, NDT_ALLFACES));
    static_assert(voxwmSolidAmbientMaterial(TILE_MATERIAL_WAVING_PLANTS, NDT_PLANTLIKE));
    static_assert(!voxwmSolidAmbientMaterial(TILE_MATERIAL_ALPHA, NDT_NORMAL));
    static_assert(!voxwmSolidAmbientMaterial(TILE_MATERIAL_PLAIN, NDT_NORMAL));
    static_assert(!voxwmSolidAmbientMaterial(TILE_MATERIAL_PLAIN_ALPHA, NDT_NORMAL));
    static_assert(!voxwmSolidAmbientMaterial(TILE_MATERIAL_OPAQUE, NDT_AIRLIKE));
    static_assert(!voxwmSolidAmbientMaterial(TILE_MATERIAL_OPAQUE, NDT_GLASSLIKE));
    static_assert(!voxwmSolidAmbientMaterial(TILE_MATERIAL_BASIC, NDT_GLASSLIKE_FRAMED));
    static_assert(!voxwmSolidAmbientMaterial(TILE_MATERIAL_BASIC, NDT_FIRELIKE));
    for (int value = TILE_MATERIAL_BASIC; value <= TILE_MATERIAL_PLAIN_ALPHA; ++value) {
        const auto material = static_cast<MaterialType>(value);
        assert(!voxwmSolidAmbientMaterial(material, NDT_LIQUID));
        assert(!voxwmSolidAmbientMaterial(material, NDT_FLOWINGLIQUID));
    }
    for (auto liquid : {TILE_MATERIAL_LIQUID_TRANSPARENT, TILE_MATERIAL_LIQUID_OPAQUE,
                        TILE_MATERIAL_WAVING_LIQUID_BASIC, TILE_MATERIAL_WAVING_LIQUID_TRANSPARENT,
                        TILE_MATERIAL_WAVING_LIQUID_OPAQUE}) {
        for (int value = NDT_NORMAL; value < NodeDrawType_END; ++value)
            assert(!voxwmSolidAmbientMaterial(liquid, static_cast<NodeDrawType>(value)));
    }
    for (float value : {0.0f, 0.35f, 0.5f, 1.0f})
        assert(voxwm_limit_lighting_floor(value) == value);
    assert(voxwm_limit_lighting_floor(-0.35f) == 0.0f);
    assert(voxwm_limit_lighting_floor(1.5f) == 1.0f);
    assert(voxwm_limit_lighting_floor(std::numeric_limits<float>::quiet_NaN()) == 0.0f);
    assert(voxwm_limit_lighting_floor(std::numeric_limits<float>::infinity()) == 0.0f);
}
"""


def native_enum(craftium, filename, name):
    source = subprocess.check_output(["git", "-C", craftium, "show", f"{PIN}:{filename}"], text=True)
    return re.search(rf"enum {name} : u8\s*{{.*?}};", source, re.DOTALL).group()


def check(craftium):
    material = native_enum(craftium, "src/client/tile.h", "MaterialType")
    drawtype = native_enum(craftium, "src/nodedef.h", "NodeDrawType")
    with TemporaryDirectory(prefix="vox2pix-material-unit-") as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src/nodedef.h").write_text(
            "#include <initializer_list>\nusing u8 = unsigned char;\n"
            "namespace video { struct SColor { explicit SColor(unsigned) {} }; }\n"
            f"{material}\n{drawtype}\n"
        )
        for patch, path in ((LEGACY_PATCH, "src/voxwm_lighting_floor.h"),
                            (PATCH, "src/client/voxwm_material_ambient.h")):
            subprocess.run(["git", "apply", f"--include={path}", patch], cwd=root, check=True)
        (root / "test.cpp").write_text(TEST)
        cxx = shlex.split(os.environ.get("CXX", "c++"))
        subprocess.run([*cxx, "-std=c++17", "-Isrc", "test.cpp", "-o", "material_test"], cwd=root, check=True)
        subprocess.run([root / "material_test"], cwd=root, check=True)
    print("Native ambient unit passed for solid/cutout, all liquid variants, and setting bounds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--craftium-dir", type=Path, default=ROOT / "third_party/craftium")
    check(parser.parse_args().craftium_dir.resolve())
