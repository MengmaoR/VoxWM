"""Exercise the pinned Luanti light table and the opt-in floor in a native unit."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "voxwm_gt" / "patches" / "lighting_floor.patch"
PIN = "ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49"

STUBS = {
    "src/config.h": "#define IS_CLIENT_BUILD 1\n#define CHECK_CLIENT_BUILD() 1\n",
    "src/irrlichttypes.h": "#include <cstdint>\nusing u8 = uint8_t;\nusing u32 = uint32_t;\nusing s32 = int32_t;\n",
    "src/util/numeric.h": """
        #include <algorithm>
        template<class T> T sqr(T x) { return x * x; }
        template<class T> T rangelim(T x, T lo, T hi) { return std::clamp(x, lo, hi); }
    """,
    "src/settings.h": """
        #include <string>
        struct Settings {
            float getFloat(const std::string &name) const {
                if (name == "lighting_boost_center") return .5f;
                if (name == "lighting_boost" || name == "lighting_boost_spread") return .2f;
                return 0.f;
            }
        };
        extern Settings *g_settings;
    """,
    "test.cpp": """
        #include "src/light.h"
        #include "src/settings.h"
        #include <algorithm>
        #include <array>
        #include <cmath>
        #include <limits>

        Settings settings;
        Settings *g_settings = &settings;
        extern const u8 *legacy_light_decode_table;
        void legacy_set_light_table(float gamma);

        int main() {
            for (float gamma : {0.33f, 1.0f, 3.0f}) {
                legacy_set_light_table(gamma);
                set_light_table(gamma, 0.0f);
                assert(std::equal(light_decode_table, light_decode_table + LIGHT_SUN + 1,
                                  legacy_light_decode_table));
            }
            std::array<u8, LIGHT_SUN + 1> original;
            std::copy_n(legacy_light_decode_table, original.size(), original.begin());
            for (float floor : {0.35f, 0.5f, 1.f}) {
                set_light_table(3.f, floor);
                assert(decode_light(0) == std::lround(floor * 255));
                assert(decode_light(LIGHT_SUN) == 255);
                assert(std::is_sorted(light_decode_table, light_decode_table + original.size()));
                for (size_t i = 0; i < original.size(); ++i)
                    assert(decode_light(i) == std::lround(floor * 255 + (1 - floor) * original[i]));
            }
            for (float bad : {-1.f, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()}) {
                set_light_table(3.f, bad);
                assert(std::equal(original.begin(), original.end(), light_decode_table));
            }
            set_light_table(3.f, 1.5f);
            assert(decode_light(0) == 255 && decode_light(LIGHT_SUN) == 255);
        }
    """,
}


def check(craftium):
    cxx = shlex.split(os.environ.get("CXX", "c++"))
    with TemporaryDirectory(prefix="vox2pix-lighting-unit-") as tmp:
        root = Path(tmp)
        for name, contents in STUBS.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        for name in ("light.h", "light.cpp"):
            contents = subprocess.check_output(["git", "-C", craftium, "show", f"{PIN}:src/{name}"])
            (root / "src" / name).write_bytes(contents)
        (root / "src" / "legacy.cpp").write_bytes((root / "src" / "light.cpp").read_bytes())
        subprocess.run(["git", "apply", "--include=src/light.cpp", "--include=src/light.h",
                        "--include=src/voxwm_lighting_floor.h", PATCH], cwd=root, check=True)
        subprocess.run([*cxx, "-std=c++17", "-Dset_light_table=legacy_set_light_table",
                        "-Ddecode_light_f=legacy_decode_light_f",
                        "-Dlight_decode_table=legacy_light_decode_table",
                        "-c", "src/legacy.cpp", "-o", "legacy.o"], cwd=root, check=True)
        subprocess.run([*cxx, "-std=c++17", "src/light.cpp", "test.cpp", "legacy.o",
                        "-o", "lighting_test"], cwd=root, check=True)
        subprocess.run([root / "lighting_test"], cwd=root, check=True)
    print("Native LUT unit passed for floor 0, 0.35, 0.5 and invalid values")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--craftium-dir", type=Path, default=ROOT / "third_party/craftium")
    check(parser.parse_args().craftium_dir.resolve())
