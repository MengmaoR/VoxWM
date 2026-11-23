"""Compile the absolute-distance policy and preserve native draw/far-plane semantics."""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
from tempfile import TemporaryDirectory

from test_vox2pix_lighting import PIN, ROOT


TEST = r'''
#include "src/client/voxwm_fog_absolute.h"
#include "src/client/voxwm_fog_probe.h"
#include <cassert>
#include <limits>
#include <sstream>
struct Sky { float getFogStart() const { return 2.f / 3.f; } };
struct Game {
    Sky sky_state;
    Sky *sky = &sky_state;
    struct Data { float fog_range = 480.f; } runData;
    static constexpr float BS = 10.f;
    float start(float absolute_start, float absolute_end) {
        const bool absolute_fog = voxwmAbsoluteFog(absolute_start, absolute_end);
        return START;
    }
    float end(float absolute_start, float absolute_end) {
        const bool absolute_fog = voxwmAbsoluteFog(absolute_start, absolute_end);
        return END;
    }
};
int main() {
    assert(!voxwmAbsoluteFog(-1, -1));
    assert(voxwmAbsoluteFog(20, 30));
    Game game;
    assert(game.start(20, 30) == 200);
    assert(game.end(20, 30) == 300);
    assert(game.start(-1, -1) == 320);
    assert(game.end(-1, -1) == 480);
    for (const auto &p : {std::array<float,2>{-1,30}, {20,20}, {30,20}, {20,-1},
            {std::numeric_limits<float>::infinity(),30}, {20,std::numeric_limits<float>::quiet_NaN()}}) {
        bool failed = false;
        try { voxwmAbsoluteFog(p[0], p[1]); }
        catch (const std::invalid_argument &) { failed = true; }
        assert(failed);
    }
    std::ostringstream out;
    voxwmWriteMapblocks(out, {{0,0,2}, {-1,0,2}});
    assert(out.str() == "[[0,0,2],[-1,0,2]]");
}
'''


def check(craftium):
    patch = ROOT / "voxwm_gt/patches/fog_absolute.patch"
    text = patch.read_text()
    assert "diff --git a/client/shaders" not in text
    assert "setFarValue" not in text
    assert 'setDefault("fog_start_distance", "-1")' in text
    start = re.search(r"^\+\s*(absolute_fog \? absolute_start.*),$", text, re.MULTILINE).group(1)
    end = re.search(r"^\+\s*(absolute_fog \? absolute_end.*),$", text, re.MULTILINE).group(1)
    camera = subprocess.check_output(["git", "-C", str(craftium), "show", f"{PIN}:src/client/camera.cpp"], text=True)
    assert "2000 * BS : viewing_range * BS" in camera
    with TemporaryDirectory(prefix="voxwm-fog48-unit-") as temporary:
        root = Path(temporary)
        for header in ("voxwm_fog_absolute.h", "voxwm_fog_probe.h"):
            if header == "voxwm_fog_probe.h":
                subprocess.run(["git", "apply", f"--include=src/client/{header}", ROOT / "voxwm_gt/patches/fog_range.patch"],
                               cwd=root, check=True)
            subprocess.run(["git", "apply", f"--include=src/client/{header}", patch], cwd=root, check=True)
        (root / "test.cpp").write_text(TEST.replace("START", start).replace("END", end))
        cxx = shlex.split(os.environ.get("CXX", "c++"))
        subprocess.run([*cxx, "-std=c++17", "test.cpp", "-o", "fog48_unit"], cwd=root, check=True)
        subprocess.run([root / "fog48_unit"], cwd=root, check=True)
    print("Native absolute fog policy passed for draw48/fog20–30, legacy defaults, invalid settings and MapBlock coordinates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--craftium-dir", type=Path, default=ROOT / "third_party/craftium")
    check(parser.parse_args().craftium_dir.resolve())
