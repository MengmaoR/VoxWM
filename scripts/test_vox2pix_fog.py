"""Compile the pinned native fog uniform/GLSL distance expressions without a renderer."""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
from tempfile import TemporaryDirectory

from test_vox2pix_lighting import PIN, ROOT


def pinned(craftium, path):
    return subprocess.check_output(["git", "-C", str(craftium), "show", f"{PIN}:{path}"], text=True)


def check(craftium):
    fragment = pinned(craftium, "client/shaders/nodes_shader/opengl_fragment.glsl")
    engine = pinned(craftium, "src/client/renderingengine.cpp")
    camera = pinned(craftium, "src/client/camera.cpp")
    clarity = re.search(r"^\tfloat clarity = clamp\([^;]+;", fragment, re.MULTILINE).group()
    parameter = re.search(r"parameter = 1\.0f / \(1\.0f - fog_start / fog_end\);", engine).group()
    assert "length(eyeVec)" in clarity
    assert "2000 * BS : viewing_range * BS" in camera
    patch = (ROOT / "voxwm_gt/patches/fog_range.patch").read_text()
    assert "diff --git a/client/shaders" not in patch
    assert '"actual_fog_start' in patch and '"shader_uniforms_bound' in patch
    # Preserve the native algebra and its final-addition workaround in the compiled check.
    clarity = clarity.replace("length(eyeVec)", "distance").replace("0.0, 1.0", "0.0f, 1.0f")
    source = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
using std::clamp;
float fog(float distance, float start, float end) {
    float fog_start = start, fog_end = end, parameter;
    PARAMETER
    const float fogShadingParameter = parameter, fogDistance = fog_end;
    CLARITY
    return 1.0f - clarity;
}
int main() {
    const float distance[] = {10, 20, 22, 24, 25, 27, 29, 30, 32};
    const float expected[] = {0, 0, .2f, .4f, .5f, .7f, .9f, 1, 1};
    for (unsigned i = 0; i < sizeof(distance) / sizeof(float); ++i)
        assert(std::abs(fog(distance[i] * 10, 30 * .6666667f * 10, 30 * 10) - expected[i]) < 1e-6f);
    assert(fog(24 * 10, 30 * .8f * 10, 30 * 10) == 0);
    assert(fog(30 * 10, 30 * .8f * 10, 30 * 10) == 1);
}
'''.replace("PARAMETER", parameter).replace("CLARITY", clarity)
    with TemporaryDirectory(prefix="voxwm-fog-unit-") as temporary:
        root = Path(temporary)
        (root / "fog.cpp").write_text(source)
        cxx = shlex.split(os.environ.get("CXX", "c++"))
        subprocess.run([*cxx, "-std=c++17", "fog.cpp", "-o", "fog_unit"], cwd=root, check=True)
        subprocess.run([root / "fog_unit"], cwd=root, check=True)
    print("Pinned native fog expressions passed for 10/20/22/24/25/27/29/30/32 blocks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--craftium-dir", type=Path, default=ROOT / "third_party/craftium")
    check(parser.parse_args().craftium_dir.resolve())
