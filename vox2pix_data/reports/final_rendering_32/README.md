PASS WITH KNOWN NATIVE VISUAL LIMITATION

# Final native RGB32 / absolute Fog20–30

The [offline report ZIP](final_rendering_32.zip) opens at
`vox2pix_final_rendering_32/report/index.html`. It contains the 12-distance
fixed-camera calibration, all original 15 RGB views, White Model / Water Count /
BACKGROUND previews, copied source arrays and actual native frame probes.
The [measurements](summary.json) separate Unit, Native, Visual and regression
checks. Formal dataset collection and model training have not started.

| Actual native measurement | Value |
| --- | --- |
| Configured RGB range / wanted range / map draw range | 32 / 32 / 32 nodes |
| Driver Fog Start / End | 20 / 30 nodes |
| Node Shader Fog distance / shading parameter | 30 / approximately 3 |
| Camera near / far | 0.1 / 2000 nodes; original native algorithm |
| Server send / generate caps | 8 MapBlocks / 128 nodes each |
| Active-object send cap | 8 MapBlocks / 128 nodes |
| Loaded camera-neighborhood MapBlocks | 68–188; scene views 87–188 |
| Gamma / global LUT floor / solid material floor | 3.0 / 0.0 / 0.35 |
| Native LUT endpoints | 0 / 255 |
| Voxel observation | unchanged 48³, half extent 24 |

The twelve distances are 10, 18, 20, 22, 24, 25, 27, 29, 30, 31, 32 and 34.
The camera stays fixed at engine XYZ `[0,2,0.5]`. A textured native wool panel
is placed at one exact distance at a time and removed before the next distance;
markers therefore cannot occlude one another. Each distance has Fog OFF and
the final Fog20–30 capture. The optional historical ratio-0.8 control was omitted.

At 20 the marker is unchanged; at 24 it remains recognizable with 40% fog;
at 27 and 29 the factors are 70% and 90%. At 30, 31, 32 and 34, the marker
core is exactly `[191,215,254]`, the actual Fog Color, with zero core RGB error
and no marker texture. Maximum measured shader-color error at 22–29 is
0.447/255. The actual marker MapBlock is submitted at 30, 31 and 32, confirming
that RGB is not hard-clipped at 24 or 30.

## Known limits

- The 34-node marker is also submitted because native culling uses MapBlock
  bounding volumes. This is not exact per-voxel culling at 32. Its texture is
  fully fogged.
- A faint uniform patch remains against the native Sky gradient. At the
  horizon its RGB is `[191,215,254]` versus adjacent sky `[188,213,254]`.
  Fog Color equality does not establish equality with every background pixel.
- Native underwater surface polygons/background boundaries remain visible.
  The Water Shader, water transparency and post-effect were not changed.
  All 15 eye post-effect RGBA values match the previous draw48 replay;
  Underwater retains `[24,46,89,60]`.
- This run uses native software OpenGL rendering on CPU. It does not test
  discrete-GPU hardware or establish broad dataset coverage.

## Verification and versions

- 25 relevant Python Unit tests passed in 2.117 seconds; the updated report
  checks passed again after adding explicit limitation reporting.
- Compiled C++ policy checks passed independent Fog20–30 with draw32/48/128,
  original ratio defaults, invalid settings and MapBlock serialization.
- The isolated native incremental build passed. Existing fixed24, integer
  capture, lighting, material ambient and absolute-fog patches were reused;
  this change requires no new C++ or Shader implementation.
- All 12 calibration distances and all 15 saved native cameras completed.
  Actual replay Node IDs / param2 and camera match the source. Exported
  Voxel, White Model, raw param2, Face and Water Count arrays are verbatim
  source copies verified by array equality; White Model was not recomputed.
  The exported BACKGROUND mask matches the saved surface IDs.
- All offline HTML image and file links passed checking. Images were visually
  reviewed with the limits above, rather than declaring perfect background removal.

Craftium base commit: `ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49`.
VoxeLibre commit: `178c71b1e7d6ec35c0d9c481c55eb306d6f330bc`.
Native execution source: `b3aeaaecfbd7a8f758fd59dd1a21907373c756be`.
The actual binary is the isolated checkout's `bin/luanti`, copied into each
temporary session and selected directly by `GTSession`, bypassing wrappers.

## Modified files

- `vox2pix_data/protocol.py`, `__main__.py`: final active profile and CLI default.
- `vox2pix_data/native.py`, `v3_report.py`: correct profile-specific range metadata.
- `vox2pix_data/final_rendering.py`, `final_calibration.lua`: native calibration and saved-scene replay.
- `vox2pix_data/final_report.py`: offline images, measurements, array regression and limitation reporting.
- `vox2pix_data/test_final_protocol.py`, `test_final_report.py`, `scripts/test_vox2pix_fog48.py`: relevant tests.
- `AGENTS.md`, `PROJECT_STATE.md`, `README.md`, `VOX2PIX_DATA_COLLECTION_PROTOCOL.md`, `vox2pix_data/README.md`: current protocol and commands.
- This report's `README.md`, `summary.json`, `final_rendering_32.zip`: measured results and offline assets.

## Reproduce

From the repository root, build and test:

```bash
bash scripts/build_vox2pix_fog48.sh third_party/craftium_vox2pix_fog48
.venv/bin/python scripts/test_vox2pix_fog48.py \
  --craftium-dir third_party/craftium_vox2pix_fog48
.venv/bin/python -m unittest vox2pix_data.test_final_protocol \
  vox2pix_data.test_final_report vox2pix_data.test_fog48_report \
  vox2pix_data.test_fog_report vox2pix_data.test_v3_projection \
  vox2pix_data.test_material_report
```

On the existing server, inherit its local graphics environment and use a fresh
output path for the authorized replay:

```bash
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
import sys
os.environ.update(json.loads(Path("vae_data/configs/collection_64x1_clip2.json").read_text())["environment"])
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.execv(sys.executable, [
    sys.executable, "-B", "-u", "-m", "vox2pix_data.final_rendering",
    "--dataset", "outputs/vox2pix_v3_lightfloor_15",
    "--craftium-dir", "third_party/craftium_vox2pix_fog48",
    "--output", "outputs/vox2pix_final_rendering_32_repeat",
    "--scene-timeout", "240",
])
PY
```

A fresh run intentionally leaves Unit/Visual confirmation pending until its
actual test results and images are reviewed. Historical datasets remain unchanged.
