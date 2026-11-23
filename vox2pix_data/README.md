# Vox2Pix data collection

## V3 native smoke

For the legacy global-LUT profile, use the existing Linux software-rendering dependencies and a separate Craftium
checkout configured for a client build. From the repository root, build its
lighting patch and collect or resume the three independently seeded scenes and
15 RGB frames.

```bash
BUILD_THREADS=22 bash scripts/build_vox2pix_lighting.sh third_party/craftium_vox2pix_floor

.venv/bin/python -m vox2pix_data collect \
  --mode smoke --lighting-profile global-lut \
  --scene-kinds surface cave underwater --scenes-per-kind 1 \
  --views-per-scene 5 --width 640 --height 360 --seed 42 --workers 3 \
  --renderer software --craftium-dir third_party/craftium_vox2pix_floor \
  --dataset outputs/vox2pix_v3_lightfloor_15 \
  --scene-timeout 180 --survey-columns 32 --sites-per-world 4

.venv/bin/python -m vox2pix_data validate --dataset outputs/vox2pix_v3_lightfloor_15
.venv/bin/python -m vox2pix_data report --dataset outputs/vox2pix_v3_lightfloor_15
```

Open `outputs/vox2pix_v3_lightfloor_15/report/index.html` directly. Copy or archive
the complete report directory when viewing it on another machine. The report
shows the capture status, image comparisons, native poses, search statistics
and same-camera Floor 0.0 and 0.35 images. Change `--scenes-per-kind 1` to `3` with a
fresh output path for a separately requested 45-frame smoke.

The published legacy RGB uses gamma 3.0 and a client LUT floor of 0.35. The underwater
observer sinks with native physics and the recorder forms its 48³ crop around
the settled position. The earlier lighting trial at
`outputs/vox2pix_v3_smoke_15` uses a different manifest.

To compare a different floor on the saved cameras, run this in the same
software-rendering environment:

```bash
.venv/bin/python -m vox2pix_data.relight \
  --dataset outputs/vox2pix_v3_lightfloor_15 \
  --craftium-dir third_party/craftium_vox2pix_floor \
  --floor 0.20 --output outputs/vox2pix_v3_floor020_rgb
```

Open the output's `index.html` for the same-session RGB pairs. The replay checks
the stored 48³ node IDs and param2 at every view. Summary JSON records any
saved cells restored in the disposable world and differences between the new
control and the original RGB, which can include moving clouds.

### Material-aware ambient comparison

Use two separate, configured Craftium checkouts. Keep the previous
`third_party/craftium_vox2pix_floor` binary for the legacy zero-floor reference.
Configure the isolated material checkout at Craftium commit
`ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49` using the existing Linux graphics
dependencies and the pinned VoxeLibre game. Use the same software-rendering
environment as the original smoke.

```bash
cmake -S third_party/craftium_vox2pix_material -B third_party/craftium_vox2pix_material \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DRUN_IN_PLACE=TRUE \
  -DBUILD_CLIENT=TRUE -DBUILD_SERVER=FALSE -DENABLE_LTO=FALSE \
  -DCMAKE_PREFIX_PATH=/user-data/txj/.tools/craftium-deps
BUILD_THREADS=22 bash scripts/build_vox2pix_material_lighting.sh third_party/craftium_vox2pix_material
python3 scripts/test_vox2pix_material_lighting.py --craftium-dir third_party/craftium_vox2pix_material
python3 scripts/test_vox2pix_lighting.py --craftium-dir third_party/craftium_vox2pix_material
.venv/bin/python -m vox2pix_data.material_relight \
  --dataset outputs/vox2pix_v3_lightfloor_15 \
  --craftium-dir third_party/craftium_vox2pix_material \
  --legacy-craftium-dir third_party/craftium_vox2pix_floor \
  --output outputs/vox2pix_v3_material_lighting/replay/lighting_material_comparison
```

The comparison replays the 15 original viewpoints. A uses native lighting,
B uses the 0.35 global LUT floor, and C uses the 0.35 solid material ambient
coefficient with the native zero-floor LUT. Open the comparison's
`index.html` together with its sibling image directories. All diagnostic
frames are kept separate from the original smoke data.

See the [native results](reports/v3_material_lighting/README.md). The
material-ambient candidate is available for a separately approved smoke.
Use a new dataset path and the patched material checkout.

```bash
.venv/bin/python -m vox2pix_data collect \
  --mode smoke --lighting-profile material-ambient \
  --scene-kinds surface cave underwater --scenes-per-kind 1 \
  --views-per-scene 5 --width 640 --height 360 --seed 42 --workers 3 \
  --renderer software --craftium-dir third_party/craftium_vox2pix_material \
  --dataset outputs/vox2pix_v3_material_smoke_15 \
  --scene-timeout 180 --survey-columns 32 --sites-per-world 4
.venv/bin/python -m vox2pix_data validate --dataset outputs/vox2pix_v3_material_smoke_15
.venv/bin/python -m vox2pix_data report --dataset outputs/vox2pix_v3_material_smoke_15
```

## Native fog calibration

The current smoke profile is `material-ambient-fog20`, versioned separately
from both previous lighting trials. Its voxel observation half extent is
24 and RGB draw range is 30. Native fog starts at 20 and ends at 30 blocks.
The same material lighting uses global floor 0 and solid ambient 0.35.
Use an isolated checkout at Craftium
`ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49` with its existing native dependencies,
configured for the client build, and VoxeLibre
`178c71b1e7d6ec35c0d9c481c55eb306d6f330bc`.

Run in the original Linux software-rendering environment. The build helper
applies the capture, material and fog patches to this checkout.

```bash
cmake -S third_party/craftium_vox2pix_fog -B third_party/craftium_vox2pix_fog \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DRUN_IN_PLACE=TRUE \
  -DBUILD_CLIENT=TRUE -DBUILD_SERVER=FALSE -DENABLE_LTO=FALSE \
  -DCMAKE_PREFIX_PATH=/user-data/txj/.tools/craftium-deps
BUILD_THREADS=22 bash scripts/build_vox2pix_fog.sh third_party/craftium_vox2pix_fog
python3 scripts/test_vox2pix_fog.py --craftium-dir third_party/craftium_vox2pix_fog
.venv/bin/python -m unittest vox2pix_data.test_fog_report -q

.venv/bin/python -m vox2pix_data.fog_calibration \
  --dataset outputs/vox2pix_v3_lightfloor_15 \
  --craftium-dir third_party/craftium_vox2pix_fog \
  --output outputs/vox2pix_fog20/replay/native_comparison \
  --scene-timeout 180
```

The offline report is `OUTPUT/index.html`. It includes 20/22/24/25/27/29/30/32
block marker images and all 15 recorded views in Fog OFF, old 0.8 and new
0.6666667 modes. Runtime probes record final driver and node-shader values.
Read `summary.json` for the measured calibration status.

On the existing server with user-installed Xvfb libraries, the following
command loads its recorded graphics environment before replay.

```bash
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
import sys

environment = json.loads(Path("vae_data/configs/collection_64x1_clip2.json").read_text())["environment"]
os.environ.update(environment)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.execv(sys.executable, [
    sys.executable, "-B", "-u", "-m", "vox2pix_data.fog_calibration",
    "--dataset", "outputs/vox2pix_v3_lightfloor_15",
    "--craftium-dir", "third_party/craftium_vox2pix_fog",
    "--output", "outputs/vox2pix_fog20/replay/native_comparison",
    "--scene-timeout", "180",
])
PY
```

For a separately approved new smoke, use `--mode smoke
--lighting-profile material-ambient-fog20`, the fog checkout and a fresh
dataset path. Legacy smoke resume commands must pass their original
`--lighting-profile global-lut` or `material-ambient`.

## Historical v2 1K compression pilot

The v2 protocol is `vox2pix-pilot-v2-camera16`. Use a newly initialized pilot
directory for this version. The compression report includes observed native
camera heights across all views.

Use the existing native GT environment on Linux, with the repository's
fixed24 integer-origin capture patch built into Craftium. Install the existing
`vae_p0/requirements.txt` for NumPy, Zarr v2, Blosc and Pillow if they are
missing. The CPU projector also needs a C++17 compiler (`CXX` defaults to
`c++`). A software session uses the existing Xvfb setup. See
[the repository setup](../README.md) and the
[Vox2Pix protocol](../VOX2PIX_DATA_COLLECTION_PROTOCOL.md).

For a user-installed Xvfb, put its executable directory on `PATH` and export
the graphics environment used by native GT. Set `X11_XKBDIR` to the bundled
XKB data directory when required. The launcher passes it to each private Xvfb
using `-xkbdir`. On standard system installs the variable can be left unset.

Run from the VoxWM repository root. Adjust `--workers` for the machine.
`--craftium-dir` can point to an existing compatible checkout.

```bash
.venv/bin/python -m vox2pix_data collect \
  --dataset outputs/vox2pix_pilot_1k_640x360 \
  --scenes 200 --views-per-scene 5 --width 640 --height 360 \
  --seed 42 --workers 8 --renderer software
```

Resume with the same command and output directory. The number of workers can
change across invocations. Slots with no qualifying terrain report their
biome and height. Running the same command searches the same bounded seed
sequence for those slots. After exhausting round 0, use
`--search-round 1 --search-rounds 2 --sites-per-world 8 --survey-columns 32`
on the same dataset to search two fresh rounds of eight worlds per missing
slot. The wider survey supplies more anchor candidates in the target biome.
Committed scenes and dataset criteria are preserved. Continue with the next
round number if the selected rounds are exhausted. Search rejections break
out missing standing cave floors and the center and full-region free-space
gates. For a small tail, `--parallel-rounds` searches distinct seed rounds for
the same slot concurrently and commits the first eligible result once. Add
`--land-priority --survey-columns 48` to rank estimated native surface and
nearby biome matches before surveying. Every fourth site still uses the
original search order. `--cave-hints` can also follow nearby native large-cave
notifications. Acceptance always uses generated terrain. An unsynchronized
response discards the interrupted world before trying another seed.

After all 200 scenes are committed, run:

```bash
.venv/bin/python -m vox2pix_data check --dataset outputs/vox2pix_pilot_1k_640x360
.venv/bin/python -m vox2pix_data compression-report --dataset outputs/vox2pix_pilot_1k_640x360
```

The last command writes `compression_report.json` and
`compression_report.md` under the dataset root. The compression gate is
based on the validated complete 1K pilot. See the
[measured pilot results and collection logs](reports/pilot_1k/README.md).

Open a second server terminal to monitor the existing collector and output.

```bash
.venv/bin/python -m vox2pix_data dashboard --dataset outputs/vox2pix_pilot_1k_640x360
```

The dashboard shows scene and frame totals, rate, ETA when available, Linux
worker and world counts, and scene status by biome and height. Add `--once`
for a single snapshot, or `--log` to read a particular collector log. It uses
committed scene directories and reads collector state without owning its processes.

The root contains `manifest.json` and `node_vocabulary.json`. Each committed
`scenes/scene_NNNN/` contains `scene.json`, `voxels.zarr/` and
`rgb/000.png` through `rgb/004.png`. Read frame `i` from each white array by
indexing it with the same `i` used in the PNG filename and `scene.json` views.

## Inspect an existing pilot

Run the read-only Viewer on the machine that holds the dataset.

```bash
.venv/bin/python -m vox2pix_data viewer \
  --dataset outputs/vox2pix_pilot_1k_640x360 --port 8767
```

If the data lives on a server, connect with `ssh -L 8767:127.0.0.1:8767 aml-g06-gpu`
and run the command above from the remote VoxWM checkout. Open
[the Viewer](http://127.0.0.1:8767/) in your local browser. The Viewer reads
the checked-in Three.js files offline and needs no simulator or GPU server.

Select a committed scene and one of its five views. Orbit the 48³ voxel grid or
switch to the selected camera pose. The height slider reveals enclosed caves
in the 3D view. The RGB frame and the three grayscale
white-model layers use the same view index. The layer inspector reads the saved
Node Name, raw param2, and signed face code at the selected pixel. Empty hits are
marked separately. The 3D view uses simplified blocks colored by Node Name.
Scene 0134, view 5 is an example with a black native RGB frame and valid
white-model hits.
