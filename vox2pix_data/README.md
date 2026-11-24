# Vox2Pix data collection

## THU server: adaptive 1K in one collection

[THU environment findings and complete installation/start/monitor commands](THU_SERVER.md).
The existing `VoxEvolve` on THU is the old repository and has uncommitted work;
create a separate `VoxWM` checkout. The retired AML host is no longer used.
Use `collect --mode production --scenes 200 --workers auto --adaptive-workers
--adaptive-start 4 --adaptive-window 60` for a single 1K dataset. The controller
ramps concurrency using committed-scene throughput and backs off on a sustained
drop. It retains all useful data and in-flight work, without separate pilot runs.
`auto` without `--adaptive-workers` remains a fixed CPU-count limit. Dashboard
and `collection_progress.json` expose tuning decisions and distinguish exhausted
sample budget from a measured slowdown. THU Native collection has not been run.

## Local 1K pilot and compression report

On this Mac, collect 200 scenes / 1,000 paired views with the final rendering
profile and two workers. This yields 120 Surface, 60 Cave and 20 Underwater
scenes. Use a fresh directory; repeating the same command resumes missing slots.

```bash
VOXWM_ENV_CONFIG= \
VOX2PIX_CRAFTIUM=third_party/craftium_vox2pix_local32 \
VOX2PIX_DATASET=outputs/vox2pix_pilot_1k_rgb32 \
VOX2PIX_WORKERS=2 bash scripts/collect_vox2pix.sh --renderer window --scenes 200

.venv/bin/python -m vox2pix_data compression-report \
  --dataset outputs/vox2pix_pilot_1k_rgb32
```

Voxel and White Model arrays are compressed with Blosc/Zstd during collection;
RGB is PNG. `compression-report` measures those existing files and writes
`compression_report.json` / `compression_report.md`. It reads Zarr metadata
and file sizes without decoding arrays or recompressing them. The report covers
the single Surface Node ID, raw param2, Face Direction and Water Count, raw 48³
inputs, PNGs and metadata. Auxiliary smoke lighting images are counted separately.
Partial datasets are marked `INCOMPLETE`. The 50K size estimate uses the actual
saved frame count; it does not apply the historical resolution/storage gate.
Logical stored bytes exclude filesystem padding, logs, runtime state and generated
reports. No `.tar.zst` archive is created by this reporting command.
Historical V2 datasets continue to use their original report implementation.

## Parallel production collection

`collect --mode production` uses the verified RGB32 / absolute Fog20–30
profile and the existing native scene search and single-surface projector.
The formal plan is 10,000 independent scene/world seeds × 5 views = 50,000
frames: 6,000 Surface, 3,000 Cave and 1,000 Underwater. Surface assigns 240
scenes to each of the 25 biome groups, Cave 120 each; Underwater follows natural
water without a biome quota. The actual loaded surface biome must match its
target. Unfilled targets remain missing. Each world seed has one 80/10/10
train/validation/test assignment shared by all five views.

Only the parent writes completed scenes; workers search and render independently.
Scene directories are committed atomically. Re-running the same command resumes
missing scenes and preserves completed data. Seed cursors advance before task
dispatch so an interrupted or failed attempt is not reused. Production stores
the five native RGB views and their runtime probes, without repeating lighting
A/B controls for every scene. The production data format retains the same
48³ Node ID / raw param2 and single-layer White Model / Water Count arrays.
Each committed scene records the collector Git revision and working-tree state.
A task exceeding its scene timeout plus 60 seconds for startup/cleanup stops
the run and cleans its owned processes; committed scenes remain available for resume.

The following commands are for manual execution from the repository root.
They have **not** been used to start a formal 50K run.

```bash
# Foreground collection; workers is configurable, not capped at eight.
bash scripts/collect_vox2pix.sh
# Use CPU affinity / available CPU count instead of a fixed worker limit:
VOX2PIX_WORKERS=auto bash scripts/collect_vox2pix.sh
```

Launcher variables are `VOXWM_PYTHON` (default `.venv/bin/python`),
`VOX2PIX_DATASET` (`outputs/vox2pix_50k_rgb32`), `VOX2PIX_WORKERS` (`8`),
`VOX2PIX_CRAFTIUM` (`third_party/craftium_vox2pix_fog48`) and optional
`VOXWM_ENV_CONFIG`. The latter reads only a JSON `environment` dictionary;
it does not read VAE quotas or start VAE collection. Extra CLI arguments take
precedence over launcher defaults.

For the server with user-installed Xvfb/Mesa, inherit its existing environment:

```bash
VOXWM_ENV_CONFIG=vae_data/configs/collection_64x1_clip2.json \
VOX2PIX_WORKERS=8 bash scripts/collect_vox2pix.sh
```

On a machine with system graphics dependencies, omit `VOXWM_ENV_CONFIG` and use
the inherited environment. The native build remains the isolated checkout
described below; no new engine setup or dependency installation runs in this
launcher. More workers are not guaranteed to improve throughput.

For SSH-independent execution, open a dedicated tmux session, then run the
foreground launcher **inside it**:

```bash
tmux new -s vox2pix50k -c "$PWD"
# Inside the new tmux shell:
mkdir -p runs/vox2pix_50k_rgb32
VOXWM_ENV_CONFIG=vae_data/configs/collection_64x1_clip2.json \
VOX2PIX_WORKERS=8 bash scripts/collect_vox2pix.sh \
  >runs/vox2pix_50k_rgb32/collect.log 2>&1
```

Detach with `Ctrl-B`, then `D`; the collector and its child processes stay in
the tmux session when SSH disconnects. tmux is not installed automatically.
In a separate terminal, monitor without owning any collection processes:

```bash
.venv/bin/python -m vox2pix_data dashboard --dataset outputs/vox2pix_50k_rgb32
# One snapshot, including all biome rows:
.venv/bin/python -m vox2pix_data dashboard --dataset outputs/vox2pix_50k_rgb32 --once
# Request Ctrl-C in the dedicated collector pane, allowing owned-child cleanup:
tmux send-keys -t vox2pix50k:0.0 C-c
```

Closing the dashboard does not stop collection. Resume with the same launcher
and dataset path; `--scene-ids 123 456` restricts work to missing chosen slots.
Runtime worker counts and retry budgets may change on resume; the dataset's
rendering, scene plan and seed configuration remain fixed. The dashboard shows
actual commits, category/biome coverage, in-flight tasks, failures, average and
recent rates and an empirical ETA. It does not infer a full-run throughput from
the previous empty-world clock test.

### Small local collection check

The current authorized test is at most two native worlds. With the local macOS
final-profile checkout, use one Surface slot and one Underwater slot from a
ten-scene test plan with the same 60/30/10 allocation rule:

```bash
VOX2PIX_CRAFTIUM=third_party/craftium_vox2pix_local32 \
VOX2PIX_DATASET=outputs/vox2pix_production_local2 \
VOX2PIX_WORKERS=2 bash scripts/collect_vox2pix.sh \
  --renderer window --scene-ids 0 3 --max-scenes 2 \
  --scenes 10 --worlds-per-task 1 --task-retries 0 --scene-timeout 120
```

The local test completed **2/2 native worlds**: one Surface and one Underwater,
five views each, for 10 saved RGB frames; zero failed tasks. Collection took
approximately 28.7 seconds. Its final status is `LIMIT_REACHED`, with 2 of the
10 planned test scenes committed, and the independent `dashboard --once` read
that progress successfully. Data: `outputs/vox2pix_production_local2/`; log:
`runs/vox2pix_production_local2/collect.log`.

The test used native macOS window rendering and a separate
`third_party/craftium_vox2pix_local32` build, preserving the older local engine.
It did not collect a Cave scene or test Linux software rendering at large
parallelism. The server can reuse its previously verified
`third_party/craftium_vox2pix_fog48` checkout. This small test is deliberately
partial and does not establish 50K throughput or completion time. Formal 50K
collection has not been executed.

## Final native rendering RGB32

The sole active profile is `material-ambient-fog20-draw32`, version
`vox2pix-v3-local48-rgb32-absolute-fog20-30`. Native RGB draw range is 32 nodes;
independent absolute Fog Start/End are 20/30. The 48³ observation keeps half
extent 24. Gamma 3.0, native zero-floor LUT and solid material ambient 0.35
remain unchanged. GPU Far Plane follows the original native camera calculation.
The White Model, Water Count, BACKGROUND and scene sampling are unchanged.

The final calibration and 15-frame replay completed with status
**VERIFIED WITH KNOWN NATIVE VISUAL LIMITATION**. The incremental engine build,
25 Python tests and C++ absolute-fog policy checks passed. The fixed-camera test
covered all 12 distances; all 15 saved poses matched the original Voxel/Camera,
and White/Water/Face/BACKGROUND arrays were unchanged. Formal 50K collection and
training have not started. Older RGB ranges, ratio-fog and global-floor trials
below are historical; use their explicit profiles only for their own data.

| Actual native measurement | Result |
| --- | --- |
| Map draw range / Fog Start / Fog End | 32 / 20 / 30 nodes |
| GPU Far Plane | 2000 nodes, original native calculation |
| Server send / generate / active-object limit | Each 8 MapBlocks / 128 nodes |
| Loaded MapBlocks | 68–188 overall; 87–188 for existing scenes |
| Marker geometry submitted | 31, 32 and bounding-volume-retained 34 nodes |
| Marker core at 30–34 | Native Fog Color exactly; zero RGB error |
| Maximum 22–29 shading-equation error | 0.447/255 |

All 15 native eye post-effect RGBA values match the corresponding historical
fog48 C views; Underwater remains [24,46,89,60]. The original underwater tint is
preserved. Visual inspection covered all 12 marker distances and 15 RGB frames. Native Sky
and Fog Color can leave a faint uniform patch after marker texture disappears
(one center/background RGB difference was [3,2,0]). Native underwater surface
polygon boundaries remain visible. This is not a claim that every fogged surface
matches every background pixel or that water-surface boundaries disappeared.
See the [measurement report](reports/final_rendering_32/README.md); its offline
bundle is [final_rendering_32.zip](reports/final_rendering_32/final_rendering_32.zip).

From the repository root, using the existing native client checkout and its
Linux graphics environment:

```bash
BUILD_THREADS=22 bash scripts/build_vox2pix_fog48.sh third_party/craftium_vox2pix_fog48
python3 scripts/test_vox2pix_fog48.py --craftium-dir third_party/craftium_vox2pix_fog48
.venv/bin/python -m unittest vox2pix_data.test_final_protocol vox2pix_data.test_final_report -q

.venv/bin/python -m vox2pix_data.final_rendering \
  --dataset outputs/vox2pix_v3_lightfloor_15 \
  --craftium-dir third_party/craftium_vox2pix_fog48 \
  --output outputs/vox2pix_final_rendering_32 \
  --scene-timeout 240
```

The checkout's historical `fog48` name identifies the build directory; its
absolute-fog patch accepts 32 independently of Fog End. No shader rewrite is
needed. On the server with user-installed Xvfb, launch the same command with
the recorded environment:

```bash
.venv/bin/python - <<'PYENV'
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
    "--output", "outputs/vox2pix_final_rendering_32",
    "--scene-timeout", "240",
])
PYENV
```

A fixed-camera disposable calibration world presents one marker at a time at
10/18/20/22/24/25/27/29/30/31/32/34 nodes to prevent occlusion. It captures Fog OFF
and final Fog20–30 without moving the camera. Existing Surface/Cave/Underwater
seeds and all five saved poses per kind supply the 15 final RGB frames; no new
world search runs. Native probes verify actual ranges, shader endpoints,
server availability and lighting. Raw saved 48³ Node ID / param2 are checked
at every replayed pose. White, Water Count and BACKGROUND previews use the
original read-only arrays.

Open `outputs/vox2pix_final_rendering_32/report/index.html` offline. It links the
calibration report and shows RGB, White Model, Water Count, BACKGROUND and pose
metadata. `manifest.json` records the final settings and source;
`summary.json` separates native measurements and visual status. A 34-node
marker can survive nominal range culling due to MapBlock bounding volumes;
any retained geometry must still be hidden by Fog. Preserve the entire output
directory when copying the offline report. Use a new output path for each replay.

## Historical V3 global-LUT smoke

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

## Historical draw30 fog calibration

The deprecated draw30 smoke profile is `material-ambient-fog20`, versioned separately
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

The draw30 profile remains available as `material-ambient-fog20` for legacy
replay. Its range is recorded independently from the historical draw48 profile.

## Independent RGB48 and absolute fog

This deprecated historical profile is `material-ambient-fog20-draw48`, version
`vox2pix-v3-local48-rgb48-absolute-fog20-30`. Its 48³ voxel observation
keeps half extent 24. Native RGB drawing uses 48 blocks and the absolute
Fog Start/End settings use 20/30 blocks. Material lighting is unchanged.
The native camera far plane follows its original calculation.

Use a separate configured checkout at Craftium
`ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49` and the pinned VoxeLibre game.
In the existing Linux software-rendering environment, build and run from
the repository root.

```bash
BUILD_THREADS=22 bash scripts/build_vox2pix_fog48.sh third_party/craftium_vox2pix_fog48
python3 scripts/test_vox2pix_fog48.py --craftium-dir third_party/craftium_vox2pix_fog48
.venv/bin/python -m unittest vox2pix_data.test_fog48_report vox2pix_data.test_v3_projection -q

.venv/bin/python -m vox2pix_data.fog_range48 \
  --dataset outputs/vox2pix_v3_lightfloor_15 \
  --craftium-dir third_party/craftium_vox2pix_fog48 \
  --output outputs/vox2pix_fog48/replay/fog_range_48_test \
  --scene-timeout 180
```

The offline report is `OUTPUT/index.html`. Calibration uses one
temporary native wool panel at 14 camera distances with Fog OFF/draw48,
Fog20–30/draw30 and Fog20–30/draw48. The three recorded scenes reuse
their existing five poses and compare the two fogged draw ranges. The
White Model and Water Count previews read the original saved arrays.
Native probes record actual loaded and submitted solid-mesh MapBlocks.
Twelve fixed24 ticks after each range switch allow the draw list to refresh.

For the existing server's user-installed Xvfb, the exact environment-aware
replay command is below.

```bash
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
import sys

os.environ.update(json.loads(Path("vae_data/configs/collection_64x1_clip2.json").read_text())["environment"])
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.execv(sys.executable, [
    sys.executable, "-B", "-u", "-m", "vox2pix_data.fog_range48",
    "--dataset", "outputs/vox2pix_v3_lightfloor_15",
    "--craftium-dir", "third_party/craftium_vox2pix_fog48",
    "--output", "outputs/vox2pix_fog48/replay/fog_range_48_test",
    "--scene-timeout", "180",
])
PY
```

For an explicit historical draw48 reproduction, use `--mode smoke
--lighting-profile material-ambient-fog20-draw48` with a fresh dataset directory.
It is deprecated for new collection. Previous datasets retain their original profile.

For a small RGB-only preview of saved View 3/4/5 at draw range 128, with
Fog OFF and absolute Fog 20–30, use the same renderer environment:

```bash
.venv/bin/python -m vox2pix_data.replay_rgb \
  --dataset outputs/vox2pix_v3_lightfloor_15 \
  --craftium-dir third_party/craftium_vox2pix_fog48 \
  --output outputs/vox2pix_rgb128 \
  --viewing-range 128 --views 3 4 5
```

This replays Surface/Cave/Underwater and writes 18 PNGs plus `index.html`.
Use `--kinds surface` to limit the preview. The production profile stays unchanged.

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
