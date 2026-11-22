# Vox2Pix 1K compression pilot

The active protocol is `vox2pix-pilot-v2-camera16`. Use a newly initialized pilot
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
gates. After all 200 scenes are committed, run:

```bash
.venv/bin/python -m vox2pix_data check --dataset outputs/vox2pix_pilot_1k_640x360
.venv/bin/python -m vox2pix_data compression-report --dataset outputs/vox2pix_pilot_1k_640x360
```

The last command writes `compression_report.json` and
`compression_report.md` under the dataset root. The compression gate is
based on the validated complete 1K pilot. The pilot remains **NOT NATIVE VERIFIED**.

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
