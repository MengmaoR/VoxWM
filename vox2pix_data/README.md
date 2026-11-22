# Vox2Pix 1K compression pilot

Use the existing native GT environment on Linux, with the repository's
fixed24 integer-origin capture patch built into Craftium. Install the existing
`vae_p0/requirements.txt` for NumPy, Zarr v2, Blosc and Pillow if they are
missing. The CPU projector also needs a C++17 compiler (`CXX` defaults to
`c++`). A software session uses the existing Xvfb setup. See
[the repository setup](../README.md) and the
[Vox2Pix protocol](../VOX2PIX_DATA_COLLECTION_PROTOCOL.md).

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
sequence for those slots. After all 200 scenes are committed, run:

```bash
.venv/bin/python -m vox2pix_data check --dataset outputs/vox2pix_pilot_1k_640x360
.venv/bin/python -m vox2pix_data compression-report --dataset outputs/vox2pix_pilot_1k_640x360
```

The last command writes `compression_report.json` and
`compression_report.md` under the dataset root. The compression gate is
based on the validated complete 1K pilot. The pilot remains **NOT NATIVE VERIFIED**.

The root contains `manifest.json` and `node_vocabulary.json`. Each committed
`scenes/scene_NNNN/` contains `scene.json`, `voxels.zarr/` and
`rgb/000.png` through `rgb/004.png`. Read frame `i` from each white array by
indexing it with the same `i` used in the PNG filename and `scene.json` views.
