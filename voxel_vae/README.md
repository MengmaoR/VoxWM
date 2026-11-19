# VoxWM E00 — Combined-Token Voxel VAE

E00 implements the minimum VAE sanity loop only: canonical semantic token
remapping, committed-snapshot indexing, a 32D embedding, the fixed 3D ResNet
VAE, equal-voxel CE + `1e-6` raw KL, balanced sampling, strict checkpoints and
tiny real-data overfits. It does not train dynamics and does not authorize E01.

## Data contract

- One or more `vae-region48-fixed24-v2` roots are accepted.
- Every root keeps its own immutable local vocabulary.
- Training IDs are a sorted union of exact `(Node Name, raw param2)` pairs;
  local IDs are remapped while loading and raw Zarr is never rewritten.
- Only `snapshots.zarr/tokens[:,16:32,16:32,16:32]` enters the model.
- `light`, biome/category/time metadata, sequence order and D4 anchors do not.
- Existing world-seed `train/validation/test` labels are authoritative.

Install data dependencies, then install/verify a PyTorch build matching the
server CUDA. The requirements file deliberately does not replace PyTorch:

```bash
.venv/bin/python -m pip install -r voxel_vae/requirements.txt
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The final check must report CUDA available before running E00 on a GPU.

## E00 commands

Prepare the deterministic vocabulary, committed-row index and fixed subsets.
This is CPU-only and does not alter either dataset:

```bash
.venv/bin/python -m voxel_vae.e00 prepare \
  --dataset-root outputs/vae_100k_clip1 \
  --dataset-root outputs/vae_100k_clip2 \
  --output-root outputs/voxel_vae/e00
```

Run E00-A in fp32. It repeats the same eight real train samples for at most
3,000 optimizer steps:

```bash
.venv/bin/python -m voxel_vae.e00 run-a \
  --output-root outputs/voxel_vae/e00 \
  --device cuda
```

Resume only a compatible E00-A checkpoint:

```bash
.venv/bin/python -m voxel_vae.e00 run-a \
  --output-root outputs/voxel_vae/e00 \
  --device cuda --resume
```

E00-B is rejected unless E00-A reaches at least 99% exact overall and non-Air
accuracy. It uses the fixed 160-state train subset, a separate 160-state
validation diagnostic subset, and 32/8/8/8/8 effective batches:

```bash
.venv/bin/python -m voxel_vae.e00 run-b \
  --output-root outputs/voxel_vae/e00 \
  --device cuda
```

Regenerate the report without training:

```bash
.venv/bin/python -m voxel_vae.e00 report \
  --output-root outputs/voxel_vae/e00
```

Outputs follow the requested layout. E00-A artifacts are additionally retained
under `phase_a/`; the root checkpoints/metrics/reconstructions belong to E00-B.
The checked-in `configs/e01.yaml` is explicitly marked `authorized: false` and
is not an execution entry point.

## Local tests

The collection virtual environment currently has Zarr but no Torch, while the
local system Python has Torch but no Zarr. The two focused suites therefore run
independently:

```bash
.venv/bin/python -m unittest voxel_vae.tests.test_data -v
python3 -m unittest voxel_vae.tests.test_model -v
```
