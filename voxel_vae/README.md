# VoxWM E00 — Combined-Token Voxel VAE

E00 implements the minimum VAE sanity loop only: canonical semantic token
remapping, committed-snapshot indexing, a 32D embedding, the fixed 3D ResNet
VAE, equal-voxel CE + `1e-6` raw KL, balanced sampling, strict checkpoints and
tiny real-data overfits. It does not train dynamics and does not authorize E01.

## Exact E00 objective

For target token grid `x` and posterior tensors `mu, logvar` of shape
`[B,16,4,4,4]`, the implemented loss is:

```text
recon_ce = mean CrossEntropy(logits, x) over B * 16 * 16 * 16 voxels
kl_raw   = mean_B sum_{16 * 4 * 4 * 4} -0.5 * (1 + logvar - mu^2 - exp(logvar))
loss     = recon_ce + 1e-6 * kl_raw
```

There is no class weighting, non-Air weighting, changed-voxel mask, temporal
loss, chain-order input, perceptual term or category-specific auxiliary loss.
The 32/8/8/8/8 sampler balances **states per category**, not the voxels inside
each state. In the completed E00-B subset, GT-changing voxels account for only
about 0.23% of adjacent dynamic training pairs (about 0.29% in validation), so
plain CE is dominated by unchanged structure. This is an intentional E00
sanity protocol limitation, not evidence that the stored dynamic GT is bad.

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
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

The final check must report CUDA available before running E00 on a GPU.
Commands below explicitly select physical GPU `0`. To use another card, change
only `CUDA_VISIBLE_DEVICES=0` to that physical GPU ID; keep `--device cuda:0`
because the selected card is re-indexed to device zero inside the process.

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
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m voxel_vae.e00 run-a \
  --output-root outputs/voxel_vae/e00 \
  --device cuda:0
```

Resume only a compatible E00-A checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m voxel_vae.e00 run-a \
  --output-root outputs/voxel_vae/e00 \
  --device cuda:0 --resume
```

E00-B is rejected unless E00-A reaches at least 99% exact overall and non-Air
accuracy. It uses the fixed 160-state train subset, a separate 160-state
validation diagnostic subset, and 32/8/8/8/8 effective batches:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m voxel_vae.e00 run-b \
  --output-root outputs/voxel_vae/e00 \
  --device cuda:0
```

Regenerate the report without training:

```bash
.venv/bin/python -m voxel_vae.e00 report \
  --output-root outputs/voxel_vae/e00
```

## Reconstruction viewer

After E00-B finishes, export a small deterministic gallery from the fixed
**validation** subset using the best checkpoint. It contains eight Static
chunks and two complete five-state clips from each of D1–D4 (48 states total).
The held-out test split is deliberately not used.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vae/bin/python -m voxel_vae.recon_viewer export \
  --output-root outputs/voxel_vae/e00 \
  --device cuda:0
```

Serve the read-only side-by-side 3D viewer on the GPU server:

```bash
.venv-vae/bin/python -m voxel_vae.recon_viewer serve \
  --output-root outputs/voxel_vae/e00 \
  --host 127.0.0.1 --port 8770
```

Forward the port from the local computer and open
`http://127.0.0.1:8770`:

```bash
ssh -N -L 8770:127.0.0.1:8770 aml-g06-gpu
```

The viewer switches among Static and D1–D4, preserves each dynamic chain's
five-state order, renders GT input and posterior-mean reconstruction side by
side, highlights exact Combined-Token errors, and provides axis slicing for
solid/underground chunks. It only reads derived reconstruction artifacts.

New runs also save `embedding_initial.npz` before optimizer step 1 and report
row-wise initialization-to-final drift. Evaluation reports dynamic change
metrics both in aggregate and separately for D1, D2, D3 and D4. These are
diagnostics only and do not change the E00 objective.

## E00-Masked paired objective

E00-Plain remains unchanged at the output root. E00-Masked writes only to
`phase_masked/` and starts from a fresh seed-42 initialization; it never loads
the trained Plain checkpoint. For each dynamic state, a CPU-only mask is built
from its complete canonical-token five-state chain: endpoints compare with the
single neighbor, while middle states take the union of differences to the
previous and next states. Masks use only the center 16³ and never enter the
model.

```text
loss = 0.5 * mean Static sample CE
     + 0.25 * mean-of-D1..D4 per-sample Unchanged CE
     + 0.25 * mean-of-D1..D4 per-sample Changed CE
     + 1e-6 * original raw KL
```

Every per-mask CE is normalized inside its sample before samples and D1–D4 are
averaged. With micro-batch 2, contributions use denominators from the complete
64-state effective batch; the 32 backward calls are summed without another
micro-batch divisor.

Run the single authorized Masked experiment on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vae/bin/python -m voxel_vae.e00 run-masked \
  --output-root outputs/voxel_vae/e00 --device cuda:0
```

Resume only its own objective-compatible checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vae/bin/python -m voxel_vae.e00 run-masked \
  --output-root outputs/voxel_vae/e00 --device cuda:0 --resume
```

Evaluate both best checkpoints on the identical validation subset and export
the paired viewer archive:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vae/bin/python -m voxel_vae.comparison \
  --output-root outputs/voxel_vae/e00 --device cuda:0

CUDA_VISIBLE_DEVICES=0 .venv-vae/bin/python -m voxel_vae.recon_viewer export-comparison \
  --output-root outputs/voxel_vae/e00 --device cuda:0
```

The usual `serve` command automatically opens
`viewer_comparison_examples.npz` when it is selected explicitly:

```bash
.venv-vae/bin/python -m voxel_vae.recon_viewer serve \
  --output-root outputs/voxel_vae/e00 \
  --examples outputs/voxel_vae/e00/viewer_comparison_examples.npz \
  --host 127.0.0.1 --port 8770
```

The comparison overlay uses the real changed mask: green marks a Plain error
fixed by Masked, magenta a Masked regression, red an error shared by both, and
orange an error in the selected model's unchanged region.

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
.venv/bin/python -m unittest voxel_vae.tests.test_recon_viewer -v
python3 -m unittest voxel_vae.tests.test_model voxel_vae.tests.test_masked_loss -v
```
