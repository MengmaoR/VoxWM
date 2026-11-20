# VoxWM E01-Masked Voxel VAE

E01-Masked uses the full `train` split from both completed 100K clips. The
balanced sampler draws 32 Static and 8 each of D1–D4 per 64-state batch.
The pool size comes from the committed index, typically about 160K states.
Sampling is step-based across the full pool. The existing world-seed split
keeps validation and test worlds separate.

The model takes the canonical Combined Token IDs of the center 16³ only.
E01 reuses the 3D ResNet VAE and the Masked objective from E00.

```text
loss = 0.50 * Static CE
     + 0.25 * Dynamic-Unchanged CE
     + 0.25 * Dynamic-Changed CE
     + 1e-6 * raw KL
```

Changed masks use the adjacent states in each complete five-state chain.
Each chain is read on first use for its mask, and its masks are cached
losslessly in memory. Per-sample loss terms use the effective-batch
denominators across micro-batches of two.

## Prepare and train manually

Run from the VoxWM repository root. Use the Python environment with both
PyTorch/CUDA and `voxel_vae/requirements.txt`. The commands use `.venv-vae`.

Prepare once in a new E01 output directory. This reads the dataset metadata
and creates a canonical vocabulary, index, and fixed diagnostic subsets.
The preparation output prints the exact pool size and composition.

```bash
git pull --ff-only origin master

.venv-vae/bin/python -m voxel_vae.e01 prepare \
  --dataset-root outputs/vae_100k_clip1 \
  --dataset-root outputs/vae_100k_clip2 \
  --output-root outputs/voxel_vae/e01
```

Start a fresh E01 run with seed 42.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vae/bin/python -m voxel_vae.e01 run \
  --output-root outputs/voxel_vae/e01 --device cuda:0
```

For an interrupted E01 run, reuse the prepared root and add `--resume`.
That loads `phase_masked/checkpoint_last.pt` and its optimizer, RNG, and early
stopping state. Checkpoint compatibility includes the exact data index and
fixed diagnostic subsets. `CUDA_VISIBLE_DEVICES` selects the physical GPU.
`cuda:0` is that card's index inside the process.

Training defaults are bf16, AdamW at `1e-4`, at most 20,000 optimizer steps.
Every 1,000 steps, the model evaluates 640 Static states and 32 complete
five-state chains apiece for D1–D4 from the held-out validation split.
The fixed 160-state train diagnostic subset is used for reconstruction
diagnostics. `checkpoint_best.pt` tracks the lowest validation total loss.
Early stopping uses the same validation set and stops after two consecutive
checks without a decrease greater than `0.001`. The test split stays held out.

## Monitor loss and reconstructions

Artifacts go to `outputs/voxel_vae/e01/phase_masked/`. The saved
`loss_curves.svg` and `loss_curves.json` include the total and component
losses, D1–D4 changed-region CE, and raw and weighted KL. The same folder
holds the checkpoints, validation diagnostics, and `report.md`.

Start the read-only monitor in another server terminal.

```bash
python3 -m voxel_vae.training_monitor \
  --output-dir outputs/voxel_vae/e01/phase_masked --port 8780
```

Forward port `8780` from your computer and open [the monitor](http://127.0.0.1:8780).

```bash
ssh -N -L 8780:127.0.0.1:8780 aml-g06-gpu
```

Export validation reconstructions from the best E01 checkpoint and open
the existing 3D viewer.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vae/bin/python -m voxel_vae.recon_viewer export \
  --output-root outputs/voxel_vae/e01 --device cuda:0

.venv-vae/bin/python -m voxel_vae.recon_viewer serve \
  --output-root outputs/voxel_vae/e01 --port 8770
```

On your computer, forward `8770` and open [the viewer](http://127.0.0.1:8770).

```bash
ssh -N -L 8770:127.0.0.1:8770 aml-g06-gpu
```

## History

`outputs/voxel_vae/e00/` retains the Plain and Masked implementation sanity runs.
E00-Masked used the 160-state E00-B training subset. Its checkpoint and
curves belong to that separate experiment.
