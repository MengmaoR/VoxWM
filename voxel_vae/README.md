# VoxWM E01-Masked Voxel VAE

E01-Masked uses the full 160,000-state train split from both completed
100K clips. The new split adds 274 states to train and 380 states to
validation from the original test pool, transferring complete world seeds
across both clips. Validation and test each contain 20,000 states. All
original train and validation rows stay in their splits. The existing
1,280 validation states and 160 training diagnostics stay fixed.

The `--epochs` mode shuffles the training pools and visits every state
once per epoch. The step-based mode draws 32 Static and 8 each of D1–D4
independently per 64-state batch.

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
and creates a canonical vocabulary, index, split overlay and fixed diagnostic
subsets.
The preparation output prints the exact pool size and composition.

```bash
git pull --ff-only origin master

.venv-vae/bin/python -m voxel_vae.e01 prepare \
  --dataset-root outputs/vae_100k_clip1 \
  --dataset-root outputs/vae_100k_clip2 \
  --output-root outputs/voxel_vae/e01
```

Start a fresh E01 run with seed 42 on GPUs 2 and 3. The two workers share a
global batch of 64. Each worker processes 32 states in 16 micro-batches, and
the final backward synchronizes gradients. Validation, early stopping and
artifact writes run on rank zero.

```bash
CUDA_VISIBLE_DEVICES=2,3 .venv-vae/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 -m voxel_vae.e01 run \
  --output-root outputs/voxel_vae/e01
```

For an interrupted E01 run on the same split, reuse the prepared root and
add `--resume` to the same command. That loads `phase_masked/checkpoint_last.pt`,
optimizer, step, early stopping state and each worker's RNG state.
A single-GPU checkpoint can resume on two GPUs using a fresh RNG stream
on the additional worker. A GPU count change alters the latent sampling sequence.

### Rebalance the existing E01 run and continue with complete epochs

Activate the split in the existing E01 directory. This writes
`split_160k.json` alongside the original index and fixed diagnostic
subsets. The clips and checkpoints remain on their existing paths.

```bash
git pull --ff-only origin master && \
  .venv-vae/bin/python -m voxel_vae.e01 rebalance --output-root outputs/voxel_vae/e01
```

Start the epoch schedule at `checkpoint_last.pt`. From the completed
20,000-step checkpoint, this targets step 25,000 with two full passes.

```bash
CUDA_VISIBLE_DEVICES=2,3 .venv-vae/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 -m voxel_vae.e01 run \
  --output-root outputs/voxel_vae/e01 --resume --epochs 2
```

Each epoch covers all 160,000 training states across 2,500 optimizer steps
at global batch size 64. The sampler spreads category remainders across
batches and uses the actual global Masked loss denominators. A checkpointed
mid-epoch resume continues at the saved offset.

The optimizer, global step and lowest validation loss carry over. Patience
restarts when switching to epoch sampling. Validation stays on the 1,000-step
cadence with extra checks at epoch boundaries, and early stopping takes effect
only at a completed epoch. Reuse the same command after interruption.
`--epochs` is the total target within the schedule started at step 20,000.
If the 25,000-step curve warrants continuing, change `--epochs 2` to
`--epochs 12` to target step 50,000. With a saved epoch plan, `--resume`
alone restores its target and offset. For a fresh run, supply `--epochs`
without `--resume` to use complete epochs from step zero.

Use the single-process command to train on GPU 2 alone.

```bash
CUDA_VISIBLE_DEVICES=2 .venv-vae/bin/python -m voxel_vae.e01 run \
  --output-root outputs/voxel_vae/e01 --device cuda:0
```

Training defaults are bf16 and AdamW at `1e-4`. Step-based runs have a
20,000 optimizer-step cap.
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
CUDA_VISIBLE_DEVICES=2 .venv-vae/bin/python -m voxel_vae.recon_viewer export \
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
