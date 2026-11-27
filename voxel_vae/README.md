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
only at a completed epoch.

With `--resume --epochs N`, complete N more epochs from the last completed
epoch. An in-progress epoch counts once it finishes. From step 25,000, use
`--resume --epochs 10` to target step 50,000. If a run is interrupted and
the checkpoint has saved the current epoch plan, use `--resume` alone to continue
toward that target at the saved offset. For a fresh run, use `--epochs N`
from step zero.

Add `--no-early-stop` for a bounded continuation with validation and best
checkpoint selection active. From step 27,500, use
`--resume --epochs 1 --no-early-stop` to target step 30,000. If interrupted
before the last checkpoint reaches 30,000, reuse all three flags. The
target is counted from the last completed epoch at step 27,500.

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

## Freeze audit of the 27,000-step checkpoint

Use the prepared E01 directory with the 160K/20K/20K split and its
validation-selected `checkpoint_best.pt`. Run from the repository root
with the PyTorch and dataset dependencies installed.

```bash
CUDA_VISIBLE_DEVICES=4 .venv-vae/bin/python -m voxel_vae.freeze_audit all \
  --e01-root outputs/voxel_vae/e01 \
  --checkpoint outputs/voxel_vae/e01/phase_masked/checkpoint_best.pt \
  --device cuda:0 --batch-size 64 --workers 4
```

The `test` and `d3` commands share a single posterior-mean pass over the
full held-out test split. `embedding` scans center-token frequencies and
compares the 27k embedding to its recorded initialization. `latent` computes
channel statistics on the full train split. `report` reads the completed
outputs. The `all` command reuses token counts collected during inference
and any completed audit files for the same checkpoint and split.

Results go to `outputs/voxel_vae/e01/freeze_audit/`. Downstream code can
import `normalize_latent` and `denormalize_latent` from
`voxel_vae.latent_normalization` and use the train-set statistics in
`latent_normalization.json`.

## History

`outputs/voxel_vae/e00/` retains the Plain and Masked implementation sanity runs.
E00-Masked used the 160-state E00-B training subset. Its checkpoint and
curves belong to that separate experiment.

## Token-anchored split for three-block training

Freeze the three eligible 16³ blocks per saved 48³ snapshot before splitting.
Static maps use center plus two rare blocks. Dynamic maps use center, one
motion-ranked block and one rarity-ranked block.

For every observed `(node_name, raw param2)`, draw one ordinal uniformly from
all matching voxels in the frozen blocks. The anchor records the snapshot and
block, then the matching voxel's zero-based ordinal within that block in ENU
C-order. The entire anchor world seed goes into Train. The remaining worlds
are assigned randomly under exact 80/10/10 quotas for each Static biome and
height cell and each Dynamic category. Each five-frame chain shares its split.
The final pass checks the actual token sets in the frozen Train, Validation and
Test blocks.

Run the read-only analysis from the repository root on the original category
packs. This historical dry run uses the saved Vox2Pix counts and the three
block indices in the proposed replacement plan.

```bash
PACKS=/Users/tianxj/Desktop/VoxWM-Dataset/VAE/data/vae_200k_by_category
PLAN=outputs/vae_static_replacement_20_20261127
SOURCE=outputs/vox2pix_substitution_20261126
.venv/bin/python -m voxel_vae.freeze_split_chunks \
  "$PACKS/vae_100k_static.tar.zst" "$PACKS/vae_100k_dynamic.tar.zst" \
  --replacement-plan "$PLAN/recapture_manifest.json" \
  --map-inventory "$SOURCE/maps.json" --static-catalog "$SOURCE/static.jsonl.gz" \
  --output "$PLAN/frozen_chunks_for_token_split"
.venv/bin/python -m voxel_vae.token_anchored_split \
  "$PLAN/frozen_chunks_for_token_split/frozen_chunks.json" \
  --output "$PLAN/token_anchored_dry_run" --seed 42
```

The split tool reads a frozen index with semantic token pairs and a block
histogram catalog. Its outputs include `token_anchors.csv`,
`world_seed_splits.csv.gz`, `quota_counts.csv` and
`token_voxel_counts_by_split.csv`. The E01 training commands above use
their original center-only index.

The 2026-11-27 replacement dry-run selected 600,000 blocks and reserved one
anchor for each of the 1,668 observed tokens. The anchors covered 1,358 world
seeds. The 480,000 Train blocks contained every observed token. The 60,000
Validation blocks and 60,000 Test blocks each had zero tokens absent from Train.

### Import the original Vox2Pix maps

The saved maps contain the original node names and raw `param2`. Import them
as 48³ VAE Static snapshots with `uint16` tokens. The `uint8` light field is
zero-filled and marked as unobserved in the overlay manifest. The importer
updates the full 48³ frequency by subtracting each archived original and adding
its replacement. It then reranks every Static and Dynamic snapshot with the
updated frequency, including the imported maps. Coverage uses the positive
Train counts from `split/token_voxel_counts_by_split.csv`.

```bash
PACKS=/Users/tianxj/Desktop/VoxWM-Dataset/VAE/data/vae_200k_by_category
.venv/bin/python -m voxel_vae.import_archived_static \
  --plan-root outputs/vae_static_replacement_20_20261127 \
  --map-inventory outputs/vox2pix_substitution_20261126/maps.json \
  --base-archives "$PACKS/vae_100k_static.tar.zst" "$PACKS/vae_100k_dynamic.tar.zst" \
  --output /Users/tianxj/Desktop/VoxWM-Dataset/VAE/data/vae_vox2pix_reranked_20261127
```

`dataset_overlay.json` links the original packs, the original samples held in
archive and the replacement files. `frozen_chunks.json` records the three-block
index and `split/` contains the new whole-seed assignments and token counts.
