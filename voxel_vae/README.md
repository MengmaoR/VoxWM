# VoxWM E00-Masked Voxel VAE

E00-Masked is the active VAE experiment. It uses 160 training states and 160
held-out validation states for an implementation sanity check. Historical
Plain results are retained. Masked training and reconstruction use
`phase_masked/` under the prepared output root.

## Objective and data

```text
loss = 0.50 * Static CE
     + 0.25 * Dynamic-Unchanged CE
     + 0.25 * Dynamic-Changed CE
     + 1e-6 * raw KL
```

Each dynamic mask compares its canonical center 16³ tokens with the adjacent
states of the same complete five-state chain. CE is normalized per sample
within each mask before averaging samples and D1–D4. The 64-state effective
batch is 32 Static and 8 of each dynamic category. Micro-batch contributions
share the effective batch denominators.

The input is the canonical Combined Token ID of each center voxel. Dataset
roots keep their original vocabularies and Zarr arrays. IDs are remapped on
read by `(Node Name, raw param2)`, and the existing world-seed split applies.
The fixed 3D ResNet VAE uses a 32D token embedding and a `[16,4,4,4]` posterior.

## Prepare and train manually

Run from the VoxWM repository root. The examples use `.venv-vae/bin/python`,
which needs PyTorch with CUDA and the dependencies in `requirements.txt`.
Reuse the existing prepared output for a resume. Prepare once for a new
output root.

```bash
.venv-vae/bin/python -m voxel_vae.e00 prepare \
  --dataset-root outputs/vae_100k_clip1 \
  --dataset-root outputs/vae_100k_clip2 \
  --output-root outputs/voxel_vae/e00
```

Start a new Masked run in an empty run directory.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vae/bin/python -m voxel_vae.e00 run-masked \
  --output-root outputs/voxel_vae/e00 --device cuda:0
```

Resume an interrupted run from its `phase_masked/checkpoint_last.pt`.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vae/bin/python -m voxel_vae.e00 run-masked \
  --output-root outputs/voxel_vae/e00 --device cuda:0 --resume
```

`CUDA_VISIBLE_DEVICES` selects the physical card. `cuda:0` is its index in
the training process. Default validation is every 500 steps, with a
5,000-step cap. Starting at step 500, two consecutive validation checks
without a meaningful drop (greater than `0.001`) in total loss trigger early
stopping. Set the policy in the `early_stopping` block of the configuration
used by `prepare`. Runs prepared before this block existed use the repository
defaults. The resume checkpoint carries the early-stopping state.

## Monitor and inspect loss

The trainer saves `metrics.jsonl`, `loss_curves.svg`, `loss_curves.json`,
`early_stopping.json`, and `run_metadata.json` in `phase_masked/`. The curves
include training and validation total losses, the reconstruction terms,
D1–D4 changed-region CE, and raw and weighted KL. By default, training
points are logged every 25 optimizer steps and validation points every 500.
On resume, the curves follow the restored checkpoint from that step forward.
Raw logs retain earlier interrupted attempts.

Start the read-only monitor in a separate server terminal.

```bash
python3 -m voxel_vae.training_monitor \
  --output-dir outputs/voxel_vae/e00/phase_masked --port 8780
```

Forward the monitor port from your computer and open [the monitor](http://127.0.0.1:8780).
It refreshes every five seconds and displays the latest logged step, losses,
best validation step, and stopping reason. The monitor has its own process.

```bash
ssh -N -L 8780:127.0.0.1:8780 aml-g06-gpu
```

`checkpoint_best.pt` stores the lowest measured validation total loss.
`checkpoint_last.pt`, `final_metrics.json`, and `report.md` describe the
last validation step, including when training ends early.

## Open the best-checkpoint reconstruction viewer

Export eight Static examples and two complete chains from each of D1–D4
from the held-out validation subset. The default checkpoint is
`phase_masked/checkpoint_best.pt`.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vae/bin/python -m voxel_vae.recon_viewer export \
  --output-root outputs/voxel_vae/e00 --device cuda:0

.venv-vae/bin/python -m voxel_vae.recon_viewer serve \
  --output-root outputs/voxel_vae/e00 --port 8770
```

On your computer, forward `8770` and open [the viewer](http://127.0.0.1:8770).
The viewer shows paired GT and Masked reconstruction, supports axis slicing,
and preserves the order of the five-state chains.

```bash
ssh -N -L 8770:127.0.0.1:8770 aml-g06-gpu
```
