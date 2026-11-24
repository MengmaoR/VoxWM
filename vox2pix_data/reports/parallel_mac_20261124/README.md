# Mac concurrency comparison — 2026-11-24

Recommendation for this Mac with its other applications left running: **4 workers**.
No collection algorithm, sampling rules, renderer or Native engine was changed for this comparison.

## Workload and measurements

Apple M1 (4 performance + 4 efficiency cores), 16 GiB unified memory, macOS native window renderer.
Tested source: `c272b4b5d3c7d29dfeae4491e36d329950406600`.
Engine: `third_party/craftium_vox2pix_local32`, the existing independently built final32 engine.
The previously recorded upstream base is Craftium `ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49`
with the existing fixed24/region/batch/lighting/material/fog patches. It was not rebuilt here.

Sequential 4 → 6 → 8 workers, fresh output directories, identical first 30 scene slots
from the 200-scene seed-42 plan. Each stage completed **18 Surface + 9 Cave + 3 Underwater**,
five 640×360 paired views each: **150 frames/stage, 450 frames total**.
The underlying 200-scene plan was retained so changing worker count did not change seed assignments.
Each CLI returned 0 / `LIMIT_REACHED`; that status means the requested 30/200 slots completed.
No formal 50K/100K collection, compression rerun or training was started.

| Workers | Full CLI wall seconds | Frames/hour | Middle frames/hour | Peak tree RSS GiB | Swap-in / out Δ GiB | 100K hours, linear |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 292.44 | 1846.5 | 1969.1 | 2.03 | 0.22 / 0.83 | 54.2 |
| 6 | 322.35 | 1675.2 | 1569.8 | 3.00 | 0.75 / 1.53 | 59.7 |
| 8 | 425.70 | 1268.5 | 1249.8 | 3.01 | 1.03 / 1.18 | 78.8 |

Full CLI time includes startup, search/rejection/retry, export, tail drain and cleanup,
measured by the outer stopwatch with up to one five-second polling interval of timing quantization.
The common middle window is commit 8 → commit 22: `(22−8)×5 / elapsed`.
This window supports the same ranking; 8-worker slowdown is not just the final slow scene.
Thirty scenes are a short comparison, not a long-run stability or optimum proof.

The historical **2-worker 1K reference** completed 1,000 frames in 2,453.966 active seconds
across two collector invocations: **1,467.0 frames/hour**, or **68.2 hours/100K** linearly.
The repair/waiting gap is excluded. Its full 200-scene workload differs from this first-30 subset,
so its ratio to these short stages is a reference rather than a paired speedup measurement.
It was not rerun.

## Resource and retry observations

- At full task occupancy, median collector-tree CPU / whole-machine summed process CPU
  were 344.85% / 671.75% (4), 386.70% / 685.40% (6), and 374.00% / 669.80% (8).
  These are `ps` process CPU statistics, not precise instantaneous core utilization.
  Increasing concurrency did not produce proportional useful CPU throughput.
- All stages reached memory-pressure warning level 2; none reached critical level 4 or
  triggered the short-test stop protection. Other applications were not closed.
  RSS is a sum over the process tree and can count shared pages repeatedly; it excludes
  WindowServer and does not measure unique physical or unified-GPU memory consumption.
- Every stage incurred actual swap activity. Existing swap stock was already about 10.6 GiB
  before stage 4. Sequential tests alter caches and resident/background pages, so the delta
  column is not an isolated ranking of each stage's total memory requirement. No claim of
  zero UI impact or long-run memory stability is made.
- Native world logs contained 6 / 16 / 7 connection-closed records for 4 / 6 / 8 workers.
  These are rejected internal world attempts, not lost committed scenes. Scene-task failures
  were 0 / 0 / 1. Stage 8 retried Cave scene 11 after a search exhausted at 186.5 seconds;
  all 30 requested scenes eventually committed. Its saved `last_error` is historical, not
  a failed final status. Do not sum cumulative counters embedded in rejection messages.
- System thermal queries recorded no thermal/performance warning, but did not provide a
  numeric throttling measurement. Resource contention, startup failures and retries are
  observed costs; their individual causal contributions were not separately benchmarked.

**4 workers is the measured best of 4/6/8.** Use 2 if interactive responsiveness is more
important. There is no measured reason to increase to 6 or 8 on this currently loaded Mac.
At the measured 4-worker end-to-end rate, 100K paired views project to **54.2 collection hours**;
rare-biome search, retry mix and long-run load can change that figure. No extrapolated time
is presented as a completed 100K run.

## 100K storage projection from completed 1K

The completed `outputs/vox2pix_pilot_1k_rgb32` compression report measured 116,290,840 bytes,
including 90,638 bytes of fixed root metadata. At 100K:

- Logical compressed payload: **11,620,110,838 bytes = 11.62 GB / 10.82 GiB**,
  using `(1K payload − fixed metadata) × 100 + fixed metadata`.
- Current APFS small-file layout, projected from `du`: about **35.8 GB / 33.4 GiB** allocated.
  This is not unique APFS physical space or a measured archive size. Allow roughly 40–50 GiB
  for the dataset plus temporary-world/headroom needs.
- The payload includes lossless RGB PNGs, raw 48³ node/param2 arrays, the current **single**
  surface White Model + Water Count fields, and metadata. No new archive was created here.

## Exact bounded collection command

Each stage used the following command with `N=4`, then `6`, then `8`:

```bash
.venv/bin/python -B -u -m vox2pix_data collect \
  --mode production --dataset outputs/vox2pix_parallel_20261124/wN \
  --scenes 200 --max-scenes 30 --workers N --seed 42 \
  --renderer window --craftium-dir third_party/craftium_vox2pix_local32 \
  --worlds-per-task 8 --task-retries 2 --scene-timeout 180
```

`N` is a placeholder, not literal CLI syntax. These directories now contain committed data;
a future independent comparison requires fresh directories. The local one-off runner is
`runs/vox2pix_parallel_20261124/benchmark.py`. It samples resources every five seconds,
limits each stage to 600 seconds and would stop its collector on sustained critical pressure.
Neither protection triggered. Collector cleanup completed; no benchmark Native processes remain.

Machine-readable results: [measurements.json](measurements.json).
Raw resource series: [4](w4_resources.csv), [6](w6_resources.csv), [8](w8_resources.csv).
Local collector logs: `runs/vox2pix_parallel_20261124/w{4,6,8}.log`.
Native datasets remain in ignored `outputs/vox2pix_parallel_20261124/w{4,6,8}/`.
