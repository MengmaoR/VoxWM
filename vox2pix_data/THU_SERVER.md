# THU: environment and one adaptive 1K collection

## Read-only observation — 2026-11-25

The SSH alias is `THU`. Inspection did not install software, change repository
content, or start Craftium/collection. The retired AML server was not contacted.

| Item | Observed |
|---|---|
| OS | Ubuntu 22.04, x86_64 |
| CPU | 2 × Xeon Platinum 8368, 76 physical / 152 logical CPUs |
| Process affinity | All 152 CPUs; no inherited finite CPU quota found |
| Existing load | Load averages about 7–11; one-second CPU utilization about 4.7% |
| Host RAM | About 1 TiB total, 712 GiB available at inspection |
| Shared account cgroup | 880 GiB limit, about 452 GiB current use; about 428 GiB headroom |
| Existing project | `/home/user/WORK/txj/VoxEvolve`, remote `MengmaoR/LTWM-CoDL` |
| Existing project changes | `experiments/controlled_ab/engine.py` modified; left untouched |
| Root/project filesystem | About 101.7 GiB available; already 88% used |
| Recommended data disk | `/data/disk2`, ext4, writable, about 2,154 GiB / 2.10 TiB available |
| Other writable disks | disk1 about 714 GiB; disk3 about 1,968 GiB; disk4 about 3,447 GiB (fuseblk) |
| Native prerequisites | CMake/C++/make/ninja present; Xvfb and multiple development libraries missing |
| Python | System 3.10; adjacent existing venv has no Craftium/NumPy/Zarr/PyAV |
| uv | `/home/user/WORK/txj/bin/uv` works, v0.13.5 |

These are shared resources, not a reservation of 152 idle CPUs. The account
memory limit matters in addition to host available RAM. Native Linux rendering
and throughput have **not** been measured here. Mac timing is not a THU estimate.

## 1. User-run installation in a separate VoxWM checkout

Leave `VoxEvolve` and its uncommitted work alone. Create a sibling checkout of
the actual active repository. The following commands are for the user to run;
none were executed remotely by Codex.

```bash
ssh THU
export PATH="/home/user/WORK/txj/bin:$PATH"
cd /home/user/WORK/txj
git-txj clone git@github.com:MengmaoR/VoxWM.git VoxWM
cd /home/user/WORK/txj/VoxWM
```

`git-txj` is the existing personal Git wrapper in that directory. Its GitHub
authentication was not exercised during read-only inspection. If this new
checkout already exists, use `git-txj pull --ff-only origin master` inside it
instead of cloning again; do not reset the old `VoxEvolve` checkout.

Install system prerequisites manually (the account belongs to the sudo group;
passwordless sudo was not tested):

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config ninja-build \
  libsdl2-dev libgl1-mesa-dev libgl1-mesa-dri mesa-utils \
  libx11-dev libxi-dev libxxf86vm-dev libjpeg-dev libpng-dev \
  libfreetype6-dev libsqlite3-dev libzstd-dev libluajit-5.1-dev \
  libogg-dev libvorbis-dev libopenal-dev libcurl4-openssl-dev \
  libjsoncpp-dev gettext libgmp-dev zlib1g-dev xvfb xauth xkb-data tmux

BUILD_THREADS=16 bash scripts/setup_gt.sh \
  --craftium-dir third_party/craftium_vox2pix_fog48
uv pip install --python .venv/bin/python -r requirements-vox2pix.txt
BUILD_THREADS=16 bash scripts/build_vox2pix_fog48.sh \
  third_party/craftium_vox2pix_fog48
```

This creates the repository's independent Python 3.11 CPU environment and fixed
Craftium checkout, then builds the existing final32/fog/material patches.
The historical `fog48` directory name does not set the active RGB range: the
production profile still uses RGB32 / Fog20–30 / 48³ observations. No Torch,
CUDA model weights, or PERSIST inference is needed. The old environment is not
used. Do not import the retired AML environment JSON or its `/user-data/...` paths.

## 2. Collect 1K once, with adaptive concurrency

The 1K is **200 scenes × 5 views**, 120 Surface / 60 Cave / 20 Underwater.
It uses `production`, the same final protocol as the large collection. There is
no separate pilot workload and no repeated dataset per worker setting.
Arrays are losslessly compressed while written; RGB is PNG.

```bash
cd /home/user/WORK/txj/VoxWM
mkdir -p /data/disk2/txj/vox2pix/tmp runs/vox2pix_thu_1k

tmux new-session -d -s vox2pix1k -c "$PWD" \
  'unset VOXWM_ENV_CONFIG; export TMPDIR=/data/disk2/txj/vox2pix/tmp LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null; exec bash scripts/collect_vox2pix.sh --dataset /data/disk2/txj/vox2pix/vox2pix_1k_rgb32 --scenes 200 --workers auto --adaptive-workers --adaptive-start 4 --adaptive-window 60 >runs/vox2pix_thu_1k/collect.log 2>&1'
```

The native code uses a private Xvfb per world and Mesa llvmpipe with one rendering
thread per process. Temporary worlds also use the data disk. tmux keeps the
collector alive across SSH disconnections. This command starts **only 1K**, not
100K. If a tmux session of this name exists, inspect it rather than launch another
collector. The same collection command resumes missing slots after interruption.

### What auto means here

- `--workers auto` alone still means a **fixed** CPU-affinity-sized worker limit.
- Adding `--adaptive-workers` makes that number a ceiling. The active limit starts
  at four and climbs 4 → 8 → 16 → 32 → … while enough actual scenes remain.
- After a 15-second settling period, full-occupancy committed scenes per wall-clock
  interval are measured (60 seconds here). Failed attempts do not count as output.
- Two consecutive windows at least 10% slower than the best observed window cause
  a rollback to that window's concurrency, held for the rest of this invocation.
- One process pool is retained. On rollback, excess in-flight scenes finish normally;
  no world is killed merely to resize, no accepted data is discarded, and neither
  seeds nor scene quotas change. Idle Python worker processes may remain in the pool.
- Tail drain or insufficient occupancy is excluded from comparisons. If fewer than
  two waves at the next level remain, exploration stops. A fast 1K may finish before
  finding a slowdown: `budget_exhausted` is honest incomplete tuning, not proof of an
  optimal limit. A later large run can ramp again under its then-current host load.

`collection_progress.json` records current/ceiling concurrency, best observed rate,
each measurement/decision, and the tuning outcome. This is a simple throughput
heuristic; scene difficulty and other users' load can affect its choice.

## 3. Monitor and stop independently

```bash
cd /home/user/WORK/txj/VoxWM
.venv/bin/python -m vox2pix_data dashboard \
  --dataset /data/disk2/txj/vox2pix/vox2pix_1k_rgb32
```

The terminal dashboard includes the current adaptive target, CPU ceiling, best
observed rate and decision state, as well as scene/category/biome progress and ETA.
It does not own collection processes. Close it with Ctrl-C without stopping capture.

```bash
# Native errors and adaptive decisions:
tail -f runs/vox2pix_thu_1k/collect.log

# Stop only the dedicated collector, allowing owned-child cleanup:
tmux send-keys -t vox2pix1k:0.0 C-c
```

## 4. After completion

```bash
cd /home/user/WORK/txj/VoxWM
.venv/bin/python -m vox2pix_data dashboard \
  --dataset /data/disk2/txj/vox2pix/vox2pix_1k_rgb32 --once
.venv/bin/python -m vox2pix_data compression-report \
  --dataset /data/disk2/txj/vox2pix/vox2pix_1k_rgb32
du -sh /data/disk2/txj/vox2pix/vox2pix_1k_rgb32
```

Expected completion is `COMPLETE`, 200 scenes / 1,000 frames. `INCOMPLETE` or
`ERROR` is not success; preserve the log and committed data and report the actual
error. The lightweight compression report uses existing compressed files and
metadata; it does not rerender, recompress, or scan every array. Inspect the real
THU output and speed before authorizing the full collection. No 100K launch is
part of this task.

## Local implementation verification

21 targeted synthetic/unit tests passed in 0.312 seconds, plus CLI help and diff
whitespace checks. They cover ramp/rollback, no tail-induced false rollback,
runtime-only flags, preserving every fixture scene across resizing, and read-only
dashboard display. No native collection, build or installation was run on THU
or the Mac for this implementation. Linux Native rendering remains user-verified.
