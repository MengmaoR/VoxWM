# VAE 200K Pilot Dataset — native 16³ chunks

Only the native engine produces voxel states. This collector does not generate VAE batches, place Water/Lava/vegetation, accelerate ABMs, or fabricate missing events. Test A and the existing GT CLI keep their defaults.

## Quotas and biome coverage

`config.json`: Ordinary 100000; Water, Lava, Fire, Vegetation each 25000; total 200000. A snapshot is one world-aligned 16×16×16 ENU chunk. The quota is never lowered when native yield is poor.

Ordinary is capped at 4000 snapshots for each of 25 explicit major-biome groups. The names/aliases come from the fixed VoxeLibre `mcl_biomes` definitions; variants such as beach/ocean/snowtop retain their raw names and are grouped by parent climate. Both raw chunk biome and surface biome are saved. Underground chunks use the surface biome for the 25-group balance, while retaining their own actual biome and height. Unmapped/underrepresented groups are reported, never silently reassigned. The full alias list is in config.json.

Random seeded v7 worlds are surveyed at randomized x/z sites and multiple native heights, including deep underground for Lava. An anchored player is only an observation probe; changing probe locations does not move terrain or generate fluid. Native liquids, timers, ABMs and other dynamics continue normally. All complete observed chunks are eligible for Ordinary, including non-target dynamics.

Water/Lava events require real liquid additions, flowing-node changes or liquid param2 changes. Vegetation prioritizes bamboo/vines, then natural saplings/plants/flora; crops have no separate quota. Plant removal alone is not growth. Fire first searches for existing natural flames. Only when none are found and the Fire quota remains may one random combustible location be ignited using the native `mcl_fire.set_fire` API; ignition and subsequent propagation are labelled separately. Fire aging without new burning cells does not qualify as propagation. Other dynamics during an ignited-fire observation are retained as Ordinary, not mislabelled natural vegetation growth.

## Install and run a bounded pilot

Run from the repository root. Native dependencies and the CPU installation are documented in the root README; no Torch/CUDA/weights are required.

```bash
bash scripts/setup_gt.sh
# Tiny logic checks: synthetic data only, no engine
.venv/bin/python -m vae_pilot.check_sampling
# Bounded real pilot. No automatic escalation to full collection.
.venv/bin/python -m vae_pilot --output outputs/vae_pilot \
  --max-worlds 1 --sites-per-world 4 --site-seconds 24 \
  --sample-seconds 0.25 --fire-ignite-after-seconds 3 \
  --wall-seconds 210 --snapshot-cap 256
# Regenerate statistics from existing data; CPU only
.venv/bin/python -m vae_pilot.report --root outputs/vae_pilot
```

A short native observation may yield zero standard dynamic clips. That is a measured shortage, not permission to repeat frames or manufacture events. Longer observation is especially relevant for the native bamboo growth ABM (fixed engine interval/chance); these engine rules are not changed. A separately manual, larger pilot can use at most 4 worlds, more sites and longer observation; it still obeys the snapshot cap and native physics.

## Dynamic clip sampling

- Events are keyed by seed/site/fixed world chunk/type; long trajectories may have 64-observation segments with a common trajectory id. A repeated local-array index after a crop shift is never interpreted as the same world coordinate.
- Drop consecutive unchanged node+param2 states. Select only globally distinct states, never padding or counting duplicates across categories/splits.
- Accumulate actual changed-voxel updates over time, including revisits such as A→B→A. Select up to 8 distinct states near equally spaced cumulative-change milestones, in original chronological order.
- Keep 5–7 if fewer than 8 are available; fewer than 5 are recorded as invalid candidates and contribute zero standard Dynamic snapshots.
- `change_bins=[16,256]` means small <16, medium 16–255, large ≥256 cumulative voxel updates. These are explicit descriptive bins, not extra quotas. Reports include valid-only and all-candidate distributions.
- 8 is clip sampling length, **not the VAE optimizer batch size**. No optimizer is implemented here.

## Data format

`dataset.sqlite` uses only SQLite/zlib/NumPy:

- `palette`: dataset-wide canonical node id → native node name/groups. It does not assume that native runtime content ids are identical between seeds.
- `snapshots`: category, split, seed, event id, integer chunk world origin/index, raw biome, biome group, terrain/height, raw native simulated time, game time, engine microseconds, Unix time, and sparse block metadata/timers.
- Each snapshot payload losslessly encodes uint16 node ids [16,16,16], then uint8 param2 [16,16,16]. Param2 retains native liquid level/orientation/growth state. Metadata is kept for native metadata-bearing nodes via find_nodes_with_meta; metadata-only clock changes do not inflate the voxel-state count.
- `clips`: original/selected timestamps, selected cumulative-change positions, distinct-state counts, cumulative updates, native-propagation/growth flag and natural/ignited source. No repeated frame is inserted.
- `sites`, `worlds`: observed opportunities, seeds, heights/biomes, native observation time and measured wall time. The initial few frames after probe relocation are only used when metadata and C++ observation cover the same world chunk; newly unmatched crops are not guessed.

The byte content of node ids+param2 has a UNIQUE constraint for exact duplicate rejection; no file hashes or audit layer are used. One World Seed maps wholly to Train/Validation/Test (80/10/10 default arithmetic seed partition), including every trajectory and segment. Content duplication across splits is also rejected. Outputs are ignored by Git.

```python
from vae_pilot.storage import iter_snapshots
for sample in iter_snapshots('outputs/vae_pilot', split='train'):
    voxel_ids = sample['voxel_ids']  # uint16, 16³
    param2 = sample['param2']       # uint8, 16³
    # Use this dataset's palette and sample['metadata']; no legacy model is required.
```

`clips.jsonl` lists each candidate/standard clip with its id, time/progress samples and change counts.

`report.json`/`report.md` contain actual yields, natural discovery rate (changed chunk trajectories / all observed chunk visits), distinct-state and change-size distributions, biome/height/split counts and per-category 200K projections. Ignition-seeded Fire output is separate from natural discovery. Zero observed yield gives a null/UNKNOWN estimate, not a finite invented cost. Engine stepping time is measured after initialization and includes site search; wall time also includes process startup. Rare missing biomes likewise make the combined estimate UNKNOWN.

## Full 200K entry — only after checking the pilot

The native collector must first have produced a successful nonempty pilot report. Native dynamics availability and all reported shortages still need to be assessed by the user. The following is a **manual entry**, not a promise that these budgets suffice. It stops at the original quotas or at the explicit budget and reports the exact deficit.

```bash
# Set your own upper bounds; these are limits, not measured cost estimates.
MAX_WORLDS=10000
MAX_SECONDS=86400
.venv/bin/python -m vae_pilot --full \
  --pilot-report outputs/vae_pilot/report.json \
  --output outputs/vae_200k --max-worlds "$MAX_WORLDS" \
  --wall-seconds "$MAX_SECONDS"
```

To continue a bounded run, repeat the same dataset configuration and add `--resume`; completed/attempted seeds are not counted again. Sampling configuration cannot be silently changed in an existing dataset. Resume does not restore a half-observed live engine trajectory: retained clips remain intact, and subsequent collection uses new seeds. No GPU, VAE training or 200K generation is started by this code delivery.


## Measured bounded pilot (2026-11-13)

The delivered CPU environment was compiled and actually run on macOS. The final bounded command above used one random world (seed191664964) and four locations. It measured 96.357 s observation time, 99.379 s post-initialization engine stepping and 133.830 s world wall time. Results: Ordinary89; Water48 from six 8-state clips; Lava/Fire/Vegetation zero valid snapshots. Nine natural water-change trajectories were discovered out of 108 chunk visits (8.33%); three were too short/duplicate-limited. One manually ignited Fire candidate had only two distinct states and was excluded. No missing category was fabricated. Ordinary covered FlowerForest6/Forest27/JungleEdge25/MegaTaiga19/Plains12; the other 20 biome groups remain unfilled.

The final report and per-clip counts are published in `reports/2026-11-13/`. Failed/earlier local attempts are retained under outputs/ and runs/, not mixed into these counts. Only the report is tracked; native snapshot blobs stay local. This pilot verifies the native read/store/progress-sampling path, **not** the feasibility of filling three zero-yield categories. The all-quota 200K world/time estimate is UNKNOWN. Per-category nonzero-yield extrapolations are rough one-world measurements and do not satisfy the 25-biome balance by themselves.
