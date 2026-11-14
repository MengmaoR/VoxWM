NO-GO — VoxWM VAE P0 real native collection

Authority: 2026-11-13 data protocol supersedes the 2026-11-12 manifest for collection. No VAE training or 200K generation.

World seed: 191664964; status: completed; wall time: 262.43s.
Native fixed-coordinate comparisons: 327,680 voxels; exact Combined-Token round trips: 72,712,192 voxels; observed vocabulary K=128.

| Category | Saved snapshots | Valid clips / attempts | Native observation s | Snapshots / sim hour |
|---|---:|---:|---:|---:|
| Ordinary | 10 | — | — | — |
| D1 Water Flow | 8 | 1/1 | 6.57 | 4380.7 |
| D2 Lava Flow | 0 | 0/2 | 32.02 | 0.0 |
| D3 Fire Propagation | 8 | 1/1 | 12.54 | 2297.1 |
| D4 Vegetation Growth | 0 | 0/2 | 157.64 | 0.0 |

Only accepted distinct states count as training snapshots. Invalid diagnostic sequences are stored separately, without padding.
One seed cannot validate train/validation/test balance or 25-biome coverage. Whole-seed split assignments are preserved.

Ordinary biome groups: {'Plains': 4, 'JungleEdge': 4, 'BirchForest': 2}.
Ordinary actual terrain strata: {'surface': 4, 'air': 2, 'shallow_underground': 2, 'deep': 2}; transition snapshots: 4.
Observed saved-clip change bins: {'large(>=256)': 1, 'medium(16..255)': 1}. These are descriptive bins, not new sampling quotas.
Vine/bamboo losses alone do not count as D4 growth. Fire requires native spread beyond the single intervention, not just ignition or fire-age changes.

## Dynamic outcomes and single bounded repairs
- D1 attempt 0: valid_clip; best fixed Chunk has 8 distinct states, cumulative target changes 633, changed positions 621; ignition=natural.
- D2 attempt 0: insufficient_distinct_states; best fixed Chunk has 1 distinct states, cumulative target changes 0, changed positions 0; ignition=natural.
- D3 attempt 0: valid_clip; best fixed Chunk has 8 distinct states, cumulative target changes 22, changed positions 10; ignition=controlled_random.
- D4 attempt 0: insufficient_native_growth_states; best fixed Chunk has 4 distinct states, cumulative target changes 4, changed positions 4; ignition=natural.
- D2 attempt 1: no_natural_candidate; best fixed Chunk has 0 distinct states, cumulative target changes 0, changed positions 0; ignition=none.
  Bounded repair: Inspect deeper native cave layers (64/128 below surface) and require a flowable edge.
- D4 attempt 1: insufficient_native_growth_states; best fixed Chunk has 8 distinct states, cumulative target changes 11, changed positions 11; ignition=natural.
  Bounded repair: Reject mature/blocked bamboo; prefer supported vines with free growth space; longer unchanged native ABMs.

## 200K feasibility

One known seed and previously discovered candidate hints; linear extrapolations are not a collection forecast. Zero yield => no estimate.
Each category retains its original quota. Missing production cannot be filled with other categories, duplicates, manual fluid sources or growth commands.
Full-dataset world count and cost remain unmeasured unless every required category has demonstrated sustainable output.

| Category | Target | Observed | Conditional world-equivalents | Conditional simulation hours |
|---|---:|---:|---:|---:|
| Ordinary | 100000 | 10 | 10000 | not estimable |
| D1 | 25000 | 8 | 3125 | 5.71 |
| D2 | 25000 | 0 | not estimable | not estimable |
| D3 | 25000 | 8 | 3125 | 10.88 |
| D4 | 25000 | 0 | not estimable | not estimable |

The retained D2 image is a static neighboring Chunk with zero lava voxels; the actual discovered lava candidate did not change. It is not a lava-flow visualization. D4 images show native vine losses, not growth.

## Actual artifacts

- `snapshots.zarr/tokens`: uint16[N,16,16,16]; `light`: raw param1 uint8[N,16,16,16], auxiliary only.
- `vocabulary.json`: append-only Node Name / raw param2 pairs; `metadata.parquet`: seed, fixed ENU origin, biome, stratum, category, event, true time/step and split.
- `statistics.json`: complete observations, rejected states, yields and explicitly conditional linear estimates.
- Diagnostic PNG/GIF/NPZ files are actual observed states, not generated illustrations or extra training snapshots.
- [Ordinary diagnostic](vae_p0/reports/2026-11-14/visualizations/Ordinary.png)
- [D1 diagnostic](vae_p0/reports/2026-11-14/visualizations/D1.png)
- [D2 diagnostic](vae_p0/reports/2026-11-14/visualizations/D2.png)
- [D3 diagnostic](vae_p0/reports/2026-11-14/visualizations/D3.png)
- [D4 diagnostic](vae_p0/reports/2026-11-14/visualizations/D4.png)

Voxel extraction uses native VoxelManip node/light/param2 arrays in the same Lua callback, cross-checked against get_node for all 4096 positions. The existing Craftium Python C++ channel supplies node/param2 only; no synthetic light or camera-derived Chunk origin is used.
No engine fluids, ABM intervals, growth metadata or native physics were modified. Fire permits one explicitly recorded ignition per event, with a pre-ignition state.
Earlier random-world candidate locations are reused as discovery hints; these are not an unbiased event-discovery survey. No global state deduplication across independent worlds/contexts.

## Decision

P0 does not yet establish all four native dynamic collection paths. Stop here; no large collection is authorized.

Initial startup attempt: First startup stopped before collection because Lua loaded a plugin outside the Mod directory (64.53s, zero snapshots); fixed by copying it into the scene Mod. Failure output is preserved in outputs/vae_p0_20261114.

Offline qualification: The provisional 8-state D4 sequence contained only native vine losses (0 new vegetation positions), not growth. Its 8 states are diagnostic only, excluded from this final dataset. A neighboring Chunk in D4 attempt 0 had one native vine addition but only 2 distinct states, below the 5-state minimum. No further native simulation was run after the one bounded D4 repair.
