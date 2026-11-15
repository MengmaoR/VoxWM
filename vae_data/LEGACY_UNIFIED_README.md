# Unified VAE Data Collection

Current implementation: `vae_data/`. Historical `vae_p0/`, `vae_p1/`, `vae_pilot/` are preserved; the old raw Node ID generator is not the 200K entry. Authority: [VAE_DATA_COLLECTION_PROTOCOL.md](../VAE_DATA_COLLECTION_PROTOCOL.md) / [current Drive Markdown](https://drive.google.com/file/d/1NFM1Cb3os2yesAQgBk8VzM6vMmcj1eg8/view), mirroring the latest Unified Collection request. The old P0 Drive protocol remains historical. VAE architecture is unchanged.

Run from the VoxWM repository root with the existing Python3.11 CPU environment (`scripts/setup_gt.sh`, then `uv pip install --python .venv/bin/python -r vae_p0/requirements.txt`). No Torch/CUDA/weights are needed. macOS uses its native graphics stack; Linux may add `--renderer software`. Original simulator source is not modified: the runtime builtin overlay is temporary.

## Small checks

```bash
.venv/bin/python -m unittest vae_data.tests.test_unified vae_p1.test_p1
luajit vae_data/tests/entity_boundary.lua
# REAL native integration; explicit hard total wall limit, never 200K
.venv/bin/python -m vae_data validate --dataset outputs/unified_validation --wall-seconds 900
```

Native validation uses previously measured natural locations as transparent discovery hints, not handcrafted scenes. It records all failures and never fills missing native outcomes with fixtures. Native findings and limits are in the repository's `vae_data/NATIVE_VALIDATION_RESULTS.md` after the task run.

## Discovery → Replay → Collection

```bash
# Low-cost scan: multiple sites per world, no long per-candidate wait
.venv/bin/python -m vae_data discover --dataset outputs/vae_dataset \
  --world-count 100 --sites-per-world 16 --wall-seconds 7200
.venv/bin/python -m vae_data status --dataset outputs/vae_dataset
```

Inspect scene IDs and their full JSON metadata:

```python
import pyarrow.parquet as pq
rows=pq.read_table('outputs/vae_dataset/scene_index.parquet').to_pylist()
print([(r['scene_id'], r['origin']) for r in rows[:10]])
```

```bash
# Replace SCENE_ID with an indexed scene. Three distinct fresh instances.
.venv/bin/python -m vae_data replay --dataset outputs/vae_dataset \
  --scene-id SCENE_ID --repeat 3 --seconds 4 --wall-seconds 240
# Natural water only; no intervention
.venv/bin/python -m vae_data collect --dataset outputs/vae_dataset \
  --category D1 --scene-id SCENE_ID --intervention never --seconds 60 --wall-seconds 180
# Controlled multi-source suspended water (valid placement type)
.venv/bin/python -m vae_data collect --dataset outputs/vae_dataset \
  --category D1 --intervention always --source-mode suspended --seconds 60 --wall-seconds 180
# Lava near indexed water, or ordinary source sites; event attribution remains separate
.venv/bin/python -m vae_data collect --dataset outputs/vae_dataset \
  --category D2 --intervention auto --seconds 180 --wall-seconds 420
# Variable-size legal fire set, independent p=.1 and bounded fallback
.venv/bin/python -m vae_data collect --dataset outputs/vae_dataset \
  --category D3 --intervention always --seconds 120 --wall-seconds 300
# Natural community trimming; no planting / cap filtering; full observation budget
.venv/bin/python -m vae_data collect --dataset outputs/vae_dataset \
  --category D4 --subtype bamboo --intervention always --seconds 1800 --wall-seconds 2100
# Use --subtype reeds or vine for the other natural plant groups.
# A completed run can be observed again with a longer budget in a NEW independent world:
.venv/bin/python -m vae_data collect --dataset outputs/vae_dataset \
  --category D4 --subtype bamboo --seconds 1800 --rollout-tag extended1800 --wall-seconds 2100
# This restarts the native world; it does not resume its old physical time.
# Ordinary exports cached native discovery chunks rather than starting a world per snapshot.
.venv/bin/python -m vae_data collect --dataset outputs/vae_dataset \
  --category Ordinary --max-scenes 10 --wall-seconds 120
```

For water/lava/fire, a measured 15-second unchanged interval can stop the current observation. This is a finite observed plateau, not proof that no later change is possible. Plants do not stop at 8 states: `--seconds` is the total native observation budget; choose600/1200/1800 or a longer explicit budget. Original ABM/chance/time mechanisms remain unchanged. Every command has a hard external wall-time watchdog, including blocking engine startup/socket paths.

## Formal 200K — user starts explicitly, not run by Codex

A sufficiently diverse indexed corpus and acceptable native validation/throughput are needed first. The command does not fabricate shortfalls or auto-download/create new worlds endlessly to meet quotas:

```bash
.venv/bin/python -m vae_data collect --dataset outputs/vae_dataset --resume \
  --full-200k --intervention auto --wall-seconds 86400
# Continue the same dataset in another explicitly budgeted run:
.venv/bin/python -m vae_data collect --dataset outputs/vae_dataset --resume \
  --full-200k --intervention auto --wall-seconds 86400
```

Remaining quotas are saved in `quota_shortfall.json`; add more diverse discovery seeds/sites if the index is exhausted. Ordinary enforces4,000 per major biome and at most5% of its quota from one seed. Missing biomes cannot be replaced with easy ones. D4 scheduling includes natural bamboo/reeds/vine; auxiliary events do not fill standard quotas.

## Data / traceability / resume

`scene_index.parquet` and SQLite describe seed, actual source commits, mapgen/config, chunk/region coordinates, biome/geometry/counts/signals, discovery phase/hash and replay reference. Discovery signals are not events. `replay_consistency.json` separates equal initial state from equal timed trajectories; if voxel hashes differ, the pipeline reports that variation.

`snapshots.zarr/{tokens,light}` are uint16/uint8[N,16,16,16], ENU. `vocabulary.json` is append-only `(Node Name,param2)`, light is auxiliary. `metadata.parquet` includes source seed/chunk, phase/hash, actual time/step/split, scene/rollout and attribution. Nested provenance columns use JSON. `collection.sqlite` is authoritative for committed samples/jobs; Parquet is an exported checkpoint. Only run one writer per dataset. Resume retains committed arrays and IDs, removes uncommitted Zarr tails, skips completed region/category jobs, and starts incomplete scenes as **new** rollouts. It does not restore a simulation to its interrupted time.

`trajectories/<rollout>/raw_trajectory.bin/jsonl` preserves actual callback/early/initial/intervention/settling/native captures with sparse token/light updates. Reuse `vae_p1.storage.Reader` to reconstruct captured frames. Raw logs include states beyond the5–8 training selection. `aux_short_event/` is separate. `scenes.jsonl`, `interventions.jsonl` and job rows retain initialization choices and audits. `collection_statistics.json`/`statistics.json` report actual measurements, biome/seed counts and same-source semantic overlap. Parquet exports are batched every five minutes and at command exit; the SQLite commit journal stays current.

Visualizations are real voxel top/side projections, GIFs and added/removed/state-change maps. They are not RGB or a learned model. Native validation publishes only light reports/plots, with raw arrays left in ignored `outputs/`. Inspect actual pre/post/settling/final phases without starting the engine:

```bash
.venv/bin/python -m vae_data.visualize --dataset outputs/vae_dataset \
  --rollout ROLLOUT_ID --category D4 --subtype bamboo
```

## Voxel-only mechanism limits

Entity/drop factories are blocked before game mods load, covering spawners, script spawns and direct plant-drop paths. The spawner cage remains a voxel; entity-only setup/display/timer is disabled. The collector player is invisible, stationary and non-interacting; target MapBlocks are forceloaded, no camera crop determines chunk coordinates.

Falling-node transitions are unsupported because upstream uses a temporary entity. Any attempted falling conversion quarantines that whole rollout before training commit; it is not a valid frozen-world sample or a replacement physics implementation. All accepted native captures must have zero active nontechnical entities, zero dropped items, zero entity-induced node mutations and no unsupported transition.

Native RNG and mapgen decoration order can affect full voxel initial hashes even with a fixed seed. Report matching chunks and variants, do not correct them by hand. Fast contacts between captures can be unconfirmed. Native outputs, rates and costs require actual runs; synthetic unit cases provide no physical evidence. Current measured results decide readiness; the presence of a CLI is not proof that200K can already be collected efficiently.
