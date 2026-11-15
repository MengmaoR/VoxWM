# Codex implementation task — VAE 100K / Center-Only 48³ / 1K Shards

**Repository:** MengmaoR/VoxWM  
**Working branch:** `codex/vae-100k-center-shards-20261115`  
**Authoritative specification:** `VAE_DATA_COLLECTION_PROTOCOL.md` (v2 on this branch)  
**Implementation status at task creation:** NOT IMPLEMENTED. No native examples have been run for this change.  
**Priority:** Complete production code and tests, but **do not launch production collection or GPU training**.

Read `AGENTS.md`, `VAE_DATA_COLLECTION_PROTOCOL.md`, existing `vae_data/`, and pinned native Craftium/VoxeLibre source before editing. Preserve existing user work, use `git pull --ff-only`, keep changes on the named branch, and commit normally. Do not modify unrelated model code, presentation/docs site, legacy logs or upstream engine sources.

## Goal and mandatory changes

1. **Spatial contract**: world-aligned centered 3x3x3 16³ chunks, 27 chunks / complete 48³ in each saved snapshot. Establish one canonical ENU-axis and origin mapping. All native captures and stored snapshots must be a contiguous time-consistent 48³, with center-chunk slice exactly [16:32]^3 in the canonical local representation. Verify boundary and orientation using unambiguous synthetic data tests; real native data remains the authority for accepted samples. Target-only changes count, neighbor-only changes do not. Do not silently substitute the 12-chunk prior layout.
2. **Quotas**: 100 complete `shard_0000..0099`, each exactly 500 static + 125 each D1/D2/D3/D4. Each dynamic chain comprises five accepted states -> 25 full chains per dynamic class per shard. Do not publish a partial shard. Static biome distribution 25 canonical groups × 20 per group with 8 surface / 4 elevated-complex / 4 shallow / 4 deep-cave. Biome-transition mark targets four per biome and shard, within the 20, not additional. Read current VoxGen manifest as needed for matching canonical biome taxonomy; do not mix in VoxGen M0–M6 mask ratios. Missing biome => quota shortage, not fake classification.
3. **Dynamic collector**: bounded previous/current native observation and <=10 unique target-valid candidate frames; no growing `_distinct`/`records` list, long-lasting raw trajectory/discovery cache, or attempts retained on disk. Dynamic remains centered, target-qualified changes only. Following intervention do not count immediate placement snapshot as VAE dynamic. 2 simulated seconds with no qualifying center change, 10 candidates, or explicit hard observation limit stops collection; log why. Strict `sim_time` (do not infer from nominal 24fps). Select uniformly from eligible contiguous windows of five **sparse candidate states** and persist true timestamps and step indices; <5 eligible => reject without auxiliary training samples.
4. **D4 solidification**: replace vegetation category. Implement D4-A (natural water + newly placed lava), D4-B (two placed sources), D4-C (natural lava + newly placed water). Source positions chosen randomly only after cheap static terrain-aware (downward + sideways) Water range 7/Lava range 3 reachability and predicted **center** contact; do not spend many native wall seconds on obviously impossible positions. Original native physics (lava cooling ABM 1 sim-second, liquid_update=1 sim-second, viscosity 7) must remain unchanged. Confirm only center conversion with valid contact geometry and correct product; avoid pre-existing stone false positives. Initial after-intervention anchor saved only for **successful** D4 chains, separated from VAE counted states and excluded from candidate pool. Pre/Reaction/Post may overlap across successive reactions; one snapshot per (rollout_id, simulation_step) with multi-role sets and reaction IDs. Require an ordered complete triplet within every selected five-state D4 window; otherwise reject. At most ten unique candidate frames plus one pending frame reference.
5. **Streaming storage**: one fully consistent `48³` (tokens uint16, aux light uint8) per selected VAE state, no extra duplicate center array, no full 72/240-frame VoxEvolve collection. Immutable `shards/shard_NNNN/{snapshots.zarr,anchors.zarr,metadata.parquet,manifest.json,READY}`; READY written atomically last only after content/counter/checksum/chain/seed split validation. Temporary staging and failed attempts removed. Single transactional writer/coordinator; separate native worker processes and ports for parallel CPU jobs. Train/val/test fixed by World Seed: static per shard 400/50/50 (per biome 16/2/2), dynamic 20/2-or-3/3-or-2 full chains per class so all four sum 80/10/10 chains; deterministic rotation for class and D4-A/B/C quotas. No world seed across splits, in any shard.
6. **Safe vocabulary**: cannot simultaneously expand VAE model embedding/decoder labels while streaming Shards. Use a fixed canonical Combined Token mapping/version validated before READY publication (or propose explicitly versioned migration with no silent relabel). Same vocabulary in every shard. Node Name + raw param2, `uint16`; light is auxiliary. If future unseen vocabulary cannot be encoded, fail explicitly; don't silently assign unknown token or shift existing IDs.
7. **User-facing commands**: `vae_data/README.md` should document prepare/discover, a **strict 2-examples-per-dynamic-class** bounded native acceptance command for coding-agent verification, and explicit **USER ONLY** commands to start shard generation and resume. CLI entry points must make accidental 1K shard/100K collection impossible during allowed tiny example validation. Ordinary examples may be generated only if essential to static/format validation; do not generate any large set in Codex.
8. **Speed**: preserve native clocks, random distributions, node timers, ABM and original liquid/fire implementations. The proposed accelerated virtual-clock engine is a separate future project. Do not modify upstream Craftium/VoxeLibre/C++ for this v2 task. Optimize only bounded candidate selection and unnecessary caching.

## Minimal native verification authorization

The user authorizes **at most TWO successfully accepted native examples PER DYNAMIC CATEGORY**, D1/D2/D3/D4, total **EIGHT accepted 5-state chains**, with actual complete 48³. This is an upper bound, not a requirement to loop without limit until green:
- Include finite `--max-attempts` and hard `--wall-seconds` guards; default must not be a full run.
- D4 two examples ideally cover two of A/B/C; the third subtype should have unit tests and may remain unverified natively within this cap.
- One physical frame with two D4 event roles counts once; verify consecutive R1/R2 and R1→flow→R2 overlap through unit tests (do not fabricate successful native events).
- Confirm native evidence: center-only changes, nonduplicate center states, actual new solidification product, no invalid entities/drops or unsupported falling, no off-center false positives.
- Record unsuccessful native attempts, actual time/steps and the honest shortfall if any. DO NOT fabricate D2 or D4 success.
- Unit tests and static checks are allowed; no unbounded benchmarks, no GPU training, no 1K shard generation, no VAE 100K, no 50K VoxGen, no long vox-evolve sequence generation or training.
- A handful of fast static tests and at most eight actual accepted dynamic chains is the **entire** Codex testing permission. All additional runs are executed manually by the user on the server.

## Verification and deliverables

Prefer targeted implementation in `vae_data/{config,core,engine,pipeline,sampling,storage,validation,cli,native/scene.lua,tests}/`, plus `vae_data/README.md` and the top-level `AGENTS.md`/README pointers. Do not unnecessarily rewrite the unrelated P0/P1 stack. Use simple APIs over a new orchestration platform.

Mandatory checks:
- Correct full 27-chunk 48³ shape and world-aligned center with unchanged physics.
- Neighbor-only changes do NOT advance candidate count or reset 2s cutoff; center target changes DO.
- Dynamic chain output always exactly 5 distinct, ordered, true-time states, from max 10 candidates.
- D4 initial anchor stored separately only when passed; complete Pre/R/Post, overlapping labels deduped by step; reactions outside target rejected.
- Every READY shard matches exact 500/125×4 and 25 biome group 20-per-group + 8/4/4/4; no partial READY.
- Seed-disjoint splits and stable frozen global combined vocabulary, atomic crash-resume, finite-memory streaming storage.
- Native examples count <=2 per category and their provenance; record nulls and NO-GO explicitly where failures occur.

At completion: update `PROJECT_STATE.md` and `vae_data/README.md` with actual code status, commits, test commands, number accepted per category, shortcomings and explicit user-run commands. **Never report an unrun test as passed.** Commit/push the implementation to the specified branch; do not merge into master without the user's explicit approval. Stop.

## Research boundary

The five-state dynamic chains are VAE snapshots, **not** a 5-frame temporal training input to the VAE, which learns independent 16³ states. Keeping `sim_time`, `step`, `anchor` and event labels serves future evaluation/reuse, without collecting current VoxEvolve GT. VoxGen can later use the spatial 48³ regions, but no VoxGen model or extra data collection is part of this implementation request.
