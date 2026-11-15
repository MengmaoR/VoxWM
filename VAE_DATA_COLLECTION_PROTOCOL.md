# VoxWM — Unified VAE Dataset Collection Protocol v2

**Status:** approved research specification, **implementation pending Codex**. Date: 2026-11-15. Scope: VAE acquisition only. This document supersedes old 200K / vegetation / 12-chunk / 5–8-state / permanent-trajectory-cache rules. Existing P0/P1 logs and earlier manifests remain historical. Do not claim the current Python implementation already satisfies v2.

## 1. Scope, units and quota

The output is a **100,000-sample VAE dataset**, represented as **100 independently ready shards of exactly 1,000 VAE training samples**. Every stored snapshot is one **contiguous, world-aligned 48×48×48 Voxel Region = 3×3×3 chunks of 16³**. Its center 16³ is the **only VAE input/target and the only spatial region used to qualify dynamic changes**. The 26 neighbor chunks are faithfully captured spatial context, *not extra samples*, and their changes alone cannot trigger a candidate frame or reset inactivity.

Every shard must contain exactly:
- Ordinary / Static: 500 independent single-snapshot regions.
- D1 Water Flow: 125 VAE states, grouped as 25 complete 5-state chains.
- D2 Lava Flow: 125 states, grouped as 25 complete 5-state chains.
- D3 Fire Propagation: 125 states, grouped as 25 complete 5-state chains.
- D4 Water–Lava Solidification: 125 states, grouped as 25 complete 5-state chains.

Thus: 500 static regions + 100 five-state dynamic chains = 1,000 training samples/shard; 100 shards total. D4 initial anchors are extra metadata/state artifacts, **not counted as VAE samples**. Preserve 48³ for each of the five selected states, even though VAE extracts only the center 16³.

**Do not record a separate VoxEvolve continuous sequence** (no 72 or 240 frame GT, no independent 10-second VoxEvolve replay, no VoxEvolve 50K supplement). Five-state dynamic chains and the D4 anchor can be reused in later VoxEvolve work, retaining their true sparse timestamps.

## 2. Ordinary 500 per shard: matched spatial recipe

Reuse the preexisting VoxGen/Ordinary **25 canonical Overworld biome groups**, and lock one explicit mapping for all 100 shards (the older pilot list and VoxGen semantics must be reconciled, particularly Edge/M variants). Exactly **20 static snapshots per biome group per shard**, distributed within each biome as:
- Surface: 8
- Elevated/Complex terrain: 4
- Shallow Underground: 4
- Deep/Cave: 4
- Biome Transition: target 4 of the 20, an overlapping label rather than a fifth stratum. Select actual spatial boundaries; never invent biome changes. If such a strict target is impossible for a rare group, leave shard incomplete and report shortage; do not silently replace with easy regions.

Preserve scene diversity across world seeds, elevations, topography and nearby generated structures. A static region is a single frame and may contain a static-looking water/lava/fire voxel; no temporal immobility claim is required. Capture once, validate and commit/discard; do not retain discovery caches, unused trajectories or all observed candidate positions.

## 3. Dynamics: one common bounded collection contract

D1–D4 use true native Craftium/VoxeLibre physics, no manual step-by-step synthetic frame updates. Native intervention is permitted only to seed a legal scene. The immediate after-intervention state **is not native evolution and does not count as a VAE dynamic candidate**.

1. Discovery and validity screening: bounded **1 simulated second** per check; same-seed reconstruction checks also bounded to one second. Keep seed, world coordinates, build/config, legal intervention descriptor and only essential metadata; delete check data and temporary worlds.
2. Use a **static geometry-based flow-reachability prefilter**, not repeated long native simulations to guess suitable positions. Respect the native Water liquid_range=7 and Lava liquid_range=3, downward/sideways propagation and actual terrain barriers. Also use center-interaction potential for legal Fire candidates. This is a cheap candidate filter, never a substitute for native event confirmation.
3. Place/ignite, and then scan the complete 48³ snapshot in real native simulation time. Compare **only center 16³ target-related Combined Tokens** with their previous state. Neighbor-only dynamics must never qualify a state or reset the timeout.
4. Keep **at most 10 distinct candidate center states**, storing the accompanying entire 48³ only for these bounded candidates. A scene stops upon 10 candidates; or after **2 simulated seconds with no qualifying CENTER change**, measured in native sim_time (nominally 48 frames at 24Hz if a strict fixed clock exists). There is a separate configurable hard maximum simulated observation duration and a finite wall-clock watchdog; never wait indefinitely.
5. For D1–D3, obtain 5–10 unique ordered valid states, then uniformly random-select **one consecutive five-state window in the ordered candidate list**; adjacency in this *sparse state list* does not mean consecutive simulation frames. Less than five valid candidates -> reject. No auxiliary 2–4-state training pool and no repeat padding.
6. For D4, apply the event rules in §4 and random-select only five-state windows containing a valid Pre–Reaction–Post event triplet.
7. Validate complete 48³ consistency, center eligibility, native-event attribution, voxel-only boundaries and uniqueness, **atomically commit five selected 48³ frames** plus metadata; otherwise discard the whole attempt. Immediately remove the other candidates, all raw tracking data and temporary worlds. A bounded previous/current frame and up to ten accepted candidates are allowed in RAM; unlimited `records`/`_distinct`, permanent `raw_trajectory` and `discovery_cache` are not.

Timing note: pinned Craftium default `liquid_update=1.0` simulated second and VoxeLibre Lava viscosity 7. Viscosity **is not a period of seven frames**. Lava cooling uses an ABM `interval=1` second. The 2-second inactivity rule is an efficiency cutoff, not proof of physical equilibrium. With currently wall-clock-derived server dt, measure simulated seconds, do not assume each observed render frame equals exactly 1/24 simulation second until a fixed timestep is implemented and checked. **Accelerated virtual-time simulation is out of scope for this change**; do not modify engine clocks or physics parameters.

## 4. D4 Water–Lava: three scene subtypes and event-aware sampling

D4-A: existing natural STATIC WATER pool, place LAVA source(s) above or to its side (dynamic lava contacts static water).
D4-B: terrain without a preexisting interacting pool, place WATER and LAVA source(s) separately at reachable positions (dynamic lava contacts dynamic water).
D4-C: existing natural STATIC LAVA pool, place WATER source(s) above or to its side (static lava contacts dynamic water).

Source positions must pass a center-biased flow-reachability prefilter before randomized choice. Permit valid near-edge neighbor sources if the real reaction is predicted within the center. Avoid paths leaving the observed 48³ volume prematurely. A simple horizontal distance-to-water heuristic alone is insufficient.

Official D4 scene requirements:
- **Initial Anchor:** after source placement, before native advancement. Store one complete 48³ anchor only if the D4 attempt passes. This anchor is not among the VAE five states or ten candidates.
- **Pre(R):** latest eligible distinct center state strictly preceding native reaction R; MUST NOT reuse the intervention anchor as a VAE candidate.
- **Reaction(R):** verified **center** conversion to obsidian/cobblestone/stone with correct previous water/lava contact geometry; pre-existing stone is not evidence. Different products at the same simulation step form one physical frame.
- **Post(R):** first later qualifying center change, whether flow or a second reaction. If reaction R2 happens immediately after R1, the R2 frame is Post(R1), and R1 frame may be Pre(R2). If a flow frame is between reactions, it may be Post(R1) and Pre(R2).

Use **one ordered frame registry keyed by (rollout_id, simulation_step)**, with a set of multiple roles and reaction_ids per physical frame. When a role overlaps a recorded frame, update labels, not snapshot count. Preserve true sim_time and step; do not merge different steps by hash, but reject padding with an identical center state. Keep only the bounded number of candidate snapshots (at most 10) plus at most one pending pre/post reference. Same-event triplets require strictly increasing physical steps; different events may share a step/role. Every 5-state output window must contain at least one confirmed full Pre–Reaction–Post triplet and five unique eligible center states. If no valid window exists, reject the scene.

D4 per-shard subtype quota: **8 / 8 / 9 chains** across A/B/C, rotate the extra ninth by shard index. Across 100 shards distribute as evenly as integer arithmetic allows (total 2,500 D4 chains).

D1 Water: native water spread/flow. D2 Lava: native lava spread/flow, do not allow D4's solidification event to silently count as standalone D2. D3 Fire: native ignition/spread/extinguish/eligible block changes, excluding source placement itself. Each requires five genuine center target-state changes and uses the same center inactivity policy.

## 5. Uniform shard distribution and incremental training

A shard is **independently trainable**; every shard has exactly the same top-level composition and 25-biome static counts. Complete every type before publishing the shard; a shortage must leave the shard not READY. Do not release partial or quota-substituted shards merely for throughput. No physical dynamic chain may be split across shards.

Respect seed-disjoint global 80/10/10 train/validation/held-out-test split, fixed for each world seed and immutable across shards. Per-shard exact 800/100/100 *VAE samples* is attainable:
- static: 400/50/50; each biome contributes 16 train, 2 validation, 2 test;
- each dynamic category's 25 chains: 20 train, and 2/3 validation/test (alternate 2/3 and 3/2 across categories, total 80 train, 10 validation and 10 test chains/shard). This yields 400/50/50 dynamic frames.
Balance the biome×terrain×split details across shards by deterministic rotation; do not claim every incompatible fractional intersection is exactly matched within one shard. Never assign a seed to two splits to meet a quota.

Atomic shard publication: each shard has a temporary write location and an immutable final directory `shards/shard_0000` ... `shards/shard_0099`, with grouped `snapshots.zarr`, `anchors.zarr` for accepted D4 anchors, `metadata.parquet`, `manifest.json` (counts, subtype and biome/strata/split checks, source builds, checksum, vocabulary version), and `READY` written last. A DataLoader must ignore anything without READY and may dynamically admit newly READY shards only at safe dataset-refresh boundaries. Training samples previously ready remain available for replay; do not learn exclusively from the newest shard.

**Vocabulary contract:** Same stable, frozen Combined Token mapping (Node Name, raw param2) for every shard and every model instance; no live decoder/embedding resizing while training a growing dataset. Validate coverage with actual registered node/param2 possibilities before allowing streaming training. Unknown IDs cause explicit rejection or a versioned migration, not silent relabeling. Light remains auxiliary `uint8`, not a VAE input. Store `uint16` tokens only when K fits; otherwise reject a silently overflowing vocabulary.

Concurrency: workers own distinct simulator processes, temporary worlds, ports and shard work items. A single coordinator/transactional writer enforces full-chain and full-shard atomicity; do not have N processes unsafely append to one SQLite/Zarr writer. Quota shortfalls, attempt rates and rejection causes are reported by category/subtype/biome.

## 6. Safety, provenance and source preservation

Voxel-only scope stays unchanged: exclude mobs/animals/NPCs/dropped items and entity-originated voxel mutations; unsupported falling-node entities quarantine the affected rollout. Keep source seed, actual world/chunk coordinates, canonical biome classification, model-ready center chunk origin, build+settings, intervention RNG and placements, real sim_time and simulation_step, per-frame event roles/IDs, complete token/light mapping and all rejection audits. No fabricated native state or simulated frame interpolation. Independent reconstruction may differ despite equal mapgen seeds; never claim exact deterministic replay without evidence.

Preserve all old P0/P1/pilot implementations, logs and results in place as historical records. Do not rewrite their measurements. The earlier permanent trajectory/discovery-cache format is deprecated for v2 collection.

## 7. Codex authorization / execution boundary

**Codex MAY implement code, run normal fast unit/static checks, and produce at most TWO accepted native dynamic example clips PER CATEGORY (D1, D2, D3, D4) — EIGHT total, each exactly five center-qualified VAE states with complete 48³ context.** D4 examples should cover two different subtypes; document the third subtype as unit-validated only if not exercised natively. Use bounded attempts and a hard finite wall budget; report shortfalls instead of expanding to a large pilot. Test examples are not evidence of scalable collection yield.

**Codex MUST NOT start** a full shard (1K), 100K data generation, large seed/biome scans, long native trials beyond the two samples/category, accelerated engine experiments, GPU training, VoxGen supplement, VoxEvolve replay or 72/240-frame trajectories. All other native generation and training runs must be launched **by the user on their own server**. Provide explicit user-run CLI commands, but do not execute them autonomously.

Implementation acceptance: unit tests pass; the 8 authorized native examples (or faithfully reported shortfall) have correct shapes, splits, center-only eligibility, D4 triplet/overlap logic and no duplicate VAE frames; safe `READY` semantics; full 100K run remains unstarted. Report exact commit and measured native checks.
