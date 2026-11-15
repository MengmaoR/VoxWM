# VoxWM — Unified VAE Data Collection Protocol

Version: 2026-11-15. Status: current data-collection authority. The latest Unified Collection user instruction supersedes historical P0/P1 collection rules. The 2026-11-12 VAE architecture remains unchanged. This Markdown is mirrored to Drive; P0/P1 source, results and protocol history remain historical.

## Scope and representation

Simulate and record voxel nodes only. No animals, mobs, NPCs, dropped items or non-voxel entity state enters the world dataset. The invisible, non-interacting collector player is a technical interface only. Chunks are fixed world-aligned 16³ in ENU (east,north,up). One uint16 token indexes `(stable Node Name, raw uint8 param2)` in one append-only dataset vocabulary, bounded at 65,536 entries. Raw uint8 light is auxiliary, never a VAE input or loss channel. Encoding/decoding must be exact. World seeds, not frames, determine train/validation/test splits.

## Native voxel-only boundary

Set mobs_spawn=false, mobs_griefing=false, mobs_drop_items=false, mcl_doTileDrops=false, enable_particles=false and enable_fire=true. These flags alone are insufficient. Install the isolated API guard in a copied runtime builtin before game mods load; block entity creation and item factories including aliases saved by later mods. Count actual active Lua entities, dropped items, blocked attempts and entity-context voxel mutations. The spawner's doll/render/timer path is disabled while the spawner voxel remains. No shared upstream source or old environment is edited.

Falling nodes use temporary native entities. Their transitions are explicitly OUTSIDE this collection domain: any attempted falling entity quarantines the entire rollout before training samples commit. No replacement falling physics, silently frozen falling scene, or invented dynamics is permitted. Original native fluid/fire/vegetation ABMs, probabilities, growth metadata and engine time remain unchanged. Accepted rollouts require zero nontechnical entities, zero dropped items, zero entity-induced voxel mutations and no unsupported transition.

## Discovery, index and reconstruction

Use Broad Discovery → persistent Scene Index → independent reconstruction → early native replay → optional intervention → native collection. Scan multiple regions in one seeded v7 world without long per-candidate waits. Index seed/build/config, aligned chunk and region bounds, biome/terrain/height, node counts, neighborhood signals, score and provenance. Flowing fluid, stone or tall plants are signals, not proof of an observed dynamic event. Rank useful scenes while balancing source seeds, biomes and geometry.

Cache valid ordinary discovery captures in grouped Zarr so later Ordinary export does not start a new world per snapshot. Reconstruction starts a fresh world with the indexed seed, coordinates, build and settings. Use the Mapgen callback's provided VoxelManip, not a recursive map read. Preserve early callback and pre-Python native captures with real simulated time and globalstep counts. Initial capture is not claimed to be mapgen time zero.

Compare three independent reconstructions. Hash canonical Node Name/raw param2, excluding light, per chunk. Report complete-state equality, matching chunk subsets and temporal trajectory equality separately. Equal terrain seed does not prove equal Lua/wall-clock RNG decoration or ABM timing. Different initial variants must not be mislabeled as the same initial state. Equal initial states with different later dynamics are independent native rollouts.

## Water and lava initialization

Prefer observed natural evolution; use controlled initialization when needed or for diversity. Random scene-level source count defaults to 1–4, with spacing 4 voxels; placement defaults to 90% attached and 10% suspended, configurable weights rather than quotas. Both are legal. Do not densely cover a chunk or write subsequent states manually. Capture immediately after intervention before native evolution begins.

Lava supports the same placements and multiple sources. Query water-bearing indexed regions for interactions. Record actual proximity/path information and direct/longer-path outcomes without requiring contact to succeed. Recognize water source/flowing, lava source/flowing, obsidian, cobblestone and stone. Confirm a reaction only with a new state transition plus sampled contact geometry; pre-existing products are not reaction evidence. Fast unobserved contacts remain unconfirmed. Event-attribution uncertainty is separate from the validity of an observed VAE voxel state.

## Fire initialization

Scan all actual legal ignition positions; record their count and rejection context. Seeded spatial sparsification and a candidate-density budget form a variable-size candidate set; there is no fixed 20-point rule. Independently draw p=0.1 for each retained candidate. If none is selected, randomly choose one; if every actual ignition fails, permit one further bounded fallback attempt. Record candidate set, probability, RNG seed, positions, outcomes, time, propagation and observed component merges. Initial ignitions do not contribute native propagation. Fire must not create dropped items.

## Natural vegetation trimming

Discover existing bamboo, sugar cane and vines; never plant new ones. Group all plants by real roots/components and world coordinates across chunk boundaries. Do not pre-filter by maturity or hidden height cap. Load enough neighborhood to identify a complete natural plant; unresolved boundaries remain indexed but are not pretended to be complete.

For natural bamboo/reeds height h, independently draw retained height x uniformly from 1..h. If x=h, leave unchanged. Otherwise use native node_dig on node x+1 and let original callbacks perform cascading removal. Keep at least the original root, never set its metadata or height, and record actual remaining members, requested/actual height and metadata comparison. If the native cascade differs, report the result; do not manually remove extra nodes to fake truncation.

For vines, preserve direct supported source anchors, trim non-anchor chains/branches using native digging, preserve direction and attachment, and permit real decay. Pure decay is not growth. Capture before, immediate after, settling and subsequent native phases. A short settling budget separates immediate intervention cascades from later changes; native decay after that remains explicitly labeled.

Observe a whole chunk community, including disconnected plants. Suggested native budgets are 600/1200/1800 simulated seconds, configurable and extendable under explicit wall-time limits. One mature plant or one 8-state chunk does not terminate other active events. Never repeatedly trim to produce snapshots, call growth manually, or accelerate ABMs.

## Ordinary and quotas

Fixed Pilot targets: Ordinary100,000; D1/D2/D3/D4 each25,000; total200,000. Ordinary includes normal dynamic content; it need not be motionless. Balance 25 major biome groups (~4,000 each), surface/shallow/deep/cave/air transitions and diverse geometry. Preserve air/stone families while bounding repetitions. Enforce a maximum source-seed share and do not fill the dataset with easy biomes. Record exact same-source cross-category overlap and semantic node-family fractions.

## Sampling, storage and resume

Preserve all captured times through initial full states plus sparse token/light deltas, including zero-change indexed observations. The log reconstructs captured states, not unobserved intervals. VAE samples are distinct independent voxel states. Preserve initial and final stable states once, with progress-spaced intermediate states; do not pad repeats. Target5–8 states per standard dynamic clip; real2–4 states go to a separate auxiliary pool. Clip length never limits the raw process or growth time. Intervention changes do not contribute native change progress; valid post-intervention states may be the baseline.

Dataset files: scene_index.parquet, scenes.jsonl, interventions.jsonl, vocabulary.json, snapshots.zarr, metadata.parquet, trajectories/, aux_short_event/, statistics/collection_statistics.json and visualizations/. Retain seed, commits, settings, coordinates, initial hashes, source/discovery phase, replay ID, intervention RNG/actions, phase, simulated time/step, split, event attribution and audits.

SQLite is the committed sample/job journal; Zarr stores compressed arrays. Save vocabulary additions before references commit. Resume truncates uncommitted array tails, retains committed samples, skips completed region/category jobs and restarts incomplete scenes as new independent rollouts. It does not pretend to restore native physical time. Ordinary discovery caches are reused. Same-source duplicate token states are bounded; unrelated worlds are not globally purged. Parquet is an exported checkpoint of the journal.

## Validation and reporting

Unit tests cover API blocking, drops, attached/suspended multi-source sampling, variable fire sets/fallback, index/coordinate/seed serialization, natural trimming/root/anchor preservation, cross-chunk plants, intervention separation, vocabulary and raw/storage resume. Only real native runs can validate physics.

Bounded native integration covers natural water discovery/replay/collection; controlled fluid; ignition/propagation; natural bamboo/reeds trimming/growth; zero entities/drops; and three independent reconstructions. Report actual outcomes and deficits. Distinguish simulation and wall time, exposure and independence units, initialization and evolution, complete/censored runs. No synthetic fixture can enter the native pool. If any endpoint fails or yield is unmeasured, report INCONCLUSIVE/NO-GO and do not invent 200K feasibility or costs.

The user starts formal 200K collection and all GPU training. This implementation task authorizes bounded local native integration only; it does not authorize launching the formal dataset or any training.
