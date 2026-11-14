INCONCLUSIVE — P1 code ready; native collection not executed

2026-11-15. Based on VoxWM master `f3be580f355db97b53f83797a8b1d72678674eff` after `git pull --ff-only`. P0 source/data/results are preserved.

The 2026-11-12 Manifest and 2026-11-13 data protocol were read. The named `VoxWM_VAE_P1_DATA_PROTOCOL_2026-11-14.md` was not found in accessible Drive searches; this implementation follows the supplied P1 Implementation prompt as explicitly authorized.

Implemented: independent fresh/delayed worlds with frozen candidate coordinates and lifecycle clocks; physical eligibility, context stratification and seeded selection; controlled water/lava/native young plants and the bounded 20-point fire rule; lava/contact/product evidence and observed fire merges; 600/1200/1800-second growth windows; post-intervention progress sampling; P0 Combined Tokens and training storage; lossless raw capture deltas, separate short-event pool; scene-level statistics and state/change visualizations.

Completed checks:

- Python and Lua compilation / CLI parsing, without loading the game.
- 14 no-game unit tests: attempt ratios, stratified selection, fire random fallback and failed-initialization retry limit, intervention exclusion, distinct progress/light-only handling, reeds/growth/decay, cross-Chunk reaction evidence, pre-existing stone exclusion, fire merges, plant windows, raw token/light round trips, short-pool isolation, manual budget and paired plan.
- Raw-phase visualization CLI and lossless replay succeeded without importing Craftium.
- Small full A/B/C Dry Run: 18 independent mock instances, all four categories and all three plants; 96 synthetic standard snapshots exported. Six B fixtures intentionally stabilized before the observation window. All quantities are fixtures, not native collection yields; native measurement fields are null.

Not executed / not yet verified: any P1 Craftium run, physical candidate availability in new seeds, actual initialization success, native long-window bamboo/reeds/vine growth, source flow, reaction capture rate or multi-fire propagation. No 200K collection or GPU training. Synthetic tests do not establish physical validity of a generated world.

Known limits:

- Finite candidate search may report no valid point, especially 20 dispersed fire sites or water-interaction strata. Failed contexts retain their attempt allocation; there is no unlimited retry or reassignment.
- Each scene observes 12 fixed Chunks (32×32×48 total). Physics can continue outside the observed domain; missing observations are not Air. The raw log reconstructs captured frames only.
- Native placement and ABM outcomes can depend on native RNG (including upstream bamboo placement's wall-clock-seeded RNG). Python/selection RNG and plan are reproducible; exact native outcomes are not promised.
- Water/lava reactions need sampled before/after states and contact geometry; very fast unobserved contacts remain unconfirmed. Fire merge labels are conservative observed-component evidence.
- First capture follows real generation/activation and selection overhead; it is not mapgen time zero. A includes changes during selection; B uses an independent instance and excludes its waiting phase from standard clips.

A single-seed, single-site ABC plan requests 2 simulation-hours initially, at most 5 with all plant extensions, excluding unmeasured initialization/discovery/IO costs. Wall time and 200K cost remain unmeasured. P0 short-run throughput is not a P1 benchmark.

User commands and output format: `vae_p1/README.md`. Future run interpretation: `vae_p1/VAE_P1_RESULTS_TEMPLATE.md`. All real runs require an explicit manual `--native --wall-seconds ...` command.
