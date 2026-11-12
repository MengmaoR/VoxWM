# VAE pilot native collection report

Status: completed; snapshots: 137; worlds: 1; observed sites: 4

| Category | Snapshots | Target | Shortfall | Natural changed trajectories | Valid clips |
| --- | ---: | ---: | ---: | ---: | ---: |
| fire | 0 | 25000 | 25000 | 0 | 0 |
| lava | 0 | 25000 | 25000 | 0 | 0 |
| ordinary | 89 | 100000 | 99911 | — | — |
| vegetation | 0 | 25000 | 25000 | 0 | 0 |
| water | 48 | 25000 | 24952 | 9 | 6 |

Observed simulation: 96.357 s; engine stepping after initialization (including site search): 99.379 s; world wall time: 133.830 s.

Zero-yield categories: fire, lava, vegetation.
Combined additional-world estimate: UNKNOWN: at least one category/biome lacks measured yield.

Per-clip data, exact timestamps/progress, biome/height counts and conditional estimates are in dataset.sqlite and report.json.
This report does not change quotas, accelerate native physics or synthesize missing clips.

| Biome group | Ordinary snapshots |
| --- | ---: |
| BambooJungle | 0 |
| BirchForest | 0 |
| ColdTaiga | 0 |
| Desert | 0 |
| ExtremeHills | 0 |
| FlowerForest | 6 |
| Forest | 27 |
| IcePlains | 0 |
| IcePlainsSpikes | 0 |
| Jungle | 0 |
| JungleEdge | 25 |
| MangroveSwamp | 0 |
| MegaSpruceTaiga | 0 |
| MegaTaiga | 19 |
| Mesa | 0 |
| MesaBryce | 0 |
| MesaPlateauF | 0 |
| MushroomIsland | 0 |
| Plains | 12 |
| RoofedForest | 0 |
| Savanna | 0 |
| StoneBeach | 0 |
| SunflowerPlains | 0 |
| Swampland | 0 |
| Taiga | 0 |

Terrain counts: {"subsurface": 57, "surface": 41, "underground": 39}
Height distribution: {"count": 137, "min": -24.0, "p50": -8.0, "p90": 8.0, "max": 24.0}

| Dynamic | Distinct states min / p50 / max | Valid clip size histogram | Change bands (all candidates) |
| --- | --- | --- | --- |
| water | 2 / 11.0 / 22 | {0: 3, 8: 6} | {'medium': 5, 'large': 3, 'small': 1} |
| lava | None / None / None | {} | {} |
| fire | 2 / 2.0 / 2 | {0: 1} | {'small': 1} |
| vegetation | None / None / None | {} | {} |
