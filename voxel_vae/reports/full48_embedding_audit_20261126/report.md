# Full 48³ embedding occurrence and unseen-token audit

PASS

Counts describe token IDs / embedding rows present in data; no embedding values, model forward or training were computed.
U0 remains the 274 tokens with zero occurrences in the original **center16³ train** baseline.

## Source and validation

Per the user's source clarification, this audit reads the complete local Static/Dynamic canonical repacks directly from tar.zst streams. No raw dataset was extracted to disk or modified.
The original clip-local IDs and old dataset_index.json are not substituted or rebuilt. Current pack vocabularies are checked semantically against the frozen training vocabulary; original LUT availability is recorded below.

```json
{
  "reference": {
    "canonical_K": 1580,
    "canonical_semantic_sha256": "25af338d8200eaceedf62bd33dc76d769dcec27f526e6df34280c43c9b3f979a",
    "training_vocabulary_file_sha256": "bc383e57f882433099ef4e27832f24e19093649456e2ef217a51df0bc81758d0",
    "token_frequency_file_sha256": "7acc34875022771203718528f0f6db628012c22b22726c442c96f6d281a336f2",
    "split_file_sha256": "eb6c3f14ed77932d99ce505bc442051605dfa0016be8f45f5b2e3b3898aacdb6",
    "original_dataset_index_expected_sha256": "9b0cdb13e41fee88acb18a3f60091f582842806806fb64657acad31142acbe3b",
    "center_train_unique": 1306,
    "U0": 274,
    "heldout_only": 27,
    "never_seen_in_center_splits": 247,
    "original_training_roots_mapping_checks": [
      {
        "root": "/user-data/txj/VoxWM/outputs/vae_100k_clip1",
        "local_K": 1538,
        "mapping_count": 1538,
        "ids_in_range": true,
        "injective": true,
        "identity_mapping": false,
        "recorded_local_vocabulary_sha256": "205c9ebf4d53888fd84afd28e7351cc44f17380482b5143b859fbfc5da0aa4e9",
        "recorded_dataset_manifest_sha256": "bc38033eeac782f13715d4ae6252fc2ea2f47da16161f0b255b479b7a9591838",
        "actual_original_local_vocabulary_verified": false,
        "semantic_check_boundary": "original local vocabulary files not supplied; structural LUT checks only"
      },
      {
        "root": "/user-data/txj/VoxWM/outputs/vae_100k_clip2",
        "local_K": 1537,
        "mapping_count": 1537,
        "ids_in_range": true,
        "injective": true,
        "identity_mapping": false,
        "recorded_local_vocabulary_sha256": "021e952508fa34ff5c1c46bc6666c567c9b52c7effa8647557f5868f9cdf8056",
        "recorded_dataset_manifest_sha256": "bc38033eeac782f13715d4ae6252fc2ea2f47da16161f0b255b479b7a9591838",
        "actual_original_local_vocabulary_verified": false,
        "semantic_check_boundary": "original local vocabulary files not supplied; structural LUT checks only"
      }
    ]
  },
  "sources": [
    {
      "archive": "/Users/tianxj/Desktop/VoxWM-Dataset/VAE/data/vae_200k_by_category/vae_100k_static.tar.zst",
      "root_name": "vae_100k_static",
      "category": "Static",
      "canonical_K": 1580,
      "canonical_semantic_sha256": "25af338d8200eaceedf62bd33dc76d769dcec27f526e6df34280c43c9b3f979a",
      "source_vocabulary_file_sha256": "aafe78de25dce55341c9822e4f05f60f3c4d7bc0267b2a7047135ae73122d49c",
      "repack_local_to_canonical": "identity; every exact semantic pair and raw param2 checked",
      "ids_in_range": true,
      "injective": true,
      "param2_preserved": true,
      "split_patch_bytes_match_published": true,
      "original_training_roots_mapping_checks": [
        {
          "root": "/user-data/txj/VoxWM/outputs/vae_100k_clip1",
          "local_K": 1538,
          "mapping_count": 1538,
          "ids_in_range": true,
          "injective": true,
          "identity_mapping": false,
          "recorded_local_vocabulary_sha256": "205c9ebf4d53888fd84afd28e7351cc44f17380482b5143b859fbfc5da0aa4e9",
          "recorded_dataset_manifest_sha256": "bc38033eeac782f13715d4ae6252fc2ea2f47da16161f0b255b479b7a9591838",
          "actual_original_local_vocabulary_verified": false,
          "semantic_check_boundary": "original local vocabulary files not supplied; structural LUT checks only"
        },
        {
          "root": "/user-data/txj/VoxWM/outputs/vae_100k_clip2",
          "local_K": 1537,
          "mapping_count": 1537,
          "ids_in_range": true,
          "injective": true,
          "identity_mapping": false,
          "recorded_local_vocabulary_sha256": "021e952508fa34ff5c1c46bc6666c567c9b52c7effa8647557f5868f9cdf8056",
          "recorded_dataset_manifest_sha256": "bc38033eeac782f13715d4ae6252fc2ea2f47da16161f0b255b479b7a9591838",
          "actual_original_local_vocabulary_verified": false,
          "semantic_check_boundary": "original local vocabulary files not supplied; structural LUT checks only"
        }
      ],
      "original_dataset_index": {
        "status": "not supplied",
        "locations": [],
        "original_index_identity_verified": false
      },
      "dataset_index_search": {
        "status": "not supplied",
        "locations": [],
        "original_index_identity_verified": false
      }
    },
    {
      "archive": "/Users/tianxj/Desktop/VoxWM-Dataset/VAE/data/vae_200k_by_category/vae_100k_dynamic.tar.zst",
      "root_name": "vae_100k_dynamic",
      "category": "Dynamic",
      "canonical_K": 1580,
      "canonical_semantic_sha256": "25af338d8200eaceedf62bd33dc76d769dcec27f526e6df34280c43c9b3f979a",
      "source_vocabulary_file_sha256": "aafe78de25dce55341c9822e4f05f60f3c4d7bc0267b2a7047135ae73122d49c",
      "repack_local_to_canonical": "identity; every exact semantic pair and raw param2 checked",
      "ids_in_range": true,
      "injective": true,
      "param2_preserved": true,
      "split_patch_bytes_match_published": true,
      "original_training_roots_mapping_checks": [
        {
          "root": "/user-data/txj/VoxWM/outputs/vae_100k_clip1",
          "local_K": 1538,
          "mapping_count": 1538,
          "ids_in_range": true,
          "injective": true,
          "identity_mapping": false,
          "recorded_local_vocabulary_sha256": "205c9ebf4d53888fd84afd28e7351cc44f17380482b5143b859fbfc5da0aa4e9",
          "recorded_dataset_manifest_sha256": "bc38033eeac782f13715d4ae6252fc2ea2f47da16161f0b255b479b7a9591838",
          "actual_original_local_vocabulary_verified": false,
          "semantic_check_boundary": "original local vocabulary files not supplied; structural LUT checks only"
        },
        {
          "root": "/user-data/txj/VoxWM/outputs/vae_100k_clip2",
          "local_K": 1537,
          "mapping_count": 1537,
          "ids_in_range": true,
          "injective": true,
          "identity_mapping": false,
          "recorded_local_vocabulary_sha256": "021e952508fa34ff5c1c46bc6666c567c9b52c7effa8647557f5868f9cdf8056",
          "recorded_dataset_manifest_sha256": "bc38033eeac782f13715d4ae6252fc2ea2f47da16161f0b255b479b7a9591838",
          "actual_original_local_vocabulary_verified": false,
          "semantic_check_boundary": "original local vocabulary files not supplied; structural LUT checks only"
        }
      ],
      "original_dataset_index": {
        "status": "not supplied",
        "locations": [],
        "original_index_identity_verified": false
      },
      "dataset_index_search": {
        "status": "not supplied",
        "locations": [],
        "original_index_identity_verified": false
      }
    }
  ],
  "actual_split_counts": {
    "train": 160000,
    "validation": 20000,
    "test": 20000
  },
  "center16_all_token_counts_match_baseline": true,
  "all_snapshot_shapes_and_token_ranges_pass": true,
  "all_chain_anchor_links_pass": true,
  "all_occurrence_denominators_pass": true,
  "original_source_snapshots_unique": true,
  "world_seed_split_consistency_pass": true,
  "old_dataset_index_rebuilt": false
}
```

## 1. Full48 Static / Dynamic frequency

| Split | Category | Snapshots (sample probability denominator) | Voxels (voxel probability denominator) | Unique tokens | U0 tokens |
|---|---|---:|---:|---:|---:|
| train | Static | 79,870 | 8,832,983,040 | 1398 | 204 |
| train | Dynamic | 80,130 | 8,861,736,960 | 1387 | 120 |
| validation | Static | 10,080 | 1,114,767,360 | 1275 | 107 |
| validation | Dynamic | 9,920 | 1,097,072,640 | 1140 | 35 |
| test | Static | 10,050 | 1,111,449,600 | 1298 | 130 |
| test | Dynamic | 9,950 | 1,100,390,400 | 1163 | 41 |

`static_dynamic_embedding_frequency.csv` contains all 1,580 tokens for each of the six groups (9,480 rows), sorted by voxel count descending within each group. Probabilities are fractions in [0,1], not percentages. Each of the five dynamic states counts as one snapshot; anchors are excluded.
Voxel probabilities sum to one in each group. Sample probabilities are per-token presence probabilities and do not have to sum to one.

Pooling all splits for distribution only: U0 appears in 213 Static token types and 125 Dynamic token types; their union covers 272 of 274. The remaining 2 are absent from every ordinary full48 snapshot. This pooled result is not used in the Train-only comparison below.

## 2. D4 anchors vs their own five-state chains

- Anchors: **5000** (all splits).
- Anchors containing U0: **54**.
- Unique U0 tokens observed in anchors: **35**.
- Anchor-exclusive unique U0 tokens: **1**.
- Anchor-exclusive voxel occurrences: **1**.

Exclusivity is decided for each anchor and its own linked five full48 states first. A token may be exclusive in one anchor and shared in another. The exclusive occurrence total includes only the exclusive anchor-token pairs.
The anchor CSV includes every canonical token. Detailed anchor/chain links concern U0 only; trained-token rows retain aggregate counts but no per-anchor unseen-audit links. Paired-chain occurrence counts are summed only over anchors that contain that token, once per anchor's own chain.

Complete anchor-exclusive list:

| Token | Node name | Raw param2 | Exclusive anchor voxels |
|---:|---|---:|---:|
| 1142 | `mcl_ocean:fire_coral` | 0 | 1 |

## 3. Train full48 unseen-token sets

- U_static: **204**; U_dynamic: **120**.
- Intersection: **63**; Static-only: **141**.
- **Dynamic-only Unseen Tokens: 57**.
- U0 absent from both full48 train categories: **13**.

Only Train ordinary snapshots are used here. All set sizes are unique canonical tokens, with no rarity threshold. Category contributions may overlap and must not be added as disjoint sets.

| Dynamic category | Dynamic-only unique tokens contributed |
|---|---:|
| D1 | 28 |
| D2 | 22 |
| D3 | 24 |
| D4 | 33 |

Complete Dynamic-only list:

| Token | Node name | Raw param2 | Static voxels | Dynamic voxels | D1 | D2 | D3 | D4 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | `mcl_amethyst:large_amethyst_bud` | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| 6 | `mcl_amethyst:large_amethyst_bud` | 1 | 0 | 3 | 0 | 0 | 0 | 3 |
| 7 | `mcl_amethyst:large_amethyst_bud` | 2 | 0 | 4 | 0 | 0 | 0 | 4 |
| 8 | `mcl_amethyst:large_amethyst_bud` | 5 | 0 | 4 | 0 | 0 | 0 | 4 |
| 9 | `mcl_amethyst:medium_amethyst_bud` | 0 | 0 | 42 | 0 | 2 | 0 | 40 |
| 10 | `mcl_amethyst:medium_amethyst_bud` | 1 | 0 | 36 | 0 | 6 | 0 | 30 |
| 11 | `mcl_amethyst:medium_amethyst_bud` | 2 | 0 | 49 | 0 | 20 | 0 | 29 |
| 233 | `mcl_core:grass_path` | 0 | 0 | 66 | 66 | 0 | 0 | 0 |
| 318 | `mcl_core:jungleleaves` | 0 | 0 | 10 | 0 | 0 | 10 | 0 |
| 324 | `mcl_core:jungleleaves` | 8 | 0 | 990 | 790 | 0 | 200 | 0 |
| 382 | `mcl_core:leaves_orphan` | 4 | 0 | 18 | 0 | 18 | 0 | 0 |
| 627 | `mcl_core:vine` | 130 | 0 | 5 | 0 | 0 | 0 | 5 |
| 628 | `mcl_core:vine` | 131 | 0 | 20 | 0 | 0 | 0 | 20 |
| 636 | `mcl_core:void` | 1 | 0 | 46895 | 0 | 14690 | 0 | 32205 |
| 644 | `mcl_core:void` | 9 | 0 | 20 | 0 | 0 | 0 | 20 |
| 645 | `mcl_core:void` | 10 | 0 | 35 | 0 | 0 | 0 | 35 |
| 646 | `mcl_core:void` | 11 | 0 | 5 | 0 | 0 | 0 | 5 |
| 647 | `mcl_core:void` | 12 | 0 | 5 | 0 | 0 | 0 | 5 |
| 648 | `mcl_core:void` | 13 | 0 | 25 | 0 | 0 | 0 | 25 |
| 757 | `mcl_doors:wooden_door_b_1` | 1 | 0 | 3 | 3 | 0 | 0 | 0 |
| 758 | `mcl_doors:wooden_door_t_1` | 1 | 0 | 3 | 3 | 0 | 0 | 0 |
| 777 | `mcl_fire:eternal_fire` | 0 | 0 | 18 | 14 | 1 | 3 | 0 |
| 778 | `mcl_fire:eternal_fire` | 5 | 0 | 1 | 0 | 0 | 0 | 1 |
| 779 | `mcl_fire:eternal_fire` | 10 | 0 | 1 | 1 | 0 | 0 | 0 |
| 854 | `mcl_fire:fire` | 74 | 0 | 3 | 0 | 0 | 1 | 2 |
| 857 | `mcl_fire:fire` | 77 | 0 | 2 | 0 | 0 | 2 | 0 |
| 863 | `mcl_fire:fire` | 84 | 0 | 2 | 0 | 0 | 2 | 0 |
| 864 | `mcl_fire:fire` | 85 | 0 | 1 | 0 | 0 | 1 | 0 |
| 865 | `mcl_fire:fire` | 86 | 0 | 1 | 0 | 0 | 1 | 0 |
| 866 | `mcl_fire:fire` | 87 | 0 | 1 | 0 | 0 | 1 | 0 |
| 1031 | `mcl_loom:loom` | 3 | 0 | 3 | 3 | 0 | 0 | 0 |
| 1125 | `mcl_ocean:dead_brain_coral` | 1 | 0 | 103 | 9 | 7 | 15 | 72 |
| 1126 | `mcl_ocean:dead_brain_coral` | 9 | 0 | 72 | 23 | 9 | 4 | 36 |
| 1128 | `mcl_ocean:dead_bubble_coral` | 1 | 0 | 152 | 16 | 17 | 26 | 93 |
| 1129 | `mcl_ocean:dead_bubble_coral` | 9 | 0 | 72 | 6 | 13 | 28 | 25 |
| 1130 | `mcl_ocean:dead_bubble_coral_block` | 0 | 0 | 800 | 66 | 37 | 80 | 617 |
| 1131 | `mcl_ocean:dead_fire_coral` | 0 | 0 | 5 | 0 | 0 | 0 | 5 |
| 1132 | `mcl_ocean:dead_fire_coral` | 1 | 0 | 34 | 9 | 0 | 5 | 20 |
| 1133 | `mcl_ocean:dead_fire_coral` | 9 | 0 | 239 | 55 | 13 | 33 | 138 |
| 1134 | `mcl_ocean:dead_fire_coral_block` | 0 | 0 | 910 | 72 | 71 | 67 | 700 |
| 1135 | `mcl_ocean:dead_horn_coral` | 0 | 0 | 4 | 0 | 0 | 4 | 0 |
| 1136 | `mcl_ocean:dead_horn_coral` | 1 | 0 | 222 | 14 | 34 | 45 | 129 |
| 1137 | `mcl_ocean:dead_horn_coral` | 9 | 0 | 120 | 10 | 16 | 30 | 64 |
| 1138 | `mcl_ocean:dead_horn_coral_block` | 0 | 0 | 803 | 40 | 62 | 134 | 567 |
| 1139 | `mcl_ocean:dead_tube_coral` | 1 | 0 | 24 | 5 | 0 | 4 | 15 |
| 1140 | `mcl_ocean:dead_tube_coral` | 9 | 0 | 146 | 37 | 11 | 25 | 73 |
| 1141 | `mcl_ocean:dead_tube_coral_block` | 0 | 0 | 828 | 81 | 49 | 111 | 587 |
| 1156 | `mcl_ocean:kelp_dirt` | 28 | 0 | 5 | 5 | 0 | 0 | 0 |
| 1157 | `mcl_ocean:kelp_dirt` | 32 | 0 | 5 | 5 | 0 | 0 | 0 |
| 1159 | `mcl_ocean:kelp_dirt` | 41 | 0 | 5 | 5 | 0 | 0 | 0 |
| 1163 | `mcl_ocean:kelp_dirt` | 69 | 0 | 5 | 5 | 0 | 0 | 0 |
| 1249 | `mcl_ocean:kelp_gravel` | 112 | 0 | 5 | 0 | 5 | 0 | 0 |
| 1331 | `mcl_ocean:kelp_sand` | 112 | 0 | 67 | 6 | 4 | 0 | 57 |
| 1357 | `mcl_portals:end_portal_frame_eye` | 0 | 0 | 5 | 0 | 5 | 0 | 0 |
| 1358 | `mcl_portals:end_portal_frame_eye` | 1 | 0 | 5 | 0 | 5 | 0 | 0 |
| 1375 | `mcl_stairs:slab_wood` | 0 | 0 | 92 | 92 | 0 | 0 | 0 |
| 1376 | `mcl_stairs:slab_wood_top` | 0 | 0 | 10 | 10 | 0 | 0 | 0 |

Analysis wall time: 228.72 seconds. This is a data-distribution audit, not evidence of training or generalization quality.

## Reproduce

Run from the VoxWM repository root (CPU Python with NumPy/Zarr/numcodecs and the `zstd` CLI). Output must be new/empty; raw archives are only read.

```bash
.venv/bin/python -B -m voxel_vae.full48_audit \
  --archive '/Users/tianxj/Desktop/VoxWM-Dataset/VAE/data/vae_200k_by_category/vae_100k_static.tar.zst' \
  --archive '/Users/tianxj/Desktop/VoxWM-Dataset/VAE/data/vae_200k_by_category/vae_100k_dynamic.tar.zst' \
  --output outputs/voxel_vae/full48_embedding_audit_20261126_rerun
```
