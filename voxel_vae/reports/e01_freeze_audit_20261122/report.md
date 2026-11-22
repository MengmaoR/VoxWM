# VoxWM VAE Freeze Audit

## 1. Checkpoint Provenance

`checkpoint_best.pt` at step 27,000 was selected on validation total loss (0.2533732643).
Training source commit `f52feff424fb61e8d9aa80b176c3e1314d118760`.
Audit source commit `bd25721663de7d7db280b2919981728068f8f262`.
Checkpoint SHA256 `543d728d20f3164f73d4e703ce341fd4ec2ad5ef0ea79c6ec10299168ec765a1`.

## 2. Split Integrity

| Partition | States | World seeds |
| --- | --- | --- |
| train | 160000 | 43068 |
| validation | 20000 | 5389 |
| test | 20000 | 5340 |

World-seed intersections are train_vs_validation = 0, train_vs_test = 0, validation_vs_test = 0.
Test covers all 20,000 states and 1,990 complete dynamic chains, with 0 D4 anchor samples.

## 3. Full Held-Out Test Results

The 20,000-state test was a confirmatory posterior-mean reconstruction evaluation of the validation-selected 27k model.
Exact Combined-Token accuracy **95.53%**. Non-air exact accuracy **93.06%**.

| Metric | Value |
| --- | --- |
| Cross entropy | 0.147644 |
| Masked total | 0.261004 |
| Static CE | 0.104778 |
| Dynamic unchanged CE | 0.189393 |
| Dynamic changed CE | 0.631513 |
| Raw KL | 3388.73 |
| Weighted KL | 0.00338873 |

## 4. Static Reconstruction

Static exact token accuracy **96.72%**. Non-air accuracy **95.21%**.
Biome and stratum results are in [test_metrics.json](test_metrics.json).

## 5. Dynamic Reconstruction

| Metric | Test |
| --- | --- |
| Dynamic exact | 94.32% |
| Changed exact | 85.90% |
| Unchanged exact | 94.35% |
| Predicted change recall | 94.17% |
| Changed-pair exact | 78.46% |
| Unchanged stability | 98.27% |

Changed-voxel metrics use the union of adjacent changes at each state's position. Pair metrics use changes between two adjacent states.

## 6. D1 / D2 / D3 / D4

| Category | Changed voxels | Exact | Non-air | Changed exact | Unchanged exact | Change recall | Pair exact | Stability | Changed CE | Unchanged CE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | 40708 | 96.02% | 91.82% | 87.84% | 96.06% | 93.01% | 77.74% | 98.86% | 0.275476 | 0.139154 |
| D2 | 28275 | 94.77% | 92.87% | 94.45% | 94.77% | 98.48% | 91.62% | 98.02% | 0.171863 | 0.169727 |
| D3 | 12735 | 95.02% | 90.30% | 53.56% | 95.07% | 85.26% | 33.36% | 98.68% | 1.67106 | 0.171053 |
| D4 | 61639 | 91.54% | 87.48% | 87.37% | 91.57% | 94.11% | 80.13% | 97.54% | 0.407653 | 0.277637 |

## 7. D3 Fire Error Decomposition

| Changed outcome | Voxels | Share |
| --- | --- | --- |
| Exact | 6821 | 53.56% |
| Param2 only | 2876 | 22.58% |
| Node error | 3038 | 23.86% |

Node-name accuracy 76.14%. Param2 accuracy given a correct node 70.34% (6,821/9,697).
Fire presence recall 83.78%, with 8,354 Changed positions containing fire in the ground truth.

**D3 structural reconstruction remains a material limitation.**

The adjacent-pair analysis covers 1,606 distinct semantic transitions. 80 meet the support threshold of 20 voxels.

### Most frequent transitions

| GT t | GT t+1 | Voxels | Pair exact | Node pair exact | Pair errors |
| --- | --- | --- | --- | --- | --- |
| `mcl_fire:fire` (3) | `air` (0) | 68 | 58.82% | 64.71% | 28 |
| `mcl_fire:fire` (4) | `air` (0) | 60 | 66.67% | 75.00% | 20 |
| `mcl_fire:fire` (5) | `air` (0) | 59 | 66.10% | 81.36% | 20 |
| `mcl_fire:fire` (6) | `air` (0) | 57 | 64.91% | 75.44% | 20 |
| `mcl_fire:fire` (2) | `air` (0) | 50 | 56.00% | 74.00% | 22 |

### Largest contributions to pair errors

| GT t | GT t+1 | Voxels | Pair exact | Node pair exact | Pair errors |
| --- | --- | --- | --- | --- | --- |
| `air` (0) | `mcl_fire:fire` (20) | 43 | 23.26% | 62.79% | 33 |
| `air` (0) | `mcl_fire:fire` (26) | 37 | 13.51% | 51.35% | 32 |
| `air` (0) | `mcl_fire:fire` (22) | 40 | 22.50% | 35.00% | 31 |
| `mcl_fire:fire` (3) | `air` (0) | 68 | 58.82% | 64.71% | 28 |
| `air` (0) | `mcl_fire:fire` (16) | 49 | 44.90% | 67.35% | 27 |

### Lowest pair accuracy among sufficiently supported transitions

| GT t | GT t+1 | Voxels | Pair exact | Node pair exact | Pair errors |
| --- | --- | --- | --- | --- | --- |
| `air` (0) | `mcl_fire:fire` (26) | 37 | 13.51% | 51.35% | 32 |
| `air` (0) | `mcl_fire:fire` (29) | 26 | 15.38% | 38.46% | 22 |
| `mcl_fire:fire` (13) | `mcl_fire:fire` (15) | 23 | 21.74% | 91.30% | 18 |
| `mcl_fire:fire` (16) | `air` (0) | 23 | 21.74% | 47.83% | 18 |
| `air` (0) | `mcl_fire:fire` (22) | 40 | 22.50% | 35.00% | 31 |

### Spatial distribution

| D3 Changed location | Errors | Changed sites | Error rate |
| --- | --- | --- | --- |
| Boundary | 1329 | 2138 | 62.16% |
| Interior | 4585 | 10597 | 43.27% |
| Isolated | 1145 | 3183 | 35.97% |
| Non-isolated | 4769 | 9552 | 49.93% |

Boundary sites touch a face of the center crop. Isolated sites have no face-adjacent Changed voxel in the same state. Axis and distance histograms are in [d3_spatial_errors.json](d3_spatial_errors.json).

## 8. Embedding Coverage Audit

1,306 tokens occur in train. 27 occur only in validation/test. 247 are absent from all center crops.
274 embedding rows have L2 drift at most `1e-6`.
Spearman correlation between frequency and drift is 0.880572.
No embedding optimization failure detected. Non-drifting rows correspond to unseen input tokens.

| Train frequency | Tokens | Median L2 | Mean L2 |
| --- | --- | --- | --- |
| 0 | 274 | 0 | 0 |
| 1–10 | 216 | 0.0940423 | 0.10233 |
| 11–100 | 300 | 0.248179 | 0.272317 |
| 101–1k | 316 | 0.419213 | 0.462499 |
| 1k–10k | 247 | 0.707207 | 0.784886 |
| >10k | 227 | 0.718599 | 0.805317 |

## 9. Latent Distribution

The train set has 16/16 active channels at the standard-deviation threshold `1e-3`. Test has 16/16 active channels at the E01 variance threshold `1e-4`.
Test posterior mean has mean 0.00469898 and std 1.54397. Mean posterior std is 0.113374.
Adjacent dynamic latent L2 over 7,960 test pairs has mean 20.2155, std 8.59867, min 1.25554, max 55.3289, with 0 exact collisions.

| Train group | States | Global mean | Global std | Minimum | Maximum |
| --- | --- | --- | --- | --- | --- |
| Global | 160000 | 0.00663838 | 1.54213 | -16.375 | 14.5 |
| Static | 79870 | -0.00526082 | 1.53321 | -11.5 | 12 |
| D1 | 20135 | 0.0919017 | 1.53639 | -15.25 | 13.125 |
| D2 | 20025 | -0.0474103 | 1.56497 | -16.375 | 14.5 |
| D3 | 19910 | 0.0820421 | 1.51798 | -14.8125 | 14 |
| D4 | 20060 | -0.0524519 | 1.57721 | -15.5 | 14.375 |

## 10. Downstream Normalization

[latent_normalization.json](latent_normalization.json) contains a single per-channel mean and population std from 160,000 train posterior means. Downstream volumes use `(mu - mean) / (std + 1e-6)` and invert with the same channel statistics.

## 11. Remaining Limitations

Reconstruction scores assess the current voxel representation. The dynamics and generation models require their own evaluation.
The semantic decomposition uses hard-token predictions. The reported fire errors reflect the tested 16³ crops and the fixed held-out split.

## 12. Freeze Decision

**FREEZE WITH KNOWN D3 LIMITATION**

| Criterion | Observed | Target | Status |
| --- | --- | --- | --- |
| Static exact | 96.72% | 95.00% | PASS |
| Static non-air exact | 95.21% | 90.00% | PASS |
| Changed exact | 85.90% | 80.00% | PASS |
| Predicted change recall | 94.17% | 90.00% | PASS |
| Unchanged stability | 98.27% | 97.00% | PASS |
| Latent | 16/16 train channels, 0 zero-L2 pairs | Active channels and distinguishable adjacent pairs | PASS |
| Embedding | 0 seen low-drift rows | Meaningful seen-row updates | PASS |
