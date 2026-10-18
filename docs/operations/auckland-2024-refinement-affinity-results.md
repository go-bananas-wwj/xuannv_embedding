# Refinement goal: affinity result

Compiled 2026-10-18 17:06 Beijing. New goal met: **False**. Goal remains active unless the acceptance has actually been verified and published.

## Fixed protocol

Same 320 tiles, five spatial folds, 64 dimensions, eleven LCDB classes, common masks/support locations and 200 labels/class. Both MLP and CNN must improve over the archived model, with mean gain at least 0.5 percentage points. Validation selection uses only the fixed archived fold-zero validation anchor. Existing query results have been inspected adaptively; this is exploratory tuning, not a blind test. No Overleaf edits.

## Classification

Macro F1 percentages; differences are percentage points.

| Representation | MLP | CNN |
|---|---:|---:|
| Original AEF | 56.9996 | 56.1062 |
| Original public | 54.3733 | 53.5269 |
| AEF + matched context | 57.6721 | 56.5121 |
| Public PCA + matched context | 57.2150 | 54.7556 |
| Archived accepted 50 m fusion | 58.6834 | 56.8089 |
| Previous validation-selected fusion | 58.8316 | 57.2448 |
| Prior public/semantic 90 m fusion | 59.1703 | 57.2169 |
| Selected new fusion | 58.8877 | 57.0911 |

Gain over archived fusion: MLP +0.2043 pp, CNN +0.2822 pp, mean +0.2432 pp.

The selected representation is `affinity_statistics_k13_t100`. Recipe: `{"affinity_temperature": 1.0, "dimensions": [32, 24, 8], "eligible": true, "kind": "blocks", "name": "affinity_statistics_k13_t100", "post_pool": 13, "sources": ["public", "semantic", "raw"]}`. The AEF control selected using validation alone is `aef_k13_d0_t100`. Controls never select the fused candidate.

Validation selection rule: `mean_gain_after_positive_MLP_and_CNN_validation_gains`.

## All class F1 results

| Class | New MLP | Delta vs archived | Delta vs AEF | New CNN | Delta vs archived | Delta vs AEF |
|---|---:|---:|---:|---:|---:|---:|
| Built-up Area (settlement) | +92.764 | +1.277 | +0.060 | +91.067 | -0.290 | -2.045 |
| Urban Parkland/Open Space | +58.618 | +2.089 | +4.467 | +54.092 | -0.178 | -1.394 |
| Estuarine Open Water | +98.956 | -0.028 | -0.112 | +98.516 | -0.273 | -0.443 |
| Short-rotation Cropland | +8.039 | +0.307 | +1.954 | +6.748 | -1.364 | -0.770 |
| Orchard, Vineyard or Other Perennial Crop | +20.110 | -4.795 | +3.119 | +26.264 | +4.160 | +11.494 |
| High Producing Exotic Grassland | +80.415 | +2.681 | +13.581 | +76.062 | -1.300 | +6.763 |
| Manuka and/or Kanuka | +68.767 | +3.111 | +1.983 | +64.523 | +0.885 | +1.120 |
| Broadleaved Indigenous Hardwoods | +13.770 | +1.885 | -2.570 | +9.579 | +0.984 | -1.886 |
| Indigenous Forest | +90.004 | +0.613 | +0.971 | +87.391 | +0.543 | +0.406 |
| Mangrove | +95.453 | -1.301 | -0.111 | +92.128 | -1.517 | -3.411 |
| Exotic Forest | +20.868 | -3.591 | -2.574 | +21.631 | +1.454 | +1.000 |

## Spatial fold outcomes

| Fold | Archived MLP | New MLP | Archived CNN | New CNN |
|---|---:|---:|---:|---:|
| 0 | 57.821 | 57.107 | 55.517 | 56.704 |
| 1 | 60.493 | 60.248 | 56.450 | 55.150 |
| 2 | 58.336 | 58.322 | 57.601 | 57.140 |
| 3 | 57.606 | 58.906 | 56.129 | 56.728 |
| 4 | 56.048 | 55.778 | 53.635 | 52.223 |

## All LiDAR regression targets

Common 5,000 training points/fold, validation-selected Ridge alpha. Pooled RMSE from summed squared errors/counts. Lower is better. Surface height includes trees and buildings, not building height alone.

| Target | AEF original | AEF matched | Archived fusion | New fusion | Public matched |
|---|---:|---:|---:|---:|---:|
| terrain_elevation_m | 31.3880 | 28.9565 | 50.8396 | 48.1308 | 51.9465 |
| terrain_slope_degrees | 5.0129 | 5.3976 | 6.6069 | 6.5428 | 6.8025 |
| surface_height_mean_m | 3.3363 | 3.9372 | 3.6275 | 3.8814 | 4.0847 |
| surface_height_p90_m | 4.3406 | 5.1034 | 4.6820 | 4.9919 | 5.2510 |

## Complete validation search

| Candidate | MLP | CNN | Minimum gain over archived validation |
|---|---:|---:|---:|
| affinity_basic_k9_t20 | 60.8078 | 59.5659 | +0.0882 |
| affinity_basic_k9_t100 | 61.2905 | 59.6113 | +0.1335 |
| affinity_statistics_k9_t20 | 61.2930 | 59.3741 | -0.1036 |
| affinity_basic_k13_t100 | 61.1402 | 60.2164 | +0.5058 |
| affinity_statistics_k9_t100 | 61.7849 | 59.4898 | +0.0120 |
| affinity_statistics_k13_t100 | 62.2547 | 60.0471 | +0.5693 |

## Reproduction and limitations

The accepted baseline remains a separate read-only, hash-verified archive. The current search reuses its existing public regional encoder and frozen DINOv3 ViT-L/16 SAT-493M aerial features from real 1 m LINZ RGB. DINOv3 is additional global satellite pretraining, not regional-only training. No AEF embeddings, LCDB or LiDAR reference maps enter the candidate embedding. Source hashes, recipe, projections, masks, fixed support locations and all negative outcomes remain available.

The first twelve refinements use public/semantic projection arrays bitwise identical to the accepted model. New default code reproduced the archived output exactly on five fixed unlabeled audit tiles. 867 tests, Black, Ruff, repository gate and package build passed. Continuous variables and rare classes can decline even when pooled classification improves; the tables above retain all outcomes.
