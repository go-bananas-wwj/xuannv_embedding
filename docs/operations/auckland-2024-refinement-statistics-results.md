# Refinement goal: statistics result

Compiled 2026-10-18 16:28 Beijing. New goal met: **False**. Goal remains active unless the acceptance has actually been verified and published.

## Fixed protocol

Same 320 tiles, five spatial folds, 64 dimensions, eleven LCDB classes, common masks/support locations and 200 labels/class. Both MLP and CNN must improve over the archived model, with mean gain at least 0.5 percentage points. Validation selection uses only the fixed archived fold-zero validation anchor. Existing query results have been inspected adaptively; this is exploratory tuning, not a blind test. No Overleaf edits.

## Classification

Macro F1 percentages; differences are percentage points.

| Representation | MLP | CNN |
|---|---:|---:|
| Original AEF | 56.9996 | 56.1062 |
| Original public | 54.3733 | 53.5269 |
| AEF + matched context | 57.9650 | 56.9066 |
| Public PCA + matched context | 57.2956 | 54.2555 |
| Archived accepted 50 m fusion | 58.6834 | 56.8089 |
| Previous validation winner (90 m) | 59.1703 | 57.2169 |
| Selected new fusion | 58.8316 | 57.2448 |

Gain over archived fusion: MLP +0.1482 pp, CNN +0.4359 pp, mean +0.2920 pp.

The selected representation is `statistics_32_24_8_k9`. Recipe: `{"dimensions": [32, 24, 8], "eligible": true, "kind": "blocks", "name": "statistics_32_24_8_k9", "post_pool": 9, "sources": ["public", "semantic", "raw"]}`. The AEF control selected using validation alone is `aef_k9_d0`. Controls never select the fused candidate.

Validation selection rule: `mean_gain_after_positive_MLP_and_CNN_validation_gains`.

## All class F1 results

| Class | New MLP | Delta vs archived | Delta vs AEF | New CNN | Delta vs archived | Delta vs AEF |
|---|---:|---:|---:|---:|---:|---:|
| Built-up Area (settlement) | +92.583 | +1.095 | -0.122 | +90.975 | -0.381 | -2.136 |
| Urban Parkland/Open Space | +57.930 | +1.401 | +3.778 | +52.433 | -1.836 | -3.053 |
| Estuarine Open Water | +98.863 | -0.121 | -0.205 | +98.765 | -0.025 | -0.194 |
| Short-rotation Cropland | +8.740 | +1.008 | +2.655 | +6.887 | -1.224 | -0.631 |
| Orchard, Vineyard or Other Perennial Crop | +20.519 | -4.386 | +3.528 | +27.740 | +5.635 | +12.970 |
| High Producing Exotic Grassland | +80.294 | +2.561 | +13.461 | +76.524 | -0.838 | +7.225 |
| Manuka and/or Kanuka | +68.183 | +2.527 | +1.399 | +65.075 | +1.436 | +1.672 |
| Broadleaved Indigenous Hardwoods | +12.914 | +1.029 | -3.426 | +9.942 | +1.347 | -1.524 |
| Indigenous Forest | +89.850 | +0.458 | +0.817 | +87.781 | +0.933 | +0.796 |
| Mangrove | +95.470 | -1.284 | -0.094 | +93.575 | -0.070 | -1.964 |
| Exotic Forest | +21.802 | -2.657 | -1.640 | +19.995 | -0.182 | -0.636 |

## Spatial fold outcomes

| Fold | Archived MLP | New MLP | Archived CNN | New CNN |
|---|---:|---:|---:|---:|
| 0 | 57.821 | 57.082 | 55.517 | 56.893 |
| 1 | 60.493 | 60.058 | 56.450 | 57.058 |
| 2 | 58.336 | 58.482 | 57.601 | 55.990 |
| 3 | 57.606 | 59.117 | 56.129 | 57.195 |
| 4 | 56.048 | 55.342 | 53.635 | 51.515 |

## All LiDAR regression targets

Common 5,000 training points/fold, validation-selected Ridge alpha. Pooled RMSE from summed squared errors/counts. Lower is better. Surface height includes trees and buildings, not building height alone.

| Target | AEF original | AEF matched | Archived fusion | New fusion | Public matched |
|---|---:|---:|---:|---:|---:|
| terrain_elevation_m | 31.3880 | 29.4902 | 50.8396 | 48.5862 | 52.4639 |
| terrain_slope_degrees | 5.0129 | 5.2442 | 6.6069 | 6.5446 | 6.8132 |
| surface_height_mean_m | 3.3363 | 3.7864 | 3.6275 | 3.9175 | 4.0886 |
| surface_height_p90_m | 4.3406 | 4.9117 | 4.6820 | 5.0454 | 5.2671 |

## Complete validation search

| Candidate | MLP | CNN | Minimum gain over archived validation |
|---|---:|---:|---:|
| statistics_40_16_8_k5 | 60.4381 | 59.3126 | -0.1963 |
| statistics_40_16_8_k9 | 61.4277 | 58.6935 | -0.7842 |
| statistics_32_16_16_k5 | 59.1736 | 56.4717 | -3.0061 |
| statistics_32_16_16_k9 | 59.6943 | 57.5005 | -1.9772 |
| statistics_32_24_8_k5 | 60.8399 | 58.4561 | -1.0217 |
| statistics_32_24_8_k9 | 61.8090 | 59.5643 | +0.0866 |
| statistics_40_16_8_localraw | 61.0521 | 58.6212 | -0.8566 |
| statistics_32_16_16_localraw | 59.4743 | 56.8413 | -2.6364 |

## Reproduction and limitations

The accepted baseline remains a separate read-only, hash-verified archive. The current search reuses its existing public regional encoder and frozen DINOv3 ViT-L/16 SAT-493M aerial features from real 1 m LINZ RGB. DINOv3 is additional global satellite pretraining, not regional-only training. No AEF embeddings, LCDB or LiDAR reference maps enter the candidate embedding. Source hashes, recipe, projections, masks, fixed support locations and all negative outcomes remain available.

The first twelve refinements use public/semantic projection arrays bitwise identical to the accepted model. New default code reproduced the archived output exactly on five fixed unlabeled audit tiles. 867 tests, Black, Ruff, repository gate and package build passed. Continuous variables and rare classes can decline even when pooled classification improves; the tables above retain all outcomes.
