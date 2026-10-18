# Refinement goal: context result

Compiled 2026-10-18 15:16 Beijing. New goal met: **False**. Goal remains active unless the acceptance has actually been verified and published.

## Fixed protocol

Same 320 tiles, five spatial folds, 64 dimensions, eleven LCDB classes, common masks/support locations and 200 labels/class. Both MLP and CNN must improve over the archived model, with mean gain at least 0.5 percentage points. Validation selection uses only the fixed archived fold-zero validation anchor. Existing query results have been inspected adaptively; this is exploratory tuning, not a blind test. No Overleaf edits.

## Classification

Macro F1 percentages; differences are percentage points.

| Representation | MLP | CNN |
|---|---:|---:|
| Original AEF | 56.9996 | 56.1062 |
| Original public | 54.3733 | 53.5269 |
| AEF + matched context | 57.9650 | 56.9066 |
| Public PCA + matched context | 57.4181 | 54.6285 |
| Archived accepted 50 m fusion | 58.6834 | 56.8089 |
| Selected new fusion | 59.1703 | 57.2169 |

Gain over archived fusion: MLP +0.4869 pp, CNN +0.4080 pp, mean +0.4475 pp.

The selected representation is `wide_k9`. Recipe: `{"dimensions": [48, 16], "eligible": true, "kind": "blocks", "name": "wide_k9", "post_pool": 9, "sources": ["public", "semantic"]}`. The AEF control selected using validation alone is `aef_k9_d0`. Controls never select the fused candidate.

## All class F1 results

| Class | New MLP | Delta vs archived | Delta vs AEF | New CNN | Delta vs archived | Delta vs AEF |
|---|---:|---:|---:|---:|---:|---:|
| Built-up Area (settlement) | +92.026 | +0.539 | -0.678 | +91.149 | -0.208 | -1.963 |
| Urban Parkland/Open Space | +58.038 | +1.509 | +3.886 | +55.096 | +0.826 | -0.390 |
| Estuarine Open Water | +98.900 | -0.085 | -0.168 | +98.432 | -0.357 | -0.527 |
| Short-rotation Cropland | +7.930 | +0.198 | +1.845 | +6.863 | -1.249 | -0.656 |
| Orchard, Vineyard or Other Perennial Crop | +26.055 | +1.150 | +9.064 | +22.668 | +0.564 | +7.898 |
| High Producing Exotic Grassland | +77.712 | -0.021 | +10.879 | +76.400 | -0.962 | +7.101 |
| Manuka and/or Kanuka | +66.716 | +1.061 | -0.068 | +66.812 | +3.174 | +3.409 |
| Broadleaved Indigenous Hardwoods | +11.956 | +0.071 | -4.384 | +11.477 | +2.882 | +0.012 |
| Indigenous Forest | +90.446 | +1.055 | +1.414 | +85.501 | -1.347 | -1.484 |
| Mangrove | +96.439 | -0.315 | +0.875 | +94.637 | +0.991 | -0.903 |
| Exotic Forest | +24.655 | +0.196 | +1.213 | +20.351 | +0.174 | -0.280 |

## Spatial fold outcomes

| Fold | Archived MLP | New MLP | Archived CNN | New CNN |
|---|---:|---:|---:|---:|
| 0 | 57.821 | 58.754 | 55.517 | 57.675 |
| 1 | 60.493 | 59.967 | 56.450 | 55.510 |
| 2 | 58.336 | 58.635 | 57.601 | 56.925 |
| 3 | 57.606 | 57.704 | 56.129 | 55.095 |
| 4 | 56.048 | 56.167 | 53.635 | 54.380 |

## All LiDAR regression targets

Common 5,000 training points/fold, validation-selected Ridge alpha. Pooled RMSE from summed squared errors/counts. Lower is better. Surface height includes trees and buildings, not building height alone.

| Target | AEF original | AEF matched | Archived fusion | New fusion | Public matched |
|---|---:|---:|---:|---:|---:|
| terrain_elevation_m | 31.3880 | 29.4902 | 50.8396 | 50.3439 | 53.6176 |
| terrain_slope_degrees | 5.0129 | 5.2442 | 6.6069 | 6.6330 | 6.8368 |
| surface_height_mean_m | 3.3363 | 3.7864 | 3.6275 | 3.9059 | 4.0626 |
| surface_height_p90_m | 4.3406 | 4.9117 | 4.6820 | 5.0421 | 5.2227 |

## Complete validation search

| Candidate | MLP | CNN | Minimum gain over archived validation |
|---|---:|---:|---:|
| wide_k7 | 61.0342 | 59.7976 | +0.3198 |
| residual_k7_d20 | 60.9199 | 59.9472 | +0.2855 |
| residual_k7_d40 | 60.5994 | 59.7270 | -0.0351 |
| wide_k9 | 61.1797 | 59.8467 | +0.3689 |
| residual_k9_d20 | 61.1102 | 59.7017 | +0.2239 |
| residual_k9_d40 | 60.7115 | 59.3767 | -0.1011 |
| source_p1_s5 | 59.6077 | 59.2281 | -1.0267 |
| source_p3_s5 | 60.3022 | 59.3560 | -0.3322 |
| source_p1_s9 | 60.0615 | 58.9796 | -0.5730 |
| source_p3_s9 | 60.5149 | 59.8710 | -0.1196 |
| weight_s50 | 60.5770 | 59.6035 | -0.0574 |
| weight_s200 | 60.6053 | 59.8316 | -0.0291 |

## Reproduction and limitations

The accepted baseline remains a separate read-only, hash-verified archive. The current search reuses its existing public regional encoder and frozen DINOv3 ViT-L/16 SAT-493M aerial features from real 1 m LINZ RGB. DINOv3 is additional global satellite pretraining, not regional-only training. No AEF embeddings, LCDB or LiDAR reference maps enter the candidate embedding. Source hashes, recipe, projections, masks, fixed support locations and all negative outcomes remain available.

The first twelve refinements use public/semantic projection arrays bitwise identical to the accepted model. New default code reproduced the archived output exactly on five fixed unlabeled audit tiles. 867 tests, Black, Ruff, repository gate and package build passed. Continuous variables and rare classes can decline even when pooled classification improves; the tables above retain all outcomes.
