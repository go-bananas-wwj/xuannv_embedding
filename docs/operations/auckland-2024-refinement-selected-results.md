# Refinement goal: selected result

Compiled 2026-10-18 18:22 Beijing. New goal met: **True**. Goal remains active unless the acceptance has actually been verified and published.

## Fixed protocol

Same 320 tiles, five spatial folds, 64 dimensions, eleven LCDB classes, common masks/support locations and 200 labels/class. Both MLP and CNN must improve over the archived model, with mean gain at least 0.5 percentage points. Validation selection uses only the fixed archived fold-zero validation anchor. Existing query results have been inspected adaptively; this is exploratory tuning, not a blind test. No Overleaf edits.

## Classification

Macro F1 percentages; differences are percentage points.

| Representation | MLP | CNN |
|---|---:|---:|
| Original AEF | 56.9996 | 56.1062 |
| Previously evaluated AEF + 90m mean | 57.9650 | 56.9066 |
| Original public | 54.3733 | 53.5269 |
| AEF + matched context | 57.6721 | 56.5121 |
| Public PCA + matched context | 57.6327 | 55.4344 |
| Archived accepted 50 m fusion | 58.6834 | 56.8089 |
| Previous validation-selected fusion | 58.8877 | 57.0911 |
| Prior public/semantic 90 m fusion | 59.1703 | 57.2169 |
| Prior public-statistics fusion | 58.8316 | 57.2448 |
| Matched original 1m encoder grid | 60.1743 | 58.1233 |
| Selected new fusion | 60.0338 | 58.6201 |

Gain over archived fusion: MLP +1.3504 pp, CNN +1.8112 pp, mean +1.5808 pp.

The selected representation is `selected_input_fusion`. Recipe: `{"affinity_temperature": 1.0, "dimensions": [40, 24], "eligible": true, "kind": "blocks", "name": "selected_input_fusion", "post_pool": 13, "sources": ["public", "semantic"]}`. The AEF control selected using validation alone is `aef_k13_d0_t100`. Controls never select the fused candidate.

Validation selection rule: `prospectively_registered_paired_input_positive_mean_validation_gain`.

## All class F1 results

| Class | New MLP | Delta vs archived | Delta vs AEF | New CNN | Delta vs archived | Delta vs AEF |
|---|---:|---:|---:|---:|---:|---:|
| Built-up Area (settlement) | +92.521 | +1.033 | -0.184 | +91.362 | +0.005 | -1.749 |
| Urban Parkland/Open Space | +58.774 | +2.245 | +4.623 | +58.830 | +4.561 | +3.344 |
| Estuarine Open Water | +98.767 | -0.217 | -0.301 | +98.726 | -0.064 | -0.233 |
| Short-rotation Cropland | +8.108 | +0.375 | +2.022 | +6.297 | -1.815 | -1.221 |
| Orchard, Vineyard or Other Perennial Crop | +34.079 | +9.174 | +17.088 | +26.101 | +3.997 | +11.331 |
| High Producing Exotic Grassland | +79.682 | +1.948 | +12.849 | +77.054 | -0.309 | +7.754 |
| Manuka and/or Kanuka | +67.848 | +2.192 | +1.064 | +66.843 | +3.205 | +3.440 |
| Broadleaved Indigenous Hardwoods | +11.564 | -0.322 | -4.776 | +12.764 | +4.169 | +1.298 |
| Indigenous Forest | +90.523 | +1.132 | +1.491 | +87.014 | +0.166 | +0.029 |
| Mangrove | +95.799 | -0.955 | +0.235 | +95.410 | +1.765 | -0.129 |
| Exotic Forest | +22.708 | -1.751 | -0.733 | +24.420 | +4.243 | +3.789 |

## Spatial fold outcomes

| Fold | Archived MLP | New MLP | Archived CNN | New CNN |
|---|---:|---:|---:|---:|
| 0 | 57.821 | 60.064 | 55.517 | 60.824 |
| 1 | 60.493 | 61.875 | 56.450 | 56.669 |
| 2 | 58.336 | 59.218 | 57.601 | 58.156 |
| 3 | 57.606 | 56.841 | 56.129 | 56.555 |
| 4 | 56.048 | 57.187 | 53.635 | 55.594 |

## All LiDAR regression targets

Common 5,000 training points/fold, validation-selected Ridge alpha. Pooled RMSE from summed squared errors/counts. Lower is better. Surface height includes trees and buildings, not building height alone.

| Target | AEF original | AEF matched | Archived fusion | New fusion | Public matched |
|---|---:|---:|---:|---:|---:|
| terrain_elevation_m | 31.3880 | 28.9565 | 50.8396 | 49.2276 | 53.2625 |
| terrain_slope_degrees | 5.0129 | 5.3976 | 6.6069 | 6.5344 | 6.8192 |
| surface_height_mean_m | 3.3363 | 3.9372 | 3.6275 | 3.7694 | 4.0863 |
| surface_height_p90_m | 4.3406 | 5.1034 | 4.6820 | 4.8531 | 5.2308 |

## Exact original-1m encoder-grid regression control

Same source information and fusion settings; only encoder sampling differs. Lower RMSE is better.

| Target | Matched original 1m grid | Selected input |
|---|---:|---:|
| terrain_elevation_m | 49.1549 | 49.2276 |
| terrain_slope_degrees | 6.5483 | 6.5344 |
| surface_height_mean_m | 3.8547 | 3.7694 |
| surface_height_p90_m | 4.9517 | 4.8531 |

## Complete validation search

| Candidate | MLP | CNN | Minimum gain over archived validation |
|---|---:|---:|---:|
| scale05_p48_s16_k5 | 61.6332 | 60.3129 | +0.8351 |
| scale05_p48_s16_k9 | 62.1865 | 59.8116 | +0.3338 |
| scale05_p48_s16_k13 | 62.1192 | 59.1245 | -0.3533 |
| scale05_p40_s24_k5 | 61.2708 | 58.6626 | -0.8151 |
| scale05_p40_s24_k9 | 62.4110 | 58.1056 | -1.3721 |
| scale05_p40_s24_k13 | 62.5626 | 60.1619 | +0.6842 |
| scale05_p32_s32_k5 | 60.9525 | 59.0063 | -0.4715 |
| scale05_p32_s32_k9 | 61.9328 | 58.6858 | -0.7920 |
| scale05_p32_s32_k13 | 62.2774 | 58.4275 | -1.0502 |

## Reproduction and limitations

The accepted baseline remains a separate read-only, hash-verified archive. The current search reuses its existing public regional encoder and frozen DINOv3 ViT-L/16 SAT-493M aerial features from real 1 m LINZ RGB. DINOv3 is additional global satellite pretraining, not regional-only training. No AEF embeddings, LCDB or LiDAR reference maps enter the candidate embedding. Source hashes, recipe, projections, masks, fixed support locations and all negative outcomes remain available.

The first twelve refinements use public/semantic projection arrays bitwise identical to the accepted model. New default code reproduced the archived output exactly on five fixed unlabeled audit tiles. 867 tests, Black, Ruff, repository gate and package build passed. Continuous variables and rare classes can decline even when pooled classification improves; the tables above retain all outcomes.


## Encoder input-scale provenance

The actual source image information remains 1m. Its pixels are bilinearly resampled to a 0.5m encoder grid, giving an 8m ViT token spacing before the common 10m output. This does not add observed detail and is not an accuracy result from the separately downloaded true 0.5m raster. The exact matched original-1m encoder-grid control is shown above.
## Interpretation and stopping point

The classification target is met. Relative to the original archived model, gains
are 1.3504 pp for MLP and 1.8112 pp for CNN, mean 1.5808 pp. Against the exact
matched original-grid fusion, aligned input changes MLP by -0.1405 pp and CNN by
+0.4968 pp. Thus the total improvement cannot be attributed entirely to image
resampling: source allocation and feature-aware context also changed. The final
paired input choice was locked using validation only before reading paired query
metrics. The original-grid counterpart's higher query MLP is retained in the table.

This is a classification-oriented result. All four selected regression targets
remain worse than original AEF; surface-height errors also exceed the original
archived model. Relative to the exact matched fusion, scale alignment slightly
reduces both height errors and slope error, while terrain elevation error rises.
No claim of superiority across all tasks or classes is supported.

Actual 0.5m imagery from the original 0.25m collection has been acquired, and its
bounds/CRS/real read resolution were audited. It did not enter these scores. The
new target is achieved, so no further native-0.5m model sweep is started. Existing
archives, all negative rounds and the ready dataset are preserved. No Overleaf edits.
