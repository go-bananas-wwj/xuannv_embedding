# Native aerial goal wave: result and next registered experiment

Goal remains active. Both pooled MLP and CNN must exceed their AEF and original public baselines; one successful head is insufficient. All figures below are percentages; differences are percentage points.

## Fixed evaluation

Same 320 tiles, five spatial folds, 64 dimensions, 200 labelled support points/class, eleven LCDB classes, unchanged masks, support locations and training rules. The benchmark has already been inspected adaptively: this is exploratory regional evaluation, not a new blind test. LCDB and LiDAR never fit the feature projections.

## Pooled classification

| Representation | MLP Macro F1 | CNN Macro F1 |
|---|---:|---:|
| AEF | 56.9996 | 56.1062 |
| Original public | 54.3733 | 53.5269 |
| Public PCA48 + 16 zero coordinates | 56.1181 | 53.4098 |
| Public PCA48 + native aerial semantic16 | 57.2654 | 55.7251 |

The fused candidate exceeds AEF by 0.266 pp for MLP but trails by 0.381 pp for CNN. Against the identical public PCA48 projection control, gains are 1.147 pp and 2.315 pp. Therefore both projection and aerial semantics contribute; the method does not yet meet the goal.

## All registered validation recipes

Selection maximizes the smaller MLP/CNN validation margin over AEF; ties use the mean. Test results do not select the winner. DINO-only and public-rotation rows are diagnostics and ineligible.

| Recipe | MLP | CNN | Minimum margin |
|---|---:|---:|---:|
| public_rotation64 | 56.1556 | 54.3239 | -2.5200 |
| native_blocks_8 | 58.6293 | 57.0130 | +0.1690 |
| native_blocks_16 | 59.0272 | 58.2458 | +1.4018 |
| native_blocks_24 | 59.3042 | 58.1044 | +1.2605 |
| native_blocks_32 | 59.9566 | 57.7423 | +0.8984 |
| native_blocks_48 | 59.0672 | 57.9421 | +1.0982 |
| native_joint64 | 58.8928 | 57.7175 | +0.8736 |
| native_semantic_only | 54.3040 | 53.1122 | -3.7318 |

## All class F1 results

| LCDB class | AEF MLP | Fused MLP | Delta | AEF CNN | Fused CNN | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Built-up Area (settlement) | 92.705 | 90.120 | -2.584 | 93.112 | 90.549 | -2.563 |
| Urban Parkland/Open Space | 54.151 | 53.004 | -1.147 | 55.486 | 52.079 | -3.407 |
| Estuarine Open Water | 99.068 | 98.773 | -0.295 | 98.959 | 98.591 | -0.368 |
| Short-rotation Cropland | 6.085 | 6.244 | 0.158 | 7.518 | 7.664 | 0.145 |
| Orchard, Vineyard or Other Perennial Crop | 16.991 | 22.053 | 5.062 | 14.770 | 22.395 | 7.624 |
| High Producing Exotic Grassland | 66.833 | 73.625 | 6.792 | 69.300 | 72.703 | 3.403 |
| Manuka and/or Kanuka | 66.784 | 66.213 | -0.571 | 63.403 | 62.056 | -1.347 |
| Broadleaved Indigenous Hardwoods | 16.340 | 12.867 | -3.474 | 11.465 | 8.368 | -3.098 |
| Indigenous Forest | 89.032 | 88.650 | -0.383 | 86.985 | 85.483 | -1.502 |
| Mangrove | 95.564 | 96.465 | 0.901 | 95.539 | 92.546 | -2.994 |
| Exotic Forest | 23.442 | 21.906 | -1.536 | 20.631 | 20.542 | -0.089 |

## Independent LiDAR regression

Pooled RMSE from total squared error / total valid pixels, with the same 5,000 training points/fold and validation-selected Ridge alpha. Surface height includes vegetation and buildings, not building height alone.

| Target | AEF RMSE | Public RMSE | Fused RMSE |
|---|---:|---:|---:|
| terrain_elevation_m | 31.3880 | 55.1844 | 51.8620 |
| terrain_slope_degrees | 5.0129 | 6.7817 | 6.6148 |
| surface_height_mean_m | 3.3363 | 3.7698 | 3.3123 |
| surface_height_p90_m | 4.3406 | 4.8220 | 4.3212 |

## Actual resolution and source provenance

Real 2024 LINZ RGB was acquired from the official Auckland 2024 0.25 m collection at 1 m overview resolution, restricted to the same study footprint. This is not interpolation of the prior 3 m cache. The mosaic is 20,480 by 25,600 pixels in EPSG:2193 with 96.10% valid coverage. Source manifests, geometry checks and native overviews are retained in the data workspace.

Frozen DINOv3 ViT-L/16 SAT-493M features use overlapping 512-pixel windows with native 16-pixel alignment, weighted stitching, correct SAT normalization and geographic pixel centres. PCA is fitted on at most 65,536 unlabeled tokens. This method uses additional global satellite pretraining and must not be described as trained from scratch only on the local tiles. A synthetic upsampled tile served only as a hardware/geometry check, never as an accuracy input.

## Next registered wave: spatial context

Test six fixed label-free 64-D representations: public/semantic dimensions 48+16, 40+24 and 32+32, each aggregated over 3 by 3 or 5 by 5 cells (30 or 50 m). Average latent unit vectors within each tile, then normalize. Do not smooth labels or predictions. Footprints, masks, output resolution and readout rules stay fixed. Retain the previous native-16 candidate as an eligible validation-only parent fallback. No new encoder training is needed.

Implementation checks: 864 pytest tests pass; Black, Ruff, repository gate and package build pass. Full results and source hashes remain in the data workspace. No Overleaf edits.
