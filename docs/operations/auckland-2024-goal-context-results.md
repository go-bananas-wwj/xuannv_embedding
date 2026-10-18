# Regional high-resolution fusion goal: completed context wave

Results compiled 2026-10-18 14:42 Beijing. Registered original-baseline acceptance: **True**. Exceeds all three equivalent-context controls for both heads: **True**. No Overleaf edits.

## Scope and fixed protocol

Same 320 Auckland-west tiles, 64 dimensions, eleven LCDB classes, 200 labelled points/class, five spatial folds, masks, support locations, optimizer and readout model selection rules. PCA and feature aggregation use no LCDB or LiDAR targets. Images from all 320 tiles train the regional encoder (transductive setting). The benchmark has already been inspected and tuned adaptively; these are exploratory regional results, not an independent blind test or evidence of universal superiority.

## All primary comparisons

Macro F1 in percent. Positive differences are percentage points.

| Representation | MLP | CNN |
|---|---:|---:|
| AEF | 56.9996 | 56.1062 |
| Public | 54.3733 | 53.5269 |
| AEF + 50 m context | 57.6135 | 56.2847 |
| Public + 50 m context | 55.0298 | 53.6695 |
| Public PCA48 + 50 m context | 57.2232 | 54.6322 |
| Public + aerial semantics + 50 m context | 58.6834 | 56.8089 |

| Reference | MLP difference | CNN difference |
|---|---:|---:|
| AEF | +1.6838 | +0.7027 |
| Public | +4.3101 | +3.2820 |
| AEF + 50 m context | +1.0699 | +0.5241 |
| Public + 50 m context | +3.6536 | +3.1394 |
| Public PCA48 + 50 m context | +1.4602 | +2.1767 |

## All class F1 results

No classes or negative differences are suppressed.

| LCDB class | AEF MLP | Fused MLP | Delta | AEF CNN | Fused CNN | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Built-up Area (settlement) | 92.705 | 91.488 | -1.217 | 93.112 | 91.357 | -1.755 |
| Urban Parkland/Open Space | 54.151 | 56.529 | 2.378 | 55.486 | 54.270 | -1.217 |
| Estuarine Open Water | 99.068 | 98.984 | -0.084 | 98.959 | 98.790 | -0.169 |
| Short-rotation Cropland | 6.085 | 7.732 | 1.647 | 7.518 | 8.112 | 0.593 |
| Orchard, Vineyard or Other Perennial Crop | 16.991 | 24.905 | 7.914 | 14.770 | 22.104 | 7.334 |
| High Producing Exotic Grassland | 66.833 | 77.733 | 10.900 | 69.300 | 77.363 | 8.063 |
| Manuka and/or Kanuka | 66.784 | 65.656 | -1.128 | 63.403 | 63.639 | 0.235 |
| Broadleaved Indigenous Hardwoods | 16.340 | 11.886 | -4.455 | 11.465 | 8.595 | -2.871 |
| Indigenous Forest | 89.032 | 89.391 | 0.359 | 86.985 | 86.847 | -0.137 |
| Mangrove | 95.564 | 96.754 | 1.190 | 95.539 | 93.645 | -1.894 |
| Exotic Forest | 23.442 | 24.459 | 1.017 | 20.631 | 20.177 | -0.454 |

## Spatial fold outcomes

All fold scores; descriptive only because the benchmark was tuned adaptively.

| Fold | AEF MLP | Fused MLP | AEF CNN | Fused CNN |
|---|---:|---:|---:|---:|
| 0 | 55.962 | 57.821 | 56.669 | 55.517 |
| 1 | 55.878 | 60.493 | 55.136 | 56.450 |
| 2 | 56.356 | 58.336 | 55.893 | 57.601 |
| 3 | 59.184 | 57.606 | 55.269 | 56.129 |
| 4 | 52.844 | 56.048 | 52.949 | 53.635 |

## Independent LiDAR regression

Ridge, 5,000 common training points/fold, validation-selected alpha. Pooled RMSE = sqrt(total squared error / total valid pixels). Surface height includes trees and buildings; it is not building-only height.

| Target | AEF RMSE | Public RMSE | Fused RMSE | Difference from AEF (%) |
|---|---:|---:|---:|---:|
| terrain_elevation_m | 31.3880 | 55.1844 | 50.8396 | +61.972 |
| terrain_slope_degrees | 5.0129 | 6.7817 | 6.6069 | +31.798 |
| surface_height_mean_m | 3.3363 | 3.7698 | 3.6275 | +8.727 |
| surface_height_p90_m | 4.3406 | 4.8220 | 4.6820 | +7.867 |

## Method and reproduction

The chosen representation combines the existing public S2/S1/Landsat annual embedding with frozen DINOv3 ViT-L/16 SAT-493M features from actual 2024 LINZ RGB read at 1 m. PCA retains 48 public and 16 semantic coordinates, each branch is normalized; concatenate to 64 dimensions, average vectors within a 5 by 5 window (50 m), and normalize again. This is inference-time feature fusion of two existing encoders, not a new jointly trained checkpoint. SAT-493M is additional global satellite pretraining and must be disclosed. AEF embeddings are never model inputs.

The native RGB product is 0.25 m; the downloaded mosaic uses real 1 m overviews, not upsampling of prior 3 m input. Semantic extraction uses overlapping 512-pixel windows and aligned 16-pixel patch tokens with weighted stitching. There are no duplicated monthly aerial acquisitions. No LiDAR, LCDB, or reference building data enters the embedding.

Source hashes, exact weights, normalizations, geographic index, native and fusion projections, software versions, recipe, and output hash are recorded in the data-workspace reproduction manifest. Reconstructing five fixed unlabeled audit tiles from saved projections reproduced the exported float16 arrays exactly. All earlier unsuccessful trials remain recorded in the wave-one and native-wave reports.

## Validation selection and controls

The pre-registered selection maximizes the smaller of MLP/CNN validation margins over AEF; ties use the mean. It selected the same candidate before full query scoring. All context candidates:

| Recipe | MLP validation | CNN validation | Minimum margin |
|---|---:|---:|---:|
| context_s16_k3 | 60.1070 | 59.3383 | +2.4944 |
| context_s16_k5 | 60.6344 | 59.4778 | +2.6338 |
| context_s24_k3 | 60.0323 | 58.7057 | +1.8618 |
| context_s24_k5 | 60.5331 | 59.0959 | +2.2519 |
| context_s32_k3 | 60.6864 | 58.1190 | +1.2750 |
| context_s32_k5 | 61.0063 | 59.3094 | +2.4655 |

The three matched context controls use the same unit-vector aggregation and unmodified readout rules. They are diagnostic controls and never choose the fused candidate. The public PCA48 control preserves the exact public projection and adds sixteen zero coordinates, isolating the contribution of high-resolution semantic information.

## Verification

864 pytest tests pass, plus Black, Ruff, repository content gate and package build. Real full-area NPU readouts and native extraction completed. Compact protocols and results are synchronized to GitHub; arrays, images, checkpoints, credentials and detailed logs stay outside Git.
