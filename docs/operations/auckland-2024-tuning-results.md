# Auckland 2024: completed high-resolution tuning results

All three candidates completed 800 successful additional updates and exported all 320 tiles. Validation selection, selected-model classification, regression and figures finished at 2026-10-17 19:33 Beijing time. The supervisor exited normally with zero recovery restarts. No additional training or Overleaf edits were started.

## Selection

The registered mean of fold-zero MLP/CNN validation Macro F1 selected `faster_fusion`: a trained public backbone/auxiliary-head initializer with fresh aerial layers, public learning rate 3e-6 and aerial/fusion learning rate 3e-4. Effective batch 48, 100-update warmup, cosine decay, weight decay 0.01 and the same data/objectives were retained. This is adaptation from the public model, not a continuation of the original LINZ model.

| Candidate | Mean validation Macro F1 (%) |
|---|---:|
| parent | 52.725 |
| continue_low_lr | 51.987 |
| protect_public | 53.282 |
| faster_fusion | 53.326 |

The selected candidate exceeds the original LINZ validation mean by 0.601 percentage points and `protect_public` by only 0.044 points. The original public reference mean is 53.876%; AEF is 56.930%. The small candidate-selection margin does not establish a reliable ordering across training seeds.

## Pooled five-fold classification

Annual mean embedding, 200 training points per eligible class, same eleven classes, label supports, spatial folds, valid pixels and readout rules. Values are percentages.

| Feature | MLP Macro F1 | CNN Macro F1 | MLP mIoU | CNN mIoU |
|---|---:|---:|---:|---:|
| AEF | 57.00 | 56.11 | 48.03 | 47.34 |
| Public embedding | 54.37 | 53.53 | 46.32 | 45.32 |
| Original LINZ embedding | 52.92 | 51.46 | 45.24 | 43.48 |
| Selected tuned LINZ embedding | 53.69 | 51.86 | 45.70 | 44.56 |

Relative to original LINZ, tuned Macro F1 increases by 0.764 / 0.403 percentage points for MLP / CNN. It remains below the public embedding by 0.688 / 1.666 points and below AEF by 3.315 / 4.245 points.

## Per-class F1

Every eligible class is included, including declines.

| LCDB class | Original MLP | Tuned MLP | Difference | Original CNN | Tuned CNN | Difference |
|---|---:|---:|---:|---:|---:|---:|
| Built-up Area (settlement) | 90.60 | 90.90 | 0.30 | 92.13 | 92.63 | 0.50 |
| Urban Parkland/Open Space | 36.37 | 38.26 | 1.89 | 27.34 | 32.61 | 5.27 |
| Estuarine Open Water | 98.94 | 98.93 | -0.01 | 98.81 | 98.66 | -0.15 |
| Short-rotation Cropland | 6.97 | 6.97 | 0.00 | 7.97 | 5.55 | -2.42 |
| Orchard, Vineyard or Other Perennial Crop | 11.57 | 11.95 | 0.38 | 14.23 | 9.53 | -4.70 |
| High Producing Exotic Grassland | 68.77 | 68.04 | -0.74 | 59.31 | 71.00 | 11.68 |
| Manuka and/or Kanuka | 65.23 | 66.15 | 0.92 | 62.57 | 62.60 | 0.04 |
| Broadleaved Indigenous Hardwoods | 8.31 | 9.16 | 0.85 | 6.37 | 4.59 | -1.78 |
| Indigenous Forest | 89.45 | 89.21 | -0.24 | 87.74 | 87.96 | 0.23 |
| Mangrove | 96.99 | 96.92 | -0.07 | 95.14 | 96.44 | 1.30 |
| Exotic Forest | 8.94 | 14.05 | 5.11 | 14.44 | 8.91 | -5.53 |

## Fold-zero test after current-round selection

| Feature | MLP Macro F1 (%) | CNN Macro F1 (%) |
|---|---:|---:|
| AEF | 55.96 | 56.67 |
| Public embedding | 53.49 | 54.14 |
| Original LINZ embedding | 50.66 | 52.15 |
| Selected tuned LINZ embedding | 52.72 | 50.74 |

Fold-zero MLP improves by 2.061 points over original LINZ; CNN declines by 1.411 points. Thus the held-out current-round comparison is mixed even though the five-fold pooled means increase. The benchmark was already inspected before this tuning round. Selection uses fold-zero validation labels, which are test labels in other folds; pooled results are adaptive regional exploration, not a new blind evaluation.

## LiDAR regression

Pooled held-out errors, lower is better. Surface height includes vegetation and buildings.

| Feature | Mean-height MAE (m) | Mean-height RMSE (m) | P90-height MAE (m) | P90-height RMSE (m) |
|---|---:|---:|---:|---:|
| AEF | 2.236 | 3.336 | 3.002 | 4.341 |
| Public embedding | 2.570 | 3.770 | 3.373 | 4.822 |
| Original LINZ embedding | 2.601 | 3.825 | 3.407 | 4.882 |
| Selected tuned LINZ embedding | 2.578 | 3.782 | 3.381 | 4.831 |

The selected high-resolution model partly recovers the original deficit, but does not exceed AEF or the public model on the primary classification or height metrics. Increasing the high-resolution branch learning rate after public initialization helped modestly in this screen; simply continuing the original high-resolution model did not improve validation. These results do not justify a broad claim that aerial fusion improves all tasks.

Selected full weights, source/config identities, representations, training curves and three standard comparison figures remain in the data workspace. The prior models and negative trial results are preserved. AEF attribution: The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind (CC-BY 4.0).
