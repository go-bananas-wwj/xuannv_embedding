# Auckland 2024: completed longer/protected training results

All three routes completed the registered 2,400 additional successful updates. Stage validation, selected-model final evaluation, regression and figures finished at 2026-10-18 09:11 Beijing time. The supervisor exited normally with zero restarts; all final telemetry reports zero skipped optimizer updates. No new run or Overleaf edits followed.

## Stage selection

Selection uses the registered mean of fold-zero validation MLP/CNN Macro F1 at 200 training points/class. Entries below are percentages; these are not final test metrics.

| Route | 800 updates | 1,600 updates | 2,400 updates | Selected step |
|---|---:|---:|---:|---:|
| long_continuation | 53.500 | 53.185 | 54.814 | 2400 |
| frozen_public | 52.294 | 54.366 | 52.941 | 1600 |
| frozen_anchored | 53.446 | 53.706 | 53.132 | 1600 |

Public reference validation mean is 53.876%; the prior selected LINZ mean is 53.326%. The selected long-continuation stage reaches 54.814%, with MLP 53.265% and CNN 56.363%. Both frozen routes select update 1,600 rather than the final checkpoint; longer training is not uniformly beneficial. They received stage validation only, not separate complete final test suites.

## Pooled five-fold test classification

Annual mean embedding, eleven LCDB classes, 200 points/class, unchanged supports, spatial folds, masks and readout rules.

| Feature | MLP Macro F1 (%) | CNN Macro F1 (%) |
|---|---:|---:|
| AEF | 57.00 | 56.11 |
| Public embedding | 54.37 | 53.53 |
| Original LINZ | 52.92 | 51.46 |
| Previous selected LINZ | 53.69 | 51.86 |
| Long-trained selected LINZ | 53.39 | 53.20 |

Relative to the previous selected LINZ model, MLP decreases by 0.295 percentage points while CNN increases by 1.337 points. The selected long-trained model remains below public by 0.983 / 0.328 points and below AEF by 3.610 / 2.908 points for MLP / CNN. The goal of exceeding both references across these primary comparisons is not achieved.

## Fold-zero test after current-round selection

| Feature | MLP Macro F1 (%) | CNN Macro F1 (%) |
|---|---:|---:|
| AEF | 55.96 | 56.67 |
| Public embedding | 53.49 | 54.14 |
| Original LINZ | 50.66 | 52.15 |
| Previous selected LINZ | 52.72 | 50.74 |
| Long-trained selected LINZ | 52.23 | 56.93 |

The selected CNN exceeds AEF by 0.260 points and public by 2.788 points on fold zero. This is a partial improvement on one fixed split and must not replace the full-area conclusion. MLP on that split remains below both references.

## Per-class F1: previous versus longer training

All classes are retained, including declines.

| Class | Previous MLP | Longer MLP | Previous CNN | Longer CNN |
|---|---:|---:|---:|---:|
| Built-up Area (settlement) | 90.90 | 90.66 | 92.63 | 92.42 |
| Urban Parkland/Open Space | 38.26 | 37.46 | 32.61 | 36.01 |
| Estuarine Open Water | 98.93 | 98.94 | 98.66 | 98.77 |
| Short-rotation Cropland | 6.97 | 7.07 | 5.55 | 5.59 |
| Orchard, Vineyard or Other Perennial Crop | 11.95 | 11.76 | 9.53 | 12.72 |
| High Producing Exotic Grassland | 68.04 | 66.94 | 71.00 | 68.62 |
| Manuka and/or Kanuka | 66.15 | 66.16 | 62.60 | 65.38 |
| Broadleaved Indigenous Hardwoods | 9.16 | 8.65 | 4.59 | 9.89 |
| Indigenous Forest | 89.21 | 89.30 | 87.96 | 86.18 |
| Mangrove | 96.92 | 96.89 | 96.44 | 96.42 |
| Exotic Forest | 14.05 | 13.45 | 8.91 | 13.17 |

## Surface-height regression

Pooled errors; lower is better. DSM minus DEM includes vegetation and buildings.

| Feature | Mean-height MAE (m) | Mean-height RMSE (m) | P90-height MAE (m) | P90-height RMSE (m) |
|---|---:|---:|---:|---:|
| AEF | 2.236 | 3.336 | 3.002 | 4.341 |
| Public embedding | 2.570 | 3.770 | 3.373 | 4.822 |
| Original LINZ | 2.601 | 3.825 | 3.407 | 4.882 |
| Previous selected LINZ | 2.578 | 3.782 | 3.381 | 4.831 |
| Long-trained selected LINZ | 2.581 | 3.790 | 3.379 | 4.836 |

## Checks and interpretation

Each frozen route passed all 399 public-state tensor comparisons at all three checkpoints. This confirms exact freezing, not downstream improvement. All component loss curves, selected checkpoint aliases, validation stages and negative results remain in the data workspace.

This remains adaptive work on a previously inspected regional benchmark. Fold-zero validation labels participate as test labels in other folds; pooled figures are not a new blind evaluation. Results support a CNN benefit from the selected longer schedule, but neither universal improvement from more updates nor broad superiority to public/AEF. Further causal claims require targeted diagnostics rather than interpreting one split as general success.

AEF attribution: The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind (CC-BY 4.0).
