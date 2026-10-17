# Auckland west 2024: interim results, 2026-10-17 08:54 Beijing time

Public-only training and all its registered readouts are complete. The LINZ encoder is still training; no final LINZ-embedding scores are available. These results do not establish an advantage over AEF.

Both encoders use the same registered 5,334-update budget and effective batch 48. The completed encoder uses all 320 tiles. Readout test predictions are pooled over five spatial folds, so every tile receives one held-out label-evaluation role. Encoder exposure to the whole area is transductive.

## LCDB v6 2023/24 classification

Eleven classes, 200 registered training-label points per class, identical support locations and valid area. Readout selection uses validation only. Embedding results below use the normalized twelve-month mean. Values are percentages.

| Feature | Readout | Macro F1 | mIoU |
|---|---|---:|---:|
| AEF 2024 | Linear | 56.64 | 47.65 |
| AEF 2024 | MLP | 57.00 | 48.03 |
| AEF 2024 | CNN | 56.11 | 47.34 |
| Public three-source embedding | Linear | 50.23 | 43.10 |
| Public three-source embedding | MLP | 54.37 | 46.32 |
| Public three-source embedding | CNN | 53.53 | 45.32 |
| Public temporal statistics | Random forest | 51.76 | 42.37 |
| Public temporal statistics | U-Net | 46.53 | 37.61 |
| Public statistics + aerial statistics | Random forest | 52.47 | 43.01 |
| Public statistics + aerial statistics | U-Net | 47.11 | 37.99 |

The public embedding trails AEF by 6.41 / 2.63 / 2.58 F1 percentage points for the linear / MLP / CNN readouts respectively. RF and U-Net consume temporal statistics; the aerial-statistics baseline adds local RGB mean, standard deviation and coverage. These are separate practical baselines, not matched encoder architectures.

## LiDAR 2024 surface-height regression

Ridge regression, the same 5,000 labeled training points per fold. Errors below are pooled over held-out pixels. Heights describe trees and buildings together, not buildings alone. All error values are metres.

| Feature | Mean-height MAE | Mean-height RMSE | P90-height MAE | P90-height RMSE |
|---|---:|---:|---:|---:|
| AEF 2024 | 2.236 | 3.336 | 3.002 | 4.341 |
| Public embedding | 2.570 | 3.770 | 3.373 | 4.822 |
| Public statistics | 2.890 | 5.143 | 3.754 | 6.453 |
| Public + aerial statistics | 2.883 | 4.966 | 3.729 | 6.388 |

The public embedding improves on the raw-statistics baselines but has larger errors than AEF. No test-driven encoder or training-budget changes have been made.

## Execution

160 of 215 classification readout settings are complete. The remaining 55 require the LINZ encoder. Watchdog heartbeat is healthy; both production restart counts remain zero. LINZ training is estimated to finish around 13:20 Beijing time, followed by export and evaluation.

AEF attribution: The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind (CC-BY 4.0). No Overleaf changes in this phase.
