# Auckland west 2024: approved paired experiment

The user approved execution on 2026-10-16 and requested GitHub synchronization after each
verified implementation step. Runtime data, credentials, absolute private paths, weights,
embeddings and logs remain outside Git. Work is published on `codex/retention-nz-2024`.

## Training

Train two independent regional models from scratch, using all 320 tiles and the twelve
months of 2024. The public control consumes Sentinel-2, Sentinel-1 and Landsat. The full
model additionally consumes one LINZ 2024 aerial RGB observation at nominal 3 m. Both use
identical public initialization, optimizer budget, WorldCover and OSM weak supervision.
Static aerial tokens are encoded once, marked undated, and shared across monthly queries;
they are not twelve observations. Outputs are 64-dimensional on a 10 m grid.

The fixed budget is 5,334 optimizer updates at effective batch 48 (approximately 800
full-region sample exposures), with seed 41. Each model is trained once. Checkpoint
selection and readout tuning must not use test scores. Preserve a temporary matched
checkpoint near 200 sample exposures for an early preview.

## Downstream evaluation

Freeze five spatial folds before model scoring. A 2x2-tile block stays in one group.
Each fold has three training groups, one validation group and one test group. Test
membership across folds covers all 320 tiles exactly once. Exclude 100 m patch borders
from sampled readout labels and do not cross patch boundaries with readout convolutions.
Encoder exposure to the full target area is explicitly transductive; downstream labels
are held out spatially. This is not evaluation on an area unseen by the encoder.

Primary targets are LCDB v6 2023/24 land cover and LiDAR 2024 mean/p90 surface height.
LCDB classification uses its prepared 20 m interior mask. Forest, agriculture/grass and
wetland/coastal per-class metrics are breakdowns of this same land-cover dataset.
Class eligibility depends on spatial support and label counts, never model performance.
DEM elevation/slope are supplemental. LINZ 2017/2022 buildings are historical diagnostics,
not 2024 building truth. Evaluation references are not encoder inputs or loss targets.

Compare official AEF 2024, public embeddings and full embeddings with the same linear,
MLP and CNN readouts, label locations, folds and valid area. Annual normalized mean is
the fixed primary representation; April and March-May means are secondary comparisons.
Include raw-public and raw-public-plus-aerial random forests and U-Nets. Classification
label budgets are 50, 200 and 1,000 points per eligible class, spatially distributed.
CNN/U-Net losses see only the registered label points, not additional dense labels.

Report macro F1, mIoU, per-class precision/recall/F1 and confusion matrices. Regression
uses MAE, RMSE and R2; DSM minus DEM describes both trees and buildings. Preserve all
eligible task results, including declines. Do not pool incompatible metric units.

## Execution and checkpoints

Prepare caches with `xuannv experiment paired-region --highres-config FULL.yaml
--output TRIAL --workers 4`. Run exact-spec validation and real-input NPU gates before
training. The public spec registers the full untrained config as its common initializer.
Use two three-card groups; leave an existing device service untouched. Calibrate micro
batch size while keeping effective batch equal. Register the actual Git commit in every
training checkpoint. Step reports are local; report stage transitions instead of polling
the language model. Export, score and render only registered fixed-budget checkpoints.

No Overleaf changes are authorized in this experiment phase. GitHub receives implementation,
protocol and compact non-sensitive stage summaries. No improvement is assumed in advance.
