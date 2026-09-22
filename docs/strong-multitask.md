# Paired five-head classification workflow

`xuannv experiment calibrate-strong` and `score-strong` connect the frozen RF, RBF-SVM,
cosine kNN, MLP and convolution readouts to the primary evaluation's locked cohort. The
full four-OSM/six-ESRI classification matrix, all registered support seeds, and both
registered tile budgets are evaluated with every head and every model realization. No
best-head-only selection occurs. These are classification comparisons, not replacements
for the primary C/R/Q, reconstruction, source-ablation or final multi-seed requirements.

A strong specification has exactly these fields:

```json
{
  "protocol": "multitask-final-strong-v1",
  "primary_spec": {"path": "/path/primary.json", "sha256": "REGISTERED_SHA256"},
  "heads": ["rf", "svm", "knn", "mlp", "conv3x3"],
  "neural_device": "cpu",
  "output": "/path/new-strong-output",
  "lock": {"path": "/path/strong-lock.json", "sha256": "REGISTERED_SHA256"}
}
```

The primary spec must itself pass its existing candidate contract and upstream geographic
evidence checks. The additional lock must have `state: locked` and `contract_sha256` equal
to `strong_multitask.contract_sha256(spec)` (canonical JSON excluding output and lock).
This binds the primary spec digest, all five heads, and neural backend choice. Use `cpu`
or an explicit `npu:0`-style index in the launcher's reserved visible device set; no device
allocation or CANN environment change is performed here. The strong output must differ
from the primary output. Never create an actual final lock before completing the registered
training selection. Synthetic fixture locks prove software behavior only.

```bash
xuannv experiment calibrate-strong --spec /path/strong.json
xuannv experiment score-strong --spec /path/strong.json \
  --calibration-identity-sha256 REGISTERED_CALIBRATION_IDENTITY_SHA256
```

Calibration reads train/validation source data only. Feature dimensions and annual/monthly
period metadata are preserved by the shared feature adapter. All models share the
intersection of feature-valid pixels, including surrounding convolution context, as well
as task-specific label validity. They use the same nested full support tiles. RF/SVM use
at most 4096 pixels per class from the fixed balanced ordering; kNN retains the first 1024
per class from that same ordering. Neural heads use all labeled valid pixels in the same
support tiles, with their separately registered fixed update budget. `support.json` binds
canonical tile order, target hash and pixel counts; `positions.npy` records exact retained
positions within that order. Different within-head sample caps never justify giving two
embedding methods different support under the same head.

Every trained readout is saved, reloaded and required to exactly replay its validation
predictions before the calibration record completes. Thresholds and SVM C are selected on
validation only; neural weights use the fixed last step. Prediction artifacts retain
scores, truth, canonical tile IDs and valid pixel positions. AP, F1, IoU, balanced accuracy
and boundary F1 are recorded for every condition. A missing held-out truth domain remains
present with undefined metrics; it is neither assigned zero nor dropped.

Scoring verifies the complete calibration contract, implementations, every saved model
payload, support record, retained position array and validation prediction before opening
source test data. It then loads each frozen readout as needed and never refits or chooses
parameters on test data. Preflight checks models one at a time to avoid retaining all
forests and convolution heads in memory. Failed stages retain a failure record and refuse
silent overwrite; use a separately registered attempt when recovery is needed. Failed
scoring explicitly records whether test data access began.

This adds executable common-domain strong-head calibration and scoring. The companion
`strong_multitask_report` supplies shared tile resampling, per-realization metric averaging,
all-head AP intervals and undefined-case handling; see `strong-uncertainty.md`. Actual
split-label bundle preparation, external CRS auditing, recipe selection/multiple training
seeds and final comparisons remain required. Historical unstandardized probes must remain
separate from these standardized/masked results.
