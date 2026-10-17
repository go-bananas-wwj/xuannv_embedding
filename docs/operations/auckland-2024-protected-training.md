# Auckland 2024: longer training with explicit public-feature protection

The user requested another parameter search and longer training to improve on both the
public model and AEF. Existing models and results remain unchanged. This is a bounded
second tuning round, with no Overleaf changes.

## Motivation

Extra input information does not guarantee better learned features under a fixed 64-D
bottleneck, finite optimization and weak targets. Earlier variants updated the public
backbone even when its learning rate was small. Their training loss improved more than
their downstream performance. This suggests testing interference and feature preservation;
it does not by itself establish the cause of the deficit.

The current strongest tuned high-resolution model has pooled MLP/CNN Macro F1 of
53.69%/51.86%, versus public 54.37%/53.53% and AEF 57.00%/56.11%. These are the targets
for improvement, not assumed outcomes of this round.

## Three bounded routes

All routes retain the same 320 tiles, 2024 twelve-month public sequence, static LINZ RGB,
WorldCover/OSM weak supervision and 64-D outputs. Effective batch is 48, seed 41, weight
decay 0.01 and warmup 100 updates. Two devices per route use micro batch 4 and accumulation 6.

| Route | Initialization | Public-backbone update | RGB/fusion LR | Public LR | Auxiliary-head LR | Anchor weight |
|---|---|---|---:|---:|---:|---:|
| long_continuation | Previous selected LINZ final | Trainable | 1e-4 | 1e-6 | 1e-5 | 0 |
| frozen_public | Trained public backbone/heads, fresh RGB layers | Exactly frozen | 3e-4 | Unused | 1e-5 | 0 |
| frozen_anchored | Same trained public initialization | Exactly frozen | 3e-4 | Unused | 1e-5 | 2 |

Each route may run for at most 2,400 additional successful updates, three times the prior
screening budget, equivalent to 360 full-area sample exposures. Check at 800, 1,600 and
2,400 updates using the original validation protocol. A two-stage failure to improve
validation stops the route; select its best validation checkpoint rather than its last.
Snapshots and exact optimizer/RNG continuation are retained between validation pauses.

Exact freezing excludes the public backbone and its reconstruction decoders from optimizer
groups. Auxiliary static/semantic heads remain trainable. At the first update and each
validation stage, compare every public state tensor against the registered initializer.
Any changed backbone tensor is an error. Fixed parameters alone do not imply an unchanged
output embedding, since the new branch still injects features between backbone blocks.

The anchor directly addresses output drift: normalize the student's annual mean and
penalize cosine distance to the existing public annual embedding on pixels with valid public
observations. The teacher array is tied by hash to its public checkpoint and geographic/time
index; mismatched patch identifiers, bounds, CRS, region, months or dimensions are rejected.
Targets are loaded without modifying the original caches. The teacher is detached and adds
no second live encoder. It uses no LCDB or LiDAR reference labels.

An anchor weight of 2 is a regularization parameter, not a proof of task improvement. Its
total loss has a different definition from unanchored routes; use component curves and
identical downstream validation scores for comparisons.

## Selection, fallback and reporting

At each stage fit the same MLP and CNN on the same fold-zero training points, 200/class.
Use their mean validation Macro F1 for stage/candidate selection. Retain both the prior
selected high-resolution model and original public model as eligible validation baselines.
If public wins, report that result without labeling it a successful high-resolution model.
Validation fallback does not guarantee non-degradation on unseen labels or superiority to AEF.

Only after current-round selection, evaluate the chosen model's test results, remaining
spatial folds, regression and fixed scenes. This is adaptive work on an already inspected
regional benchmark, not a new blind test; pooled folds reuse some labels involved in
validation selection. Preserve fold-zero results and every negative class/task result.

The local supervisor checks every 30 seconds, enforces a 100 GiB reserve, limits recovery
attempts and honors a STOP file. The language model does not poll continuously. Stage
validation milestones and completion are automatically published to GitHub. Runtime weights,
teacher arrays, exports, logs and plots stay in the data workspace. Peak additional artifacts
are expected to remain well below the existing 150 GiB preparation budget.
