# Auckland 2024: high-resolution diagnosis and bounded tuning

The user requested fitting diagnostics and parameter tuning on 2026-10-17. This stage
preserves the prior paired models and keeps LCDB/LiDAR outside encoder training. No
Overleaf edits are included.

## What the existing checkpoints show

The original run overwrote a progress JSON rather than retaining a complete loss history.
There is no original step-by-step curve to recover. The following values were recomputed
on five fixed, unmasked input tiles at initialization and the retained checkpoints. They
are diagnostic objective snapshots on encoder-exposed samples, not held-out convergence
or overfitting measurements.

| Scheduled update | Public total loss | LINZ total loss |
|---|---:|---:|
| 0 | 2.9233 | 2.9233 |
| 400 | 2.3187 | 2.3116 |
| 1,334 | 1.4786 | 1.4438 |
| 5,334 | 0.8763 | 0.8365 |

LINZ semantic-probe loss decreases from 2.0665 to 1.0989; its static WorldCover loss
decreases from 2.3943 to 0.5311. The corresponding public final values are 1.1943 and
0.5309. There is clear fitting of the current objectives. A lower objective alongside
worse downstream performance motivates testing initialization and backbone interference,
rather than assuming missing gradient updates or simply increasing model capacity.
The checkpoint spacing does not establish whether the final local loss slope is flat.

AEF shares the public Sentinel/Landsat observation families but not the same complete
training setup. Its training uses additional reconstruction/text targets and globally
sampled sequences. Official sources:
[paper, data and training sections](https://arxiv.org/html/2507.22291v2),
[production dataset description](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL).
This study must not describe the two pretraining datasets as identical.

## Registered three-candidate screen

Keep architecture, public/aerial inputs, weak supervision, effective batch 48 and seed 41
fixed. Each candidate gets 800 additional successful updates, 100-update warmup, cosine
decay and weight decay 0.01. These are continuation/adaptation runs, not three additional
from-scratch models. Public-backbone protection uses a small positive learning rate;
it is not an exact parameter freeze.

| Candidate | Initialization | Public LR | RGB/fusion LR | Auxiliary/decoder LR | NPU devices |
|---|---|---:|---:|---:|---|
| continue_low_lr | Original LINZ final weights | 1e-5 | 5e-5 | 1e-5 | 1, 2 |
| protect_public | Trained public backbone and heads, fresh RGB/fusion layers | 1e-6 | 1e-4 | 1e-5 | 3, 4 |
| faster_fusion | Same trained public initialization | 3e-6 | 3e-4 | 1e-5 | 5, 6 |

The public initializer is a complete explicit state conversion: every public weight and
auxiliary head is loaded strictly, fresh high-resolution layers are preserved, and parent
hashes are recorded outside Git. It is marked as an initializer rather than a resumable
training checkpoint. Existing trained model files are not overwritten.

Each two-card group uses micro batch 4 and six accumulation passes. A real-data first
update is saved and validated before exact-state continuation. A supervisor checks locally
every 30 seconds, permits at most two recovery attempts per candidate, requires a valid
checkpoint after an interruption, and stops on a manual STOP file or 100 GiB storage reserve.
It owns only the candidate process groups; unrelated device services are untouched.

## Validation and final reporting

Use the original fold-zero training/validation partitions and the same 200 points/class.
Fit the same MLP and CNN readouts on training points. Select the candidate by the mean
of their validation Macro F1, including the original LINZ model as a retainable parent.
Current-round test metrics are not used for this selection. If no candidate improves the
validation criterion, retain the parent and report the unsuccessful tuning.

After selection, reuse the fold-zero validation-trained heads for its test predictions.
The selected candidate also receives the remaining spatial folds, height regression and
the original fixed-scene figures. Because the benchmark was already inspected and fold-zero
validation tiles also appear as test tiles in other folds, pooled five-fold results are
adaptive regional exploration, not a new blind test. Keep the fold-zero result identifiable.

New training appends successful-update component losses, group learning rates, elapsed time
and AMP skip counts every 20 updates. Values are averaged across accumulation passes and
distributed ranks, with missing optional semantic statistics aggregated by key. Checkpoint
and sample artifacts stay outside Git; protocol, code and compact milestones are published.
