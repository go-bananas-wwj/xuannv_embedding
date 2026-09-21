# Frozen validation choices for final multi-task prediction

`downstream.frozen_readouts` separates calibration from prediction for the registered linear
classification/coverage-regression readouts and cosine-prototype retrieval. Calibration takes
only explicit training support and validation arrays. The final caller must verify their
split, common valid domain and registered support positions before passing them here.

`fit_classification` fits the scaler on training support only and selects Ridge regularization
by validation AP. `fit_regression` uses validation RMSE of predictions clipped to [0, 1]. Both
default to candidates 10, 1, 0.1, preserving the existing 1e-12 tie rule favoring stronger
regularization. Classification stores its validation-F1 threshold. Support/validation data
and selected validation prediction hashes, all trials and library versions are recorded.

The frozen object stores numerical scaler, coefficient and intercept arrays. It preserves the
original float32/float64 calculation path, including StandardScaler's two float32 in-place
casts and the estimator's original coefficient shape. Calibration fails unless the frozen
parameters reproduce the selected validation predictions exactly on the current runtime.
Binary Ridge coefficient layouts can be one- or two-dimensional; neither is silently
reshaped at serialization time.

`freeze_retrieval` stores nonzero training prototypes. Pass `normalized=True` for already
normalized outputs of the existing registered training-component selector; this verifies
unit norms without normalizing a second time. Retrieval prediction uses the existing
maximum cosine similarity and zero-query policy. It has no validation-selected parameters.

`save_readout` creates a new directory containing a JSON identity and a numeric NPZ, never
an executable pickle. `load_readout` requires the caller's previously frozen identity hash,
checks the parameter and implementation hashes, and validates array shapes/dimensions.
Use the producer's pinned source snapshot to load its readouts. Existing output directories
are rejected. `verify_validation` then replays the exact recorded C/R validation domain and
prediction hash after loading, with no refitting. Query prediction accepts features only;
it cannot use query labels to change regularization, a scaler or a threshold.

This implements frozen parameter storage and replay, not the full final evaluation controller.
The final caller still needs to bind G3 feature identities, geographic/common validity audits,
support selections, the candidate lock, strong heads, label scoring and G2 task-family
uncertainty. Actual AEF and held-out comparisons remain after the candidate lock. Existing
running experiments continue to use their original immutable code snapshots.

Tests reproduce the existing C/R/Q validation evaluator's synthetic predictions exactly,
including persisted reloads, support-only scaling, regression clipping, stronger-alpha ties,
query invariance, tamper rejection and replay failure when validation observations change.
