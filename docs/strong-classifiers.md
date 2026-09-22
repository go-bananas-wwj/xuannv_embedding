# Frozen classical classification readouts

`downstream.strong_classifiers` fits RF, RBF-SVM and cosine kNN on explicit training support,
selects SVM C and a classification threshold only on validation data, and saves a reloadable
readout. It does not choose support tiles, read datasets, lock candidates or replace the
paired final controller. The companion `neural_readouts` module supplies MLP/convolution
calibration and replay. The shared strong-head file workflow remains required. Existing
pinned training/evaluation experiments are unaffected.

The caller must provide the common spatial/validity domain and already paired class-balanced
training positions from the G5 contract, in registered order. RF and SVM accept no more than
4096 support pixels per class. kNN retains the first 1024 positions of each class from that
already randomized sequence. This preserves nested internal support rather than drawing new
positions at test time. Different caps describe the registered algorithms, not different
support budgets between compared embedding methods using the same head. Preserve the saved
relative input positions so the file controller can bind them to original spatial positions.

StandardScaler fits only the actual retained training support, using float64 arithmetic;
it never sees validation or test features during fitting. Keep original feature dimensions.
RF uses 200 trees, square-root feature sampling, minimum leaf size 2, balanced class weights
and the registered support seed. One tree-prediction worker keeps probability accumulation
deterministic. SVM uses an RBF kernel with `gamma=scale`, balanced classes, and C candidates
0.1, 1, 10. Validation AP selects C, ties within 1e-12 retain the smaller C. The existing
float64/NumExpr RBF decision implementation is reused and checked against scikit-learn
decision functions to absolute tolerance 1e-10 on synthetic data.

kNN averages labels of the five highest-cosine training neighbors after training-support
standardization and unit normalization. If fewer than five support observations exist, it
uses all of them. Equal similarities choose earlier registered support positions; a zero
query follows this same explicit tie rule. This is a newly fixed standardized comparison,
not an assertion of numerical equivalence to historical unstandardized torch top-k probes.
Historical scores and the new common-protocol results must stay in separate tables.

Each readout selects its validation-F1 threshold and records all candidate APs, fit/query
times, retained pixel counts, support/validation hashes, prediction hash and numerical
runtime identity. Prediction takes features only and processes fixed batches of 1024 with
bounded CPU threads; no labels or fitting calls enter prediction. Empty query arrays return
empty scores. `verify_validation` requires exact prediction replay after loading, without
refitting or silently tolerating a changed validation domain.

`save_classifier` writes numeric scaler/support/label/position arrays, and, for RF/SVM, a
joblib estimator. `load_classifier` requires the previously registered identity SHA-256,
checks exact producer source/runtime and every payload digest before deserializing, and
validates shapes and estimator family. Load only trusted artifacts produced by this
experiment: joblib is pickle-based, and an arbitrary supplied hash is not a trust boundary
for third-party model files. Use the producer snapshot and environment for replay. Existing
directories are rejected rather than overwritten.

Tests verify persisted RF/SVM/kNN predictions, support-only scaling, independent cosine
neighbor ranking, selected SVM regularization, deterministic ties, kNN pixel caps, empty
queries, forbidden refitting, payload checks before joblib loading, changed runtime/data,
and invalid inputs. These are synthetic software checks, not a strong-baseline accuracy
result. Shared file-backed strong-head calibration/scoring and the actual locked-candidate
comparison still need integration and execution.
