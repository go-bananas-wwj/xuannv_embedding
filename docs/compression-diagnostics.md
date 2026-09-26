# Compression task diagnostics

`xuannv experiment diagnose-compression --spec spec.json` fits PCA only on common
training positions, then evaluates native and fixed PCA dimensions through the
existing classification and positive-prototype retrieval readouts. It does not
open held-out test features or labels, promote a candidate, or choose a dimension.

The specification uses protocol `compression-validation-v1`, registered
`reference_cache`, `models` feature contracts, and exactly two `labels` bundles
(`train`, `validation`). It also fixes `dimensions`, classification `budgets`,
`retrieval_budgets`, `support_seeds`, `sample_step`, `sample_offset`, and `output`.
Feature dimensions are retained explicitly; projections larger than any source
dimension are rejected. Standardization for classification is fitted only on its
training support, while validation chooses the registered Ridge regularization.

The native representation is a separate control even when the largest PCA output
has the same dimension. Centering and rotation can change cosine retrieval and
subsequent per-channel scaling without discarding any principal component.

PCA parameters, frozen readouts, support identities, predictions and point AP are
saved per model/dimension. Intermediate projected maps are temporary, while
original registered feature arrays and archived predictions remain available for
verification. All registered tasks, budgets and support draws are evaluated;
undefined AP is preserved instead of silently dropping a task.

These are development diagnostics, not independent test accuracy. A final test
comparison must reuse the fixed training projection and calibrated readouts after
the candidate/evaluation lock. Explained variance does not replace task scores.
