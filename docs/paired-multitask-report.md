# Spatial uncertainty from archived primary predictions

After the locked primary scoring phase, run:

```bash
xuannv experiment summarize-primary --spec SPEC.json \
  --test-identity-sha256 DIGEST --output NEW_REPORT_DIRECTORY --threads 2
```

The command uses 2,000 paired tile bootstrap draws with seed 20260921 and one to four CPU
threads. It requires the registered G5 test identity and complete test result, the unchanged
evaluation lock, common-domain artifacts and calibration identity, and all saved prediction
hashes. It opens archived predictions, not new source features or labels, and never refits
or selects a readout. The original scoring artifacts remain unchanged. Existing report
directories are rejected; progress and failures remain recorded.

## Fixed aggregation

All methods, model realizations, support seeds and conditions share the same ordered tile
list and resampling multiplicities, including tiles with no eligible observations. Within
each draw, compute pooled AP or RMSE separately for each model realization and support seed.
Average these metrics, never the predictions. Method groups preserve the actual realization
names; ordinal IDs passed to the numeric routine do not infer random seed values. Real
training-seed provenance and the common-parent limitation require the separate training audit.

The report retains every registered task/budget condition and each realization/support
point metric. Missing, duplicate or changed conditions fail. For each condition, the saved
point metric is independently recomputed from archived predictions and must match within
1e-12. Truth, tile mappings and valid positions must agree between methods, support seeds,
and budgets. All ordered method pairs are reported, including unfavorable comparisons.

For raw family metrics, average support and model-realization metrics per task/budget.
Average tasks and budgets equally within each source. Classification then gives OSM and
ESRI one half each; regression averages the six ESRI tasks; retrieval averages the four OSM
tasks. This prevents six ESRI classification tasks from outweighing the four OSM tasks.
Compute the same family means inside every common draw before calculating paired differences
and percentile intervals. Do not average already-computed confidence limits.

Positive improvement means candidate AP minus baseline AP for C/Q, and baseline RMSE minus
candidate RMSE for R. Relative R improvement is the paired difference divided by baseline
family macro RMSE inside that draw; a zero reference is explicitly undefined. Report both
absolute and relative R changes. The numerical primary rule requires all three observed
family improvements to be nonnegative, and at least one family to meet its practical
threshold (C/Q AP +0.01, R relative RMSE reduction 0.02) with its 95% improvement interval
strictly above zero. This checks the registered primary numerical criteria only. It does
not certify a complete paper claim, independent seeds, external geography, strong heads,
monthly reconstruction or robustness.

The normalized error-gain score remains separate from these raw-unit criteria. For each
task/budget/support condition, first average the model-realization error, then compute
`(reference_error - candidate_error) / max(reference_error, 1e-6)`. Average these gains over
support seeds, tasks/budgets, source-balanced C and finally the three equally weighted
families. Normalizing only after pooling all support errors would change the registered
condition weights. AP error is 1-AP and R error is RMSE. The report counts near-zero
reference conditions/draws and records the floor. The B0-referenced comparison supplies
the registered reference score; other pairs are explicitly named as gains relative to
their own reference and are not silently called B0 scores. No training selection changes.

No `nanmean` drops an undefined condition, support realization or bootstrap draw. If any
draw is undefined, its corresponding unconditional 95% interval is withheld and the number
of defined draws is reported. An affected interval cannot support the positive-confidence
criterion. Other fully defined families remain separately interpretable. Undefined point
metrics also prevent the all-three-nonnegative criterion from passing.

## Outputs and verification

The report stores the tile schedule, individual `[model realization, support seed, draw]`
metric arrays, the corresponding point metrics, per-method/task/budget/support arrays,
all pairwise family/condition summaries, normalized scores, implementation and numerical
runtime identities, and file hashes. These arrays can be independently replayed without
fitting or rereading source observations. The Python API permits small synthetic draw
counts for tests and marks these as a nonregistered schedule; the CLI exposes no such
override.

Tests cover source balance, within-draw family aggregation, practical thresholds,
degraded-family rejection, undefined draws, zero reference RMSE, preserved support-wise
normalization, invalid/missing conditions, one-versus-multiple model realizations,
recomputed point metrics, unchanged score artifacts, unavailable source test files,
prediction tampering, differing positions despite revised hashes, and CLI dispatch.
This is synthetic workflow validation; actual AEF and real test comparisons follow the
registered candidate lock and remaining data, strong-head and training audits.
