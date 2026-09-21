# Paired spatial uncertainty for fixed predictions

`downstream.multitask_bootstrap` supplies the AP and RMSE numerical layer for the final
classification, retrieval and coverage-regression comparison. It does not fit a readout,
choose a checkpoint, open a dataset or authorize test evaluation. Existing validation
experiments continue to use their pinned code and selection protocol.

Call `tile_weights(tile_ids)` once for the ordered evaluation tile list. The registered
default is 2,000 equally likely tile draws with replacement, seed 20260921. Reuse exactly
these multiplicities for every method, training seed, support seed and task condition.
Retain tiles without eligible observations in the common tile list.

For each fixed task/budget/readout condition, `seed_metric_draws` takes predictions shaped
`[training_seed, support_seed, observation]`, the common retained truth and each observation's
position in the ordered tile list. The caller must first verify the same geographic domain,
reference validity and frozen validation-selected readout choices. No thresholds or other
parameters may be selected inside a bootstrap draw.

AP pools ranked observations across the sampled tiles and handles tied scores jointly. RMSE
pools squared errors and counts before taking the square root. Neither is an average of
per-tile metrics. `paired_seed_summary` then averages **metrics** across training and support
seeds within each draw and reports candidate minus baseline. It does not ensemble predictions.
Positive AP differences and negative RMSE differences favor the candidate. A published fixed
embedding may have one upstream training realization; the methods need not have equal counts
of training seeds, but their support-seed identities must match.

Pairing checks cover ordered tile IDs, truth and observation-to-tile mapping hashes, exact
draw schedule, metric and support seeds. Arrays preserve undefined values: AP is undefined
without sampled positives; RMSE is undefined without sampled observations. No `nanmean`
silently drops seeds. If any paired draw is undefined, the unconditional 95% percentile
interval is withheld and the defined/total counts are returned. Such a result cannot support
a positive-confidence claim. The undefined-case policy must remain fixed before scoring.

This layer does **not** yet provide the final file-backed evaluator, validation-frozen readout
artifact checks, annual/monthly feature adapter, strong-head comparisons, task-family
aggregation or a paper-level pass decision. Those remain separate required integration work.
Its tests use synthetic predictions and literal tile duplication, including unequal tile
sizes, tied AP scores, missing classes/domains, seed averaging and mismatched pairing.
