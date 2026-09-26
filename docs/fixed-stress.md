# Fixed-readout missing-input evaluation

`xuannv experiment score-fixed-stress --spec stress.json` evaluates a registered set of
high-resolution input retention curves using the C/R/Q readouts from a completed, locked
primary comparison. It never fits a scaler, coefficient, threshold or retrieval prototype.
The baseline stage is `calibration` for validation diagnostics or `test` for held-out scoring.
This path requires a locked primary contract even for validation; provisional development
diagnostics should not be presented as final held-out evidence.

The JSON specification has exactly these fields:

- `protocol`: `fixed-readout-highres-stress-v1`.
- `base_spec`: `{path, sha256}` for the locked primary specification.
- `calibration_identity_sha256`: hash of the primary calibration identity.
- `baseline_identity_sha256`: hash of the original scored split's identity.
- `split`: `validation` or `test`.
- `sources`: sorted names of the retained high-resolution sources.
- `fractions`: unique descending fractions including 1 and 0.
- `mask_seed`: nonnegative integer matching the inference exports.
- `variants`: named objects `{model, fraction, features}`. `model` references a primary
  model; `features` uses the same explicit manifest/cache/tile/selection contract as primary
  evaluation. Every included model must have exactly one variant at every fraction.
- `output`: a new directory.

Exports must use the same checkpoint, configuration, training-code identity, cache, feature
period, layout and adaptation metadata as the original model. Only the registered
high-resolution retention may change; prefix masking and additional source drops are rejected.
The exporter records the shared deterministic, nested mask rule described in
[highres-retention.md](highres-retention.md). The full-retention endpoint must reproduce
the original feature values and frozen predictions exactly. These checks do not themselves
audit raw-image georeferencing or prove the exporter implementation: run the separate raw
input mask gate and the original geographic audit.

Use `experiment export --export-split validation` or `--export-split test` when only that
partition is needed. The manifest retains the complete original grid and split definitions,
and records `exported_indices` and `exported_splits`. Unmaterialized record paths are explicit
future locations, not existing artifacts. The feature reader rejects requests outside the
materialized partitions before opening tile arrays. No other partition's cached samples are
hashed or decoded by a partial export, and automatic legacy probing is disabled. Omitting
the argument preserves full-export behavior. This avoids duplicating training and buffer
embeddings for every artificial mask. An actual partial full-retention export must still
pass the numerical identity control before interpreting a robustness curve.

Each model's exported validity mask must be unchanged. Evaluation reuses the original
comparison's common valid domain, truths, query order and regression blocks. It fails instead
of shrinking the domain to hide newly missing features. All saved prediction files are checked
against the original truth, tile IDs and query positions. Their hashes, unchanged readout
identities, full per-condition metrics and baseline metrics remain auditable. `fraction=0`
means removing all originally valid selected high-resolution inputs, not deleting evaluation
targets. The full curve is a paired observation-removal experiment; it is not a separately
trained no-source ablation or a claim about natural cloud distributions.

Tests exercise all three task families on validation and held-out splits, forbid every
fitting path, remove held-out files during validation, and reject changed checkpoints,
caches, mask seeds, validity domains, full-retention values and missing endpoint controls.
