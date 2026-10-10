# Monthly versus static-mean reconstruction

`xuannv experiment reconstruct-ridge` accepts `--representation monthly|mean`
and `--normalize-embedding`. Defaults preserve the historical monthly, unaltered
feature extraction. The selected representation and normalization are recorded in
run identity and used for both training support and validation queries.

`mean` averages the model's monthly outputs only after the target observations and
availability have been removed and the masked inputs have been re-encoded. It
never reuses a mean computed from the complete unmasked sequence. Prefix mode
removes future observations before either representation is computed.

Use the same normalization flag and common Ridge settings for paired comparisons.
With normalization enabled, both outputs receive the same per-pixel L2 operation;
a zero vector remains zero. Mean pooling includes all output time indices, even in
prefix mode, but none of these outputs may receive hidden future observations.
This remains retrospective inference through an already pretrained encoder, not
proof of causal training or out-of-time prediction.

The decoder has the same feature dimension for both representations. Input masking
identities must match within each model/month pair. Numerical gates do not prove
that monthly features improve reconstruction; actual shared-domain errors and the
simple time baseline are still required.
