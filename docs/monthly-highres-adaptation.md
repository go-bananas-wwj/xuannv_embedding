# Monthly high-resolution adaptation

The experimental adapter supports `data.monthly_highres: true`. The default is
`false`, preserving the archived static-input behavior and parameter names.
Monthly mode requires registered incremental adaptation from a public-observation
base. It does not change the base checkpoint or claim finer output resolution.

Monthly samples carry high-resolution images `[T,C,H,W]`, input masks
`[T,1,H,W]`, and separate immutable `highres_quality_masks`. Only observations
within the same configured month are composited. Missing months remain masked.
The residual encoder processes each month independently using shared weights;
missing pixels/months return the corresponding incoming embedding exactly.
High-resolution reconstruction targets use valid-weight-normalized resampling.
Public-source preprocessing is retained for compatibility with the fixed base.

Training month dropout hides the same months in public and high-resolution
inputs. Monthly high-resolution spatial visibility is mapped from the public
mask over the common geographic extent; it is not independently sampled per
resolution. Targets and quality masks remain unchanged. Input visibility masks
are cleared for hidden high-resolution pixels even when the legacy public
availability policy retains availability. Geographic alignment must be audited
before building the immutable cache. This option does not implement arbitrary
GSD conditioning or an alignment-tolerant loss.

## One model across multiple devices

`xuannv experiment run` supports `torchrun` with HCCL (NPU) or Gloo (`--device cpu`
for regression tests). Configuration `data.batch_size` is **per rank**. Effective
batch is batch size × world size × accumulation; the last accumulation window can
be smaller. Training uses an epoch-seeded distributed sampler, with padding count
recorded. Validation uses non-overlapping rank-strided samples without padding.
Metrics aggregate sample-weighted local losses; batch uniformity is computed
within each rank, not across a gathered global batch.

All ranks synchronize trainable gradients; only rank zero publishes registration,
logs and checkpoints. Checkpoints retain per-rank random/scaler states and require
the same world size on resume. `best.pt` is chosen by the globally aggregated
unmasked validation loss. Device throughput and peak memory must be measured
before choosing batch size; higher occupancy alone is not a speed criterion.

Export is single-device and reconstructs the registered adapter from the parent
checkpoint and configuration. Old static caches cannot be used for monthly runs.
Old caches without `monthly_highres` are interpreted as static, not silently
upgraded. Production data, weights, diagnostics and experiment configuration
artifacts remain outside Git.

A source can declare explicit acquisition-date to representative-month assignments,
for example `data.highres_month_assignments.highres_optical` with the quoted key
`"2026-04-30"` mapped to `"2026-05"`. This changes the model period assignment;
it never renames the raw files or asserts a different acquisition date. Assignments
are validated against configured high-resolution sources and periods and included
in the immutable cache/configuration hashes. Without an assignment the actual
acquisition month is used. Dataset-specific assignments belong in run configuration.
