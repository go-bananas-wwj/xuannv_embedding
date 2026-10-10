# Same-month high-resolution late-fusion baseline

`xuannv experiment export-late-fusion --spec spec.json` appends fixed statistics
to an existing public embedding without training an encoder or reading labels.
This is a low-cost comparison, not a substitute for a matched learned fusion
ablation.

The strict JSON specification has `protocol: monthly-late-fusion-v1`, a registered
input `cache` (`path`, `sha256`), a registered `base`, ordered high-resolution
`sources`, a selected `month`, explicit `splits`, and a new `output` directory.
`base` contains `manifest_path`, `manifest_sha256`, `cache_path`, `cache_sha256`,
`tile_sha256`, and `channels`. Both cache layouts and observation periods must
match. Native inputs must already have an independently audited common footprint.

Each source contributes per-channel area-weighted means and population standard
deviations, then its valid-area fraction. Only the selected month's forward input
frames and masks enter these statistics. Label, loss-only masks and target fields
are not read. Noninteger scale ratios use exact rectangular overlaps; moments are
centered before variance calculation. Invalid observations contribute zero area.

For a 64-dimensional base with three optical and one SAR channel, the output has
74 dimensions. Missing high-resolution sources add zero statistics and zero
availability. Feature validity follows the base, so missing high-resolution data
does not erase otherwise usable public features. Invalid base pixels are masked.

The output is a registered single-month NPZ embedding with explicit timestamps,
original channel dimension, validity mask and provenance. The public component can
still include observations from its whole time window; this baseline makes no
causal forecasting claim. Downstream standardization is fitted on training support
under the common readout protocol, including the explicit availability channels.

Selected-split export must not load test or buffer tiles unless requested. The
verification output records file identities and input selection, but is not a
performance result or an independent physical registration audit.
