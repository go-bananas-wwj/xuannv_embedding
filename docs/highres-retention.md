# Controlled partial high-resolution availability

The experiment exporter supports `--retain-highres-source SOURCE ...`,
`--highres-retention FRACTION` and `--retention-seed SEED`. They must be registered
together before evaluation; the source must be high-resolution and cannot also be
listed in `--drop-source`. Automatic probe fitting is disabled for these exports.

For each tile/source, a stable hash selects one of four edges. In each month,
originally valid pixels are ordered in an edge-anchored raster scan and the first
`floor(fraction * valid_count)` are retained. This gives nested masks and a spatial
prefix with at most a partial boundary row/column, rather than randomly scattered
pixel deletion. Existing gaps are preserved. The edge is shared across months;
the cut location can change with the actual valid footprint.

Values and input masks are removed together. Original batches, target values,
target masks and other sources remain unchanged. All-missing sources stay missing;
the global random state is untouched. Fraction one is the unaltered input and
fraction zero removes the entire selected source. Integer rounding is explicit.

The export records the requested fraction, seed and rule plus the usual before/
after mask digests and per-month availability. This is a registered simulation of
missing observations, not a claim that a real sensor outage follows that geometry.
Compare methods on identical masks using readouts fixed on complete inputs;
retraining a readout under each missingness condition is a different experiment.
