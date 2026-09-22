# Input-matched monthly observation baseline

`xuannv experiment export-raw-monthly --spec spec.json` creates fixed raw features
from the same cached forward inputs as a monthly high-resolution model. It does not
train a representation, use labels to choose features, or compute an accuracy metric.
This differs from the historical 150-dimensional baseline with static high-resolution
imagery and target-derived public validity masks. Keep their results separate.

The specification has exactly these fields:

- `protocol`: `monthly-input-raw-v1`.
- `cache`: registered `{path, sha256}` of the input cache JSON.
- `sources`: explicit ordered list containing every `model_inputs` source exactly once.
- `period`: the final month in the cache's ordered observation window.
- `splits`: distinct canonical train/validation/test/buffer names.
- `output`: a new directory; never overwrite an earlier export.

For each source, concatenate each month's bands followed by one availability channel.
Source order comes from the specification, month order from the cache, and band order
is the original cache order. The manifest enumerates every output channel. All months
are concatenated once into a static vector indexed by the window endpoint; this is
not a monthly embedding or a claim of online temporal prediction. Constant timestamps
are registered metadata shared by every tile, not redundant predictor channels.

Temporal public inputs preserve their cached bands and forward time masks, with absent
months zeroed. The broadcast time mask is their availability channel. No target-derived
pixel validity is added as a new feature: the public encoder receives only the time mask.
For high-resolution inputs, preserve the monthly native-resolution frame and binary
forward availability mask. On the same north-up footprint, integrate exact rectangular
pixel overlaps separably in float64, divide the masked area sum by available area, then
cast the mean to float32. The extra channel is available area divided by output-cell
area. This handles noninteger sizes (e.g. 427 to 128) without rounding an assumed scale.
Empty areas have zero values and zero availability. The overall feature-valid mask is
the union of availability across all sources and observed months.

The same-footprint north-up assumption requires upstream geographic provenance. This
preparer does not infer a CRS or certify native high-resolution geolocation. Area means
discard within-cell texture; this is an explicit raw summary baseline, not a lossless
representation of native imagery or a learned native-resolution competitor. Existing
strong readout heads receive the full declared raw feature dimension.

Only forward frame/mask dictionaries and timestamps are used. Reconstruction targets,
target masks, semantic labels and loss-only quality masks are ignored even though the
cache file may also contain them. Sources, shapes, timestamps, binary masks, visible
finite values and every selected sample digest are checked. Wholly missing modalities
remain represented by zero values and availability; no valid zero is inferred missing.

Exports retain the complete cache layout but write only requested split files, record
exact exported indices and tile digests, and round-trip through `multitask_features`.
Unexported records do not have digests and cannot pass that reader's inventory check.
Failures retain status and partial files without a complete manifest. Source file hashes,
ordered channels, software identity and runtime are recorded. Before formal test access,
follow the final candidate lock protocol; this command does not select or lock a model.

Synthetic tests independently check noninteger overlap arithmetic, masked means and
fractions, ignored labels, missing modalities, input errors, source tampering, selected-only
file access and the final feature-reader contract. Real feature preparation is not evidence
that one method is more accurate than another.
