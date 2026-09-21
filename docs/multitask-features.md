# Fixed features for multi-task comparisons

`downstream.multitask_features.read_features` reads registered monthly embeddings, annual
embedding products and last-month raw-feature exports without assuming a common feature
dimension. `FeatureSelection` explicitly declares the kind, source period, evaluation month
and original channel count. For example, an annual 2025 product can be evaluated against a
May 2026 reference, but its recorded temporal resolution remains annual. No monthly
observations or future-input guarantees are inferred from that comparison.

The reader requires the exact export-manifest, source-cache and selected tile-file SHA256
values. The export must identify that cache and match its ordered tile IDs, bounds and split
metadata. Every tile must have the declared `[period, channel, height, width]` shape. Monthly
embeddings require timestamps matching all manifest months; annual products use the explicit
`annual_YYYY` manifest period and cannot carry monthly timestamps. Raw exports must identify
their kind as `raw`; their one-period files may lack timestamps. No channel padding,
truncation, normalization or geographic resampling is performed.

Only requested canonical splits are read; the default is train and validation. Cache metadata
may retain auxiliary spatial groups or pilot subsets, but those aliases cannot be requested
as an alternative route to test features. An explicit test request is recorded in the returned
identity. The caller is responsible for enforcing the final-candidate lock before making it;
this low-level loader neither authorizes final evaluation nor opens labels.

Each tile may store a boolean `valid_mask` with shape `[H,W]` or `[period,H,W]`. Invalid values
are numerically filled with zero **and remain invalid in the separate returned mask**. A
nonfinite value on the declared valid domain fails. Without a mask, the finite exported grid
is the representation's declared domain; this is not a claim of full source-observation
availability. Final comparisons must intersect feature/reference validity consistently across
methods before drawing support or computing predictions. They must not treat filled zeroes as
additional valid observations.

The optional new `output` directory stores memory-mapped feature/validity arrays and writes
the identity only after all selected tiles pass. Existing directories are rejected. A failed
read may leave incomplete arrays without an identity; those must not be treated as a completed
bundle. Dimensions and original feature values are retained as float32, consistent with the
existing export pipeline.

Geographic identity here is **inherited from cache-bound export provenance**. Numeric bounds
alone cannot prove a coordinate reference system or a correct external reprojection. Existing
external-product CRS, alignment and source-validity audits remain required upstream; the
reader does not invent a CRS from coordinate magnitudes or a region name. Cross-model callers
must verify the common geographic reference in addition to comparing layout hashes.

The running validation-only evaluator remains unchanged and uses its pinned snapshots.
The final file-backed evaluation still needs to connect this reader to frozen validation
choices, common-domain support sampling, stronger heads and task-family bootstrap reporting.
Synthetic tests cover period/dimension handling, split isolation, file tampering, masks,
timestamp and geometry mismatches, future cutoffs and memory-mapped output.
