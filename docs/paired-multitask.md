# Paired primary multi-task calibration and final scoring

`xuannv experiment calibrate-primary --spec SPEC.json` fits and archives the primary
classification, coverage-regression and example-retrieval readouts on training/validation
data. After reviewing and registering the resulting `calibration/identity.json` digest,
`xuannv experiment score-primary --spec SPEC.json --calibration-identity-sha256 DIGEST`
loads those frozen parameters and scores test data without refitting. Each phase requires
a new output subdirectory; a partial or complete run is never silently overwritten.

This is the primary C/R/Q controller, not completion of the research benchmark. Strong
heads, real label-bundle preparation/provenance review,
external geographic audit, and the actual locked-candidate comparison remain required.
Software tests use synthetic data and do not establish an accuracy improvement.
Use [paired-multitask-report.md](paired-multitask-report.md) for spatial uncertainty and
task-family aggregation of the archived predictions.

## Locked specification

The JSON object has exactly these fields:

| Field | Contract |
| --- | --- |
| `protocol` | `multitask-final-primary-v1` |
| `output` | New run root with separate `calibration` and `test` directories |
| `reference_cache` | `{ "path": "...", "sha256": "..." }` for the canonical cache JSON |
| `labels` | `train`, `validation`, `test`, each with a path and SHA-256 |
| `models` | Named model realizations, each using the G3 feature contract below |
| `month` | Reference evaluation month present in the canonical cache |
| `support_seeds` | Distinct positive integer support seeds |
| `budgets` | Distinct positive training-tile budgets for C/R |
| `retrieval_budgets` | Distinct positive training-component budgets for Q |
| `method_groups` | Method name to nonempty list of realization names; partitions all models |
| `geographic_audit` | Path and SHA-256 of reviewed upstream geographic evidence |
| `lock` | Path and SHA-256 of the candidate/evaluation contract lock |

Each model requires `manifest_path`, `manifest_sha256`, `cache_path`, `cache_sha256`,
`tile_sha256`, and `selection`. The selection contains `kind`, `period`,
`evaluation_month`, and original `channels`, as in [multitask-features.md](multitask-features.md).
An annual product retains its annual source period; it is not relabeled as a monthly
observation. Different feature dimensions are accepted without padding or truncation.

The lock must contain `state: "locked"` and `contract_sha256` equal to the
`paired_multitask.contract_sha256(spec)` result. The digest binds every specification field
except `output` and `lock` using sorted compact JSON. It does not decide which candidate
wins: create the real lock only after the registered validation selection and training
recipe are complete. Model and label identities, period, budgets and seeds cannot then
change without an explicitly different registered comparison.

The upstream geographic record must bind `reference_cache_sha256`, exact
`model_manifest_sha256` mapping, `state: "verified"`, and a nonempty `evidence` list of
path/SHA-256 descriptors. The controller verifies those identities and matching ordered
tile bounds, layouts and partitions. It does not independently reproject external rasters
or establish the CRS from numeric bounds. The upstream audit must actually verify CRS,
pixel grid, decoding, nodata and reprojection before its result is registered here.

## Labels and paired domains

Each split uses its own numerical NPZ with exactly `indices`, scalar `cache_sha256`,
`esri`, `osm_building`, `osm_road`, `osm_water`, and `osm_green`. Integer `indices` must
match the canonical cache split in order. Arrays are `[split_tiles, H, W]`; OSM encodes
unknown as -1 and binary classes as 0/1, while ESRI encodes unknown as -1 and six classes
as 0..5 in water/trees/range/crops/built/bare order. These are reference map labels;
coverage regression is a map-derived proxy, not independent biophysical ground truth.

Bundle creation and geographic provenance review must preserve the existing registered
labels and splits; this controller does not infer or regenerate reference labels. Separate
bundles permit calibration while every test label and feature file is unavailable.
Calibration opens only train/validation files; score opens test files after verifying the
lock, producer code, archived validation predictions and every frozen readout identity.

The comparison intersects all model feature-validity masks on the same canonical tiles,
then intersects that domain with each task's reference-label validity. All methods use
identical support tiles, sampled support pixels/components, query truth, and query tile
mapping. The common feature mask and task labels are archived with hashes. Feature values
remain in their original dimensions; invalid values do not become valid zero observations.

## Conditions, fitting and reported metrics

Classification covers four OSM and six ESRI one-versus-rest tasks. Coverage regression
covers six ESRI tasks using 16-by-16 grid blocks with at least 80% jointly valid pixels.
At the registered 10 m resolution these are 160 m blocks; upstream geographic evidence
must establish resolution. Retrieval covers the four OSM tasks using training connected
components of at least four pixels. Five support seeds, C/R budgets 5 and 10, and Q budgets
1, 3 and 5 produce 220 conditions per model. All conditions are fixed before final scoring.

Sampling matches the registered validation evaluator. Classification uses nested
positive/negative support tiles and up to 4096 pixels per class; regression uses the
fixed seed/tile hash ordering; retrieval uses fixed seed/tile/component ordering.
Support descriptors must agree across all models. Scalers use only training support.
Ridge alpha is selected from 10, 1, 0.1 on validation AP or clipped validation RMSE;
classification threshold is selected by validation F1. Saved C/R parameters must reproduce
validation predictions exactly after loading. Retrieval saves normalized training
prototypes without a second normalization.

Outputs include classification AP/F1/IoU/balanced accuracy and pooled boundary F1 at one
and two pixels; coverage RMSE/MAE/R-squared/bias plus the training-support-mean baseline;
retrieval AP and top 1%/5% precision/recall. Test labels enter scoring, never parameter
selection. Prediction files archive scores, truth, canonical tile IDs, and C/Q valid
positions. An empty test reference domain remains an explicit condition with undefined
metrics rather than disappearing. No-positive AP, undefined R-squared and other undefined
metrics use JSON null; later aggregation must preserve and report undefined conditions.

The two phases save per-condition predictions, supports, identities, selected parameters,
results, and running/failed/complete status. Code hashes bind the controller, feature reader,
readouts, sampling/boundary routines and task metric routines. Linear readout loading also
checks its producer's numerical-library versions. Use the archived producer environment
and source snapshot for replay. Changing current source does not change existing pinned
training or evaluation jobs.

Tests remove all synthetic test files during calibration, forbid fitting during score,
compare paired truth/tile domains, verify unchanged calibration artifacts, retain empty
test domains, reject changed locks/geographic evidence/parameters/label partitions, and
exercise the two CLI commands. The scaler regression test covers nontrivial float32 and
float64 statistics in both C and Fortran memory layouts.
