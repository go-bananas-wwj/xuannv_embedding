# Refinement goal after the first accepted regional representation

The user requested a separate archive of the accepted result followed by another goal
run on 2026-10-18. The accepted MLP/CNN pooled Macro F1 values are 58.6834026561% and
56.8088815524%. Their exact outputs, source code snapshot, both encoder checkpoints,
three projections, configurations, geographic index, support/target references,
readout weights, predictions and all compact results were independently archived.
The archive contains 65 hash-verified files (2.229 GiB logical size), uses independent
inodes and read-only files. Original acquisition rasters remain in the data workspace.
It is a reproducible inference-fusion model, not a newly jointly trained checkpoint.
The archive and its local manifest are separate from every new candidate directory.

## Acceptance and fixed protocol

Keep 320 tiles, 64 dimensions, five spatial folds, all common masks, support locations,
200 points/class and original readout budgets. Both pooled classification heads must
exceed the archived model; their mean gain must be at least 0.5 percentage points.
Report every class and all four LiDAR regression targets. Prefer reducing the height
regression loss introduced by uniform spatial aggregation, but do not select a model
using query regression scores or hide negative results. Do not modify Overleaf.

Current benchmark scores have been inspected adaptively. Additional improvements
remain exploratory regional tuning, not an independent blind evaluation. Model
selection uses only the registered fold-zero validation predictions, maximizing the
smaller gain over the archived parent's MLP/CNN validation scores, then their mean.
The frozen test result supplies acceptance thresholds only, never candidate ranking.
Retain the validated parent if none improves the validation criterion.

## Motivation and bounded first search

The original accepted candidate gains classification accuracy from spatial context
but loses height detail. Test twelve label-free alternatives without retraining the
encoders or adding ground-truth inputs:

| Group | Fixed settings | Count |
|---|---|---:|
| Wider context | Public48 + semantic16, joint 7x7 or 9x9 mean | 2 |
| Local residual | Same, 7x7/9x9 context with 20% or 40% native detail | 4 |
| Separate source scales | Public/semantic kernels 1/5, 3/5, 1/9, 3/9, then branch normalization | 4 |
| Relative source strength | Separate 5/5 kernels; public/semantic weights 1/0.5 or 1/2 | 2 |

Every representation is 64-D, unit-normalized and remains on the same 10 m grid.
Pool only within a tile. Per-source aggregation preserves zero values at unavailable
source centres. Source projections use the same unlabeled seed/sample budget as the
accepted recipe; no LCDB, LiDAR or AEF feature values fit our embeddings. All candidate
settings are registered before validation. Only the selected winner gets the full
query suite. Negative validation results remain recorded.

General motivation comes from complementary temporal/spatial satellite fusion
([M3Fusion](https://arxiv.org/abs/1803.01945)) and frozen dense visual representations
([DINOv3](https://arxiv.org/abs/2508.10104)). The residual/scale choices above are our
hypotheses; those papers do not establish their benefit on this area. We continue to
disclose SAT-493M's additional global pretraining. No global-model training claim is
made for the regional model itself.

## Controls and execution

Compare against the archived fused model, original AEF/public and the existing
50 m context controls. If a wider/residual filter wins, also give AEF and the exact
public PCA48 control that same joint spatial filter, with unchanged readout rules.
If separate source scales win, use AEF with each of its distinct spatial scales as
validation-only alternatives and query only the selected AEF control. Controls never
select the fused winner. All differences and drops are reported.

Use available NPU devices 1-7; preserve the unrelated device-zero service. A local
supervisor checks quietly, reuses completed validation work, retries failed jobs at
most twice, and stops below the 100 GiB storage reserve. Publish source, registration
and meaningful milestones to GitHub; large arrays, checkpoints and local provenance
stay outside Git. The accepted archive is not modified by subsequent runs.

If this bounded search does not meet the goal, inspect validation evidence before
registering a new hypothesis: complementary annual public spectral/seasonal statistics
or an unlabeled feature-preserving adapter. Do not launch an unbounded sweep because
a device is idle, and do not promote the best query-scoring failed candidate.

Source-strength settings are diagnostic: per-channel support-set standardization
largely cancels constant branch scaling where both branches are fully observed.
Any difference must be interpreted through missing-source normalization and spatial
geometry, rather than claimed as new information from multiplying a constant.

## Conditional multiscale follow-up, not launched

The twelve completed validation cases selected 9x9 joint averaging. Increasing the
local residual did not improve the balanced criterion. If its full query result fails
the new 0.5-point goal, test preserving two scales as separate coordinates instead of
averaging them into one vector. This addresses information loss without new imagery.

For each kernel 5 or 9, allocate public-local/public-context/semantic-local/semantic-
context dimensions as 24/16/16/8, 16/16/16/16, 16/24/8/16 or 32/16/8/8. Each totals
64 dimensions. Fit PCA only on the same unlabeled positions, apply kernels 1/k/1/k
within tiles, normalize branches and concatenate. There is no extra joint pooling.
Keep the previous best validation candidate eligible with its cached readouts; rank
all candidates against the same archived validation anchor. Only a newly selected
candidate receives new query fits. Do not use the first wave's query scores to choose
the dimensional allocations or the multiscale winner.

## First refinement outcome and multiscale activation

The 9x9 candidate completed all five query folds: MLP 59.1703%, CNN 57.2169%,
improving the archived candidate by 0.4869 and 0.4080 points (mean 0.4475). It also
exceeds AEF with matched 9x9 averaging (57.9650%, 56.9066%). The mean gain is below
the registered 0.5-point threshold, and surface-height RMSE increased further. The
goal therefore remains active. [All results](auckland-2024-refinement-context-results.md)
include all classes, folds, continuous targets and twelve validation trials.

Activate the eight previously registered multiscale candidates. Retain `wide_k9` as
the validated parent, while both validation gains and query acceptance thresholds
remain anchored to the original read-only archive. This keeps the ranking rule
consistent across waves; the parent's query score does not select new candidates.
If the parent wins, do not refit or retest it. Matching AEF controls at 1/5/9 cells
are reused from completed work rather than trained again. The public-only control
retains both public scales and exactly the selected source projections.
