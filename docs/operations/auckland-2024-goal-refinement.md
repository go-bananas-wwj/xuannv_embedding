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

## Bounded wider-context follow-up

The eight multiscale candidates did not beat the retained parent's validation
criterion, so no new query suite was run. See the [complete validation-only report](auckland-2024-refinement-pyramid-results.md).

Register four further 64D candidates before validation: 11x11 joint mean, 13x13 joint
mean, and 13x13 aggregation with 10% or 20% native detail mixed back before final unit
normalization. Keep public48/semantic16 projections, images, masks, geometry, readouts,
fixed validation anchor and original acceptance threshold unchanged. Retain `wide_k9`
as the parent. Query only a new validation winner. AEF and public-PCA48 controls receive
exactly the selected spatial filter. This tests a limited increase in context with
partial detail preservation; it does not assert that wider pooling benefits height
regression. Keep the first accepted archive and both completed search records.

## Complementary public statistics: registered follow-up

The four wider-context candidates also failed the validation criterion; no query
suite was run. Retain `wide_k9` and the original read-only anchor.

Next test information complementarity using the already prepared 95-channel public
spectral/seasonal statistics (annual per-band mean/std/min/max, availability and
vegetation/water-index statistics). These are the same three public satellite inputs,
with no new sensors or ground-truth maps. Fit StandardScaler and PCA only to the fixed
unlabeled image positions, save every parameter and preserve missing-source zeros.

Register eight 64D cases: public/semantic/raw dimensions 40/16/8, 32/16/16 or 32/24/8,
each with joint 5x5 or 9x9 context (six cases). Also test 40/16/8 and 32/16/16 with
public/semantic/raw branch kernels 9/9/1 and no final averaging (two cases). Keep the
same archived validation reference, validated parent, readout rules and acceptance.
Only a new validation winner gets query scoring. The no-high-resolution control keeps
both public encoder features and the exact raw-statistics branch; only semantic image
features are zeroed. Existing matching AEF controls at 1/5/9 cells are reused.

Before any complementary-statistics readout is trained, align its selection rule
with the stated new goal: require positive MLP and CNN validation gains, then maximize
their mean gain (ties use the smaller gain). Candidates sacrificing either head rank
below every joint improvement. The original archived validation anchor remains fixed.
A validation-only audit of all completed refinement candidates still selects `wide_k9`
under this rule. No query result or new raw-source readout determined this change.
Earlier published rounds retain their recorded balanced-minimum criterion. This
prospective selection change and all adaptation remain explicit in the study record.

## Feature-affinity context: registered follow-up

The complementary-statistics winner reached MLP 58.8316%, CNN 57.2448%, a mean gain
of 0.2920 points over the original archive. It does not meet the 0.5-point goal.
Both the 90 m-only candidate and this tradeoff remain recorded, with all class,
fold and regression outcomes. The latest validation-selected statistics candidate
is retained as the parent regardless of its query tradeoff.

Register six label-free affinity candidates: public48/semantic16 and
public32/semantic24/raw8, each with 9x9 context at temperatures 0.2 or 1.0, and
13x13 context at temperature 1.0. For unit feature vectors, a neighbor's weight is
exp((cosine_similarity - 1)/temperature); normalize by the sum of weights inside
the same tile, then unit-normalize. No label/prediction or reference-map value enters
the affinity. This checks feature-aware context while retaining sharper transitions.

This is our direct cosine-feature filtering hypothesis, not an implementation of the
[Fast Bilateral Solver](https://arxiv.org/abs/1511.03296). That work motivates studying
edge-aware processing; it establishes no result for our region or feature vectors.
Selection remains prospective positive-mean validation improvement against the fixed
archived anchor, with the latest best-validation parent retained. Only a new winner
gets query scoring. Apply exactly the selected affinity filter to the AEF and public-
only controls; do not reuse uniform-average controls as if they included affinity.

## Actual aerial resolution follow-up

The affinity winner completed MLP 58.8877%, CNN 57.0911%, also below the original
0.5-point mean-improvement goal. Its matched controls and all outcomes are retained.

The [official DINOv3 model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md#training-data)
states that SAT-493M uses 512x512 Maxar images at 0.6 m. Our current RGB is read at
1 m. The prospective hypothesis is improved physical-scale alignment and genuinely
finer local texture, not a guarantee of improvement from more pixels.

Prepare the same 320-tile, EPSG:2193, 2024 LINZ footprint at 0.5 m from the original
0.25 m imagery, selecting only native source/overviews at resolution <=0.5 m. Reuse
the same 84 official source URLs and five ROI strips with four download workers.
Budget 20 GiB for this acquisition and retain a 100 GiB free-space reserve. No national
imagery, point clouds, new sensors or reference labels are downloaded. Preserve the
existing accepted archive, 1 m raster and all former results.

After bounds, CRS, real read resolution and coverage checks, use the same frozen
SAT-493M weights, normalization, overlapping 512-pixel windows, PCA sample budget,
10 m output grid and 64D representation. Crop batches may use the existing supported
maximum of eight. No duplicate monthly aerial frames are introduced.

Register public/semantic dimensions 48/16, 40/24 and 32/32, each with joint 5x5,
joint 9x9 or 13x13 affinity at temperature 1.0 (nine candidates). Keep the archived
validation anchor and positive-mean rule; retain the latest best-validation parent.
Query only a new validation winner, with exact matched AEF/public controls. Also
retain or fit a matched 1 m setting for the selected 0.5 m recipe, so changes in input
resolution are not conflated with dimension/context selection. Negative results,
all classes and all LiDAR targets remain reported. No Overleaf edits.

## Encoding-scale alignment control, using existing imagery

A separate prospective control retains the original real 1 m image information but
resamples it bilinearly to a 0.5 m input grid. A VRT records the original raster hash;
this is explicitly interpolated 1 m data, not newly observed 0.5 m imagery. ViT-L/16
then has an 8 m native token spacing instead of 16 m, while both outputs remain 10 m.
This separates encoder sampling/physical-scale alignment from extra source detail.

First verify two actual input tiles at crop batch eight, then extract the complete
320-tile frozen feature bank and test the same nine dimension/context cases. Only
validation selects a winner. Compare the selected recipe with the exact same recipe
using the original 1 m source and 1 m encoder grid, as well as AEF and public-only
controls. The previous accepted model, prior 90 m results, spectral tradeoffs and all
negative trials stay recorded. Retain the existing real-half-meter download as a
separate dataset. If the scale-only control meets the refinement goal, finish that
acceptance and preserve true 0.5 m data for a later approved/goal continuation rather
than automatically launching another sweep. If not, test the actual 0.5 m source.

The scale-only experiment never supports a claim of added observational detail.
No Overleaf edits are made.
