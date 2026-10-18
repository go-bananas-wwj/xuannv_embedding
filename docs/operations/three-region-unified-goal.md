# One high-resolution method that passes in three regions

The user explicitly started a goal run on 2026-10-18. Completion requires one
shared method to work in Haidian, Harbin New Area and west Auckland. A separate
winning recipe, reader or favorable task subset per region does not meet it.

## Manuscript anchor

The actual Overleaf manuscript was fetched and inspected at
`a63fa74d5c4ed9e23c92d287530daa6a6a35ff44`. Its body matches the inspected local
manuscript; newer changes add poster material. The title is
“xuannv-earth：面向不同区域的多源地理嵌入模型”. The existing method preserves public
temporal features and adds geographically aligned high-resolution information
through a residual, availability-aware branch. Each region trains its own weights;
the method and representation are shared across tasks. Preserve that scientific
question and the existing introduction/related-work foundation. Auckland is an
additional validation region, not evidence already present in the manuscript.

The manuscript's source-level comparisons already acknowledge weaknesses on
WorldCover/Dynamic World. The completed frozen-feature experiment additionally
finds that 40+24 fusion with 130 m aggregation does not transfer uniformly. Its
no-aggregation ablation improves Harbin, but Haidian CLCD and Harbin DW still trail
AEF. Those results remain protected and are not a completed three-region claim.

## Acceptance locked before new candidate scores

Use a common binary land-cover evaluation to avoid mixing the earlier Auckland
200-points/multiclass protocol with China's dense support-tile protocol:

| Region | Main reference | Classes/tasks | Full registered region |
|---|---|---:|---:|
| Haidian | CLCD 2025 | four original registered classes | 320 tiles |
| Harbin New Area | Dynamic World 2025 | five original registered classes | 422 tiles, original core mask |
| West Auckland | LCDB 2023/24 | eleven originally eligible classes | 320 tiles, original reference mask |

All three use the existing binary MLP/CNN implementations, 20 complete support
tiles/task, 100 updates, the same fixed support/calibration positions for every
representation, and a 64D output. Preserve reference-specific masks, classes and
native grids; do not remove unfavorable classes or change masks after scoring.
Auckland uses the original fold-zero support/calibration pools and scores all
registered tiles, matching the full-region manuscript estimand. All eleven classes
have at least twenty mixed positive/negative support tiles (minimum 23), checked
before scoring. The old five-fold multiclass benchmark remains separate evidence.

The same candidate must strictly exceed AEF and the genuine public-only control
in both readers in every region. Target a mean gain of at least 0.5 percentage
points over AEF in each region; gains elsewhere cannot offset one failed region.
These are six per-reader gates plus three practical-margin gates, not a pooled
three-city average. Retain the existing Chinese primary tasks as secondary checks.
After the main gates pass, extend the original paper's other readers/budgets and
retrieval/regression diagnostics, keeping negative results and claim boundaries.

Full-region scores include support/calibration locations and encoder exposure to
regional imagery. CLCD/DW/LCDB are not upstream targets, but repeated inspection
makes this exploratory regional map agreement, not a new independent blind test.
Before a stronger generalization claim, register an additional untouched spatial
or temporal confirmation. Do not relabel existing inspected results as blind.

## Same method and high-resolution contribution

Architecture, source handling, latent width, loss, physical neighborhood and
validation selection rule must be the same. Region-specific normalization/PCA
statistics, learned weights, CRS, observation dates and genuinely missing sources
are fitted/handled by that same rule. Never combine the cities into one training
set. All authorized local tiles remain available for regional representation
learning; downstream fitting/calibration roles remain distinct.

Restore Haidian's genuine three-public-source embedding from its retained terminal
checkpoint. The previous transfer used an already high-resolution-trained Haidian
base, so it is not a clean no-high-resolution control. The audited public ablation
configuration and terminal-weight registry are preserved. Harbin and Auckland
already have retained genuine public-only embeddings. Ancestor weights are not
needed to cold-load the complete retained states.

Compare AEF, genuine public-only, the same fusion with high-resolution unavailable,
the full high-resolution fusion, and DINO-only when DINO is used. Keep adaptation
budgets and map-label exposure matched in learned ablations. If DINO becomes part
of the final method, also control for its global pretraining with public RGB input;
do not attribute all benefit of an extra globally pretrained encoder to local
image resolution. No AEF, CLCD, DW or LCDB target maps enter candidate embeddings.

## Ordered search

1. Restore/verify the public baselines and prepare the common reference protocol.
   Reuse complete valid results; do not rerun old scores just for reassurance.
2. With existing frozen regional and DINO features, test a small preregistered set
   of public-preserving residual/aligned fusion and small/no neighborhoods. Preserve
   the original 64 coordinates instead of automatically replacing 24 of them.
   Constant concatenated branch scaling is not a useful standalone search because
   downstream per-channel standardization largely cancels it.
3. If fixed label-free fusion is insufficient, train one lightweight zero-residual
   availability gate on each region independently. Start from the public output,
   use common reconstruction/map supervision and a public-preservation constraint;
   avoid rewriting/retraining the entire backbone as the first response.
4. Test spectral/seasonal public statistics as an additional common source when
   validation indicates missing land-cover information. Compute the same statistics
   from S2/S1/Landsat in all regions; do not feed evaluation maps into training.
5. Select one shared recipe by its worst validation gain over AEF/public across all
   six region/reader slots, then its mean gain. Only a newly selected common winner
   receives full-region scoring. Do not choose three separate winners or change
   targets after a failed query. Register subsequent bounded waves explicitly.

The study may not achieve the desired result. Keep the goal active while there is
meaningful authorized work; do not mark it complete, invent improvements, or put
an unsupported three-region conclusion into Overleaf. A faithful fallback is to
report demonstrated regional/task limitations and discuss them with the author.

## Execution and publication

Use the available NPU devices concurrently; leave occupied unrelated processes
alone. A local controller owns bounded retries, disk checks, progress receipts and
checkpoint recovery, sleeping between checks. Do not spend conversation turns
polling unchanged training states. Keep at least 100 GiB free, preserve the three
original model bundles and all source/label/reference data. Store arrays, weights,
credentials, logs and detailed execution artifacts outside Git.

Synchronize code, compact plans and milestone results to GitHub. Capture Overleaf's
current version and preserve its body until verified results justify method,
experiment and figure updates. Do not rewrite the introduction or fabricate a
three-region victory while experiments are ongoing. Goal completion requires the
complete acceptance matrix, high-resolution ablation, reproducible artifacts and
an evidence-supported manuscript integration plan.

## Started baseline stage

Haidian's pure-public terminal bundle passed a strict cold load. All 320 public
samples and mean embeddings have been restored without encoder training and
aligned to the protected geographical index. Auckland's eleven-task binary
registry preserves all 2,618,840 original reference-valid pixels. Eight necessary
baseline jobs are queued on devices 1–7 with dynamic load balancing; unchanged
Chinese baseline scores are reused through their existing references. A local
controller handles bounded retries, stop requests and disk reserve checks. This
stage does not count as goal completion or as evidence for a new fusion method.
