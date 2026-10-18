# One calendar-aware regional fusion encoder

The static95-channel observation summaries discard month association. This
experiment instead uses all twelve ordered monthly observations and fixed
visual features, with one architecture and loss rule fitted independently in
every region. No primary CLCD, DW or LCDB label is an encoder training target.

The encoder contains:

1. Separate stems for eleven S2, two S1 and six Landsat physical bands, each with
   its explicit observation mask. Missing values are cleared before encoding.
2. Calendar-associated pooling using the constant, annual sine/cosine and
   second-harmonic bases. Moving a scene to a different month changes the static
   representation; jointly permuting scenes and their dates does not.
3. Two-resolution spatial context from those temporal summaries, public annual
   S2 RGB features and public Copernicus terrain. Region-specific normalization
   is fitted from observations with a common rule.
4. Monthly64D public features and a gated, zero-initialized local optical visual
   correction. A missing local source gives the exact current public path.
   The static descriptor is the mean of the twelve monthly features.

Both public and local visual inputs use the same frozen SAT493M weights. The
local optical features are static period context, not twelve independently
acquired images. Local SAR is not an input in this experiment; public S1 remains
part of the monthly observations. Monthly outputs use full-period context and
are not a causal forecast or validated change-detection product.

The loss combines public monthly reconstruction(source weights0.4/0.125/0.225),
genuine twelve-index ESA WorldCover2021v200 supervision(weight0.45,zero ignored),
public/local visual reconstruction(weight0.1each),and a local-increment penalty
(weight0.02). The weak-label domain comes from the original public observation
masks and does not expand when local imagery is added. Reconstruction retains
continuous information beyond coarse map classes.

The first30%of updates use no artificial input hiding. Afterwards, random
source, month, spatial-block and static-visual hiding encourage reconstruction
from complementary observations. Actual invalid observations remain masked
throughout. Public corruption draws are identical with and without local input.

Register nine fits before their downstream scores: three regions times native
local optical, public-only and degraded-local roles. All use width32,residual
scale0.5,1,200updates,AdamW learning rate0.0003,weight decay0.01,and seed41. Every
registered tile must be visited. Final checkpoints are selected by the fixed
training budget rather than different regional checkpoints. Optimizer recovery
is removed only after the independent final state loads strictly.

CPU regression tests cover calendar association, monthly/mean consistency,
optional-local fallback, real weak targets, label-domain equality, deterministic
public corruption, checkpoint recovery and map-free export. Real two-tile NPU
gates passed in all regions at128x128,including finite forward/backward, exact
zero initialization, exact missing-local fallback and identical restored output.
Device7batch smokes measured:

| Batch | Second-step seconds | Tiles/second | Peak allocated memory |
| ---: | ---: | ---: | ---: |
| 16 | 0.2913 | 54.92 | 17.36GB |
| 32 | 0.5755 | 55.60 | 34.71GB |

Use one common batch32for the regional fits. These are hardware measurements,
not downstream performance. Training uses BF16 arithmetic with float32 stored
weights. Devices1–7run independent fits; device0belongs to a separate service.

Use the original registered evaluation cohort while the optional balanced
calibration proposal remains unapproved. Preserve all previous results and
primary tasks. Evaluate both MLP and CNN under the same support budget and
selection rule; no software gate or training-map fit establishes the three-region
AEF objective. No manuscript body has been changed for this experiment.

The experiment has now completed all nine fits and all 120 calibration conditions.
See the [complete calibration result](calendar-temporal-fusion-results.md).
It fails the registered gate; no full-region query was launched for this candidate.
