# Larger physical context with the same frozen visual encoder

The balanced-v2 standardized-joint study finished all 120 primary conditions and
fails acceptance. Its native MLP/CNN macro F1 percentages are 77.038/84.271 in
Haidian, 51.247/59.593 in Harbin and 57.429/58.896 in Auckland. The corresponding
AEF references remain 83.875/84.382, 58.396/58.796 and 63.434/64.683. The public
envelope did not decrease. All negative scores remain retained; no full query
was launched, and no manuscript claim was made.

The twelve separate source-diagnostic conditions also completed, with sampling
and target contracts verified against the corresponding v2 AEF records. For
Auckland's native visual descriptor alone:

| Task | Visual MLP/CNN F1 % | AEF MLP/CNN F1 % |
| --- | ---: | ---: |
| Urban Parkland/Open Space, LCDB 2 | 29.644 / 42.747 | 56.779 / 62.783 |
| Exotic Forest, LCDB 71 | 27.071 / 25.494 | 77.776 / 80.060 |

The full 64D visual source is insufficient for these tasks under the fixed reader
protocol. This does not prove that the original 1024D teacher lacks information;
its own regional projection, spatial context and task fitting also matter. It
rules out treating the subsequent sixteen-coordinate fusion cut as the sole
explanation. The diagnostic subset remains excluded from primary model ranking.

## Registered context experiment

The three roles currently read exactly registered 0.5 m working-grid RGB images.
A 512-pixel encoder window therefore spans only 256 m, even for public S2 imagery
upsampled onto that grid. Distinguishing land use or forest types may benefit
from broader surroundings. Test this hypothesis without changing the global
SAT-493M weights or acquiring new imagery.

Add a strict optional `input_reduction` to the frozen feature extractor. The
legacy default is 1. Values 1, 2, 4 and 8 are accepted; booleans, floats and other
values are rejected. Average only observed source pixels in each exact block,
retain fractional observation coverage, and require divisible source dimensions
so the geographic footprint is unchanged. Exclude nonfinite unobserved fillers;
reject invalid masks and nonfinite observed values. Patch alignment uses the
reduced source dimensions over the original bounds, and the saved status records
both dimensions and the reduction. The default returns the original tensors.

Use reduction 4 everywhere: the same registered input images now reach the encoder
on a 2 m grid. Each 512-pixel crop can span 1,024 m; patch tokens sample 32 m rather
than 8 m. This deliberately trades fine visual detail for context, so an accuracy
improvement is a hypothesis. Keep source imagery, dates, radiometry, masks and
model normalization. Native data remain genuinely finer than the degraded 10 m
control; resampling public S2 never adds observed spatial detail.

Re-extract nine visual sources, independently fit one unlabeled regional PCA64
per role with seed 41 and at most 65,536 samples, then align to each unchanged
128-by-128 representation grid. There is no cross-region pooled fitting. Keep
the existing role-matched learned descriptor and original physical observations.
Use the current common magnitude-preserving allocation: learned16/visual16/raw32,
without branch/final unit normalization, for every region and role. The learned
branch still contains its original finer-input information; it is not retrained.

Fit all eighteen fresh readers and 120 primary conditions with the approved v2
cohort, original masks/supports/seeds, 20 support tiles and 100 reader updates.
Keep all verified v2 AEF and strongest genuine public references and raise the
public envelope if necessary. Require the unchanged six-slot and per-region
margin gates before full-region queries. Preserve all task-level failures.

## Verification and execution boundary

Failing-first tests cover observed-only weighted means, missing coverage, legacy
identity behavior, invalid options/observations, footprint alignment and actual
extraction before model windowing. Validate three real regional native samples
with the frozen SAT-493M encoder on free NPUs, including finite features and
unchanged output footprint. Run full tests, format/lint, repository policy and
package build before launching the full extraction queue. No unrelated device,
protected model, previous result or manuscript is modified.

Verification completed before full extraction: all 23 focused tests pass; the full
suite passes 1,008 tests in 335.84 seconds (two warnings). Black, Ruff, repository
policy and build pass. The wheel and source archive contain the exact changed
implementation. Real frozen-encoder NPU gates pass for one observation-selected
sample from each region: working source 640-by-640, complete original extent
2,560-by-2,560, finite native features and finite 64-by-128-by-128 outputs. These
are software/data gates, not downstream accuracy or three-region acceptance.
