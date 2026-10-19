# Preserve public context and add local fine detail

The context experiment finished all nine visual exports, nine fusion exports and
eighteen readers, covering all120 primary task-reader conditions. It fails the
common acceptance gate, so no full-region queries were launched. Calibration
macro F1 percentages are:

| Region | Reader | Native local | Matched public | Degraded local | AEF |
| --- | --- | ---: | ---: | ---: | ---: |
| Haidian | MLP | 83.494 | 82.895 | 83.331 | 83.875 |
| Haidian | CNN | 84.712 | 84.725 | 84.803 | 84.382 |
| Harbin | MLP | 56.883 | 57.317 | 57.732 | 58.396 |
| Harbin | CNN | 59.025 | 60.755 | 58.972 | 58.796 |
| Auckland | MLP | 60.974 | 58.875 | 61.620 | 63.434 |
| Auckland | CNN | 61.233 | 56.333 | 60.742 | 64.683 |

Native context improves over its finer magnitude-preserving predecessor by
1.168/0.583 percentage points in Haidian and1.403/1.275 in Auckland. Harbin
changes by-0.060/-0.593. The genuine public control raises registered reference
floors in several slots, including Harbin CNN to60.755%. Do not discard these
stronger public results. The same new native method remains below AEF in four
of six slots and cannot support a three-region victory claim.

Auckland Urban Parkland/Open Space improves with broader context, while Exotic
Forest still has a large deficit. Haidian MLP impervious F1 decreases2.69 points
relative to the finer predecessor, and Harbin CNN bare land decreases5.13 points.
Auckland Exotic Forest native F1 is40.890/38.602%, versus the AEF77.776/80.060%;
the degraded-local context even performs better on this class. Retain these
failures. Bigger context is not proof that more image detail always helps.

## One registered follow-up with fixed shared public information

The previous candidate substitutes a local visual branch for the public visual
branch. Its learned encoder is also role-matched, so a local input can displace
useful public information. Test one common composition that explicitly retains
public context and then supplies a smaller fine-detail branch:

| Branch | Dimensions | Inputs across the three roles within each region |
| --- | ---: | --- |
| Learned public calendar | 16 | The exact same genuine public-only encoder output |
| Broad public visual | 16 | The exact same public S2 context features on the2m working grid |
| Fine visual | 8 | Native local RGB, public S2 RGB, or degraded local RGB, all from the existing0.5m working-grid extraction |
| Public physical observations | 24 | The exact same original95-channel annual summaries |

The three shared source arrays are identical between roles within a region.
Only the eight-coordinate fine source changes. Each branch fits its unlabelled
regional PCA using the same seed41 and65,536 sample cap; physical coordinates
are standardized as before. Preserve branch/final magnitudes. Verify shared
projection means/components and the resulting shared coordinates agree across
roles. All outputs remain64D. Regions fit independently with one common rule.

This uses the existing public encoder weights and frozen visual arrays. There is
no new neural training, image extraction or acquisition. No native/degraded
learned encoder enters the candidate. Regional training of the public backbone
remains separate, with the same model and training rule across all three regions.
Local SAR remains absent; public S1 is retained in the temporal and physical data.

Existing label-free spectra retain only96.2%of Haidian public-encoder variance
in8 dimensions versus99.4%in16. Keep the sixteen learned coordinates. Reducing
the physical block from32to24 retains97.6–98.3%of its standardized variance in
all regions and frees eight fine coordinates. This is a single declared budget
tradeoff, not a primary-label dimension search or a guarantee of accuracy. The
candidate simultaneously changes anchoring and allocation; it is not a
single-factor ablation of the previous model.

Fit all eighteen fresh readers and120 conditions with the same approved balanced
cohort, original class masks, support pools/seeds,20 support tiles and100 updates.
Calibration reference labels used for threshold selection are additional to the
support budget and must be disclosed separately. No CLCD/DW/LCDB value enters
feature fitting. Reuse verified v2 AEF and strongest genuine public references;
the public envelope can only increase. Keep the six-slot and regional0.5-point
margin gates; frozen full-region queries follow only an eligible common model.

All prior task-level negatives are retained. Exact registrations, full comparison
CSV, sources, metrics and queue receipts stay outside Git. The existing block
exporter supports this recipe without production changes. The optional-vision
CI fixture issue is repaired;1,008 local tests pass, and GitHub's test/quality
checks plus its clean-install/build workflow pass at1063bde. That is software
verification, not evidence of task improvement. The manuscript body stays unchanged.
