# Calibration distribution review and supplemental cohort

**Current status:** the user approved adding the frozen balanced cohort on
2026-10-19, while retaining all old results. Follow the
[version 2 evaluation contract](balanced-calibration-v2.md). The review below
records the original proposal before that approval; its old metrics remain
version 1 results and are not relabelled as version 2.

The common six-coordinate semantic budget completed all120 fixed conditions and
failed acceptance. Native high-resolution macro-F1(MLP/CNN,percent) was
69.097/76.335 in Haidian,56.161/59.131 in Harbin,and57.494/58.107 in Auckland.
All rows are retained; no new full-region queries were run. Preserving99.9%of
the auxiliary probability variance did not ensure downstream improvement.

A source-label audit then found a large geographic coverage difference:

| Haidian CLCD2025 task | Full-region positive pixels | Current calibration positives |
| --- | ---: | ---: |
| Cropland | 118,972 | 12,489 |
| Forest | 87,444 | 277 |
| Water | 4,902 | 1,000 |
| Impervious | 340,322 | 102,001 |

The67 calibration tiles contain only0.317%of all registered forest pixels,
spread across12tiles. The95 remaining tiles outside the support and calibration
pools contain48,454 forest pixels. The existing numbers remain valid comparisons
on that particular calibration pool; its class coverage does not describe the
whole study area.

The four registered binary maps match their declared raw CLCD codes at every
calibration observation(115,767 per task). The official classification workbook
also identifies code2as Forest. No class-code correction was made. The workbook
was read from the [official CLCD2025 record](https://zenodo.org/records/18180184).
A separate fixed-mask consistency diagnostic shows low agreement between
WorldCover2021tree cover and these particular CLCD2025forest pixels; differing
dates and class definitions require interpretation rather than relabelling.

An optional additional calibration cohort is prepared but **not approved or used**:

- Keep the original support pools, support sampling seeds, annotation budgets
  and calibration tile counts(67/63/64).
- Select only tiles with valid primary reference observations and outside the
  original support pool. The registered grids remain320/422/320tiles; buffers
  and empty reference tiles are not added to calibration.
- Use seed41and the same class-count balancing algorithm in every region.
  Targets follow full-region class prevalence, with at least five positive
  tiles per class where feasible. Selection uses reference distribution only,
  never candidate predictions or scores.
- The concrete Haidian proposal contains18,289forest pixels. Harbin and Auckland
  use the same rule. Preserve the original cohort and every negative result.
- Adopting this cohort requires the user's choice because the active objective
  fixed the previous evaluation protocol. All affected AEF, public and local
  readouts must use the new cohort together; old thresholds/checkpoints selected
  on the former calibration cannot be treated as newly calibrated baselines.

The proposed cohorts remain exploratory: previously inspected areas do not
become an independent blind test by repartitioning. No manuscript body, active
evaluation data, baseline score or method acceptance rule has been changed.

Independent input preparation continues. Public RGB composites now use all
twelve available months under one clear-pixel-median rule: June2025–May2026,
November2024–October2025,and January–December2024. Auckland's already matching
annual images/features are reused. New geometry-only image-encoder indices carry
the actual twelve-month provenance and no cached image/label tensors. The two
Chinese areas are encoded with the same frozen SAT493M weights, seven shards,
one regional PCA after merging, and verified owned-native-cache cleanup. This
preprocessing neither adopts the proposed cohort nor starts new neural fitting.
