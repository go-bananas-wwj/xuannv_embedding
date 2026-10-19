# Balanced v2 results and a magnitude-preserving paired experiment

All 72 fresh readers and all 480 primary task-reader conditions completed under
the human-approved balanced cohort. Every support/target/observation contract
matched across models. No old version 1 number was used as a version 2 reference.
None of the three common candidates passes acceptance; no full-region query was
launched. The best common candidate by worst-slot gain is the three-branch skip.

| Region | Reader | Native local | Matched public | Degraded local | AEF | Registered public envelope |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Haidian | MLP | 82.349 | 81.810 | 82.180 | 83.875 | 82.592 |
| Haidian | CNN | 84.721 | 84.302 | 84.632 | 84.382 | 84.905 |
| Harbin | MLP | 57.625 | 55.709 | 56.576 | 58.396 | 57.628 |
| Harbin | CNN | 60.032 | 58.398 | 58.673 | 58.796 | 59.467 |
| Auckland | MLP | 58.267 | 54.377 | 54.479 | 63.434 | 57.350 |
| Auckland | CNN | 59.032 | 55.558 | 55.768 | 64.683 | 59.769 |

These are exploratory calibration macro F1 percentages. Native imagery improves
over the matched public and degraded controls in all six slots for this candidate,
but that is insufficient to beat AEF and the stronger registered public envelope.
Auckland is now the largest deficit. All three candidates, every task, negative
result and original version 1 result remain retained outside Git.

## Label-free diagnosis

The learned calendar descriptor was trained for reconstruction and weak semantic
prediction, rather than with a spherical embedding objective. The fusion currently
centers and compresses each source with PCA and then normalizes every pixel's
branch and final descriptor to unit length. This discards radial information.

A fixed seed 41 diagnostic used forty fitting and twenty different held tiles per
region, at most 256 observation samples per tile, and no primary map labels. It
compared predictions of the saved WorldCover auxiliary head on the original
learned descriptor against reconstructions from its sixteen-component PCA branch.
Then it removed the projected radius and reconstructed with the fitting samples'
median radius. The latter is an approximation to the lost magnitude, not the
actual downstream reader or an information-theoretic bound.

| Region | Weak-head argmax agreement after PCA | Agreement after direction-only reconstruction |
| --- | ---: | ---: |
| Haidian | 99.45% | 94.51% |
| Harbin | 99.67% | 94.67% |
| Auckland | 99.79% | 93.48% |

This suggests testing radial information retention before adding more model
capacity. It does not prove that normalization causes the primary task deficit
or that removing it will improve classification.

## One fixed follow-up

Reuse the best common three-branch v2 configuration and its existing sources:
16 learned calendar, 16 frozen visual and 32 public physical components. Change
only branch/final unit-length normalization: preserve all PCA-score magnitudes.
Keep the same regional PCA fitting seed 41, 65,536 sample cap and source data.
The output remains 64D and is not required to have unit norm. The existing reader
still standardizes each coordinate using its labeled valid support pixels only.
Do not describe this descriptor as vMF-trained or intrinsically spherical.

Every region and role uses this one rule. Native and degraded roles retain their
respective local visual source and encoder; public-only uses the genuine public
encoder and full-year public S2 visual source. Public control dimensions and
normalization receive the identical change. No encoder, source data, primary
label, geographic mask, support budget or approved v2 calibration position changes.
The original orderless physical statistics are used; the unsuccessful seasonal
replacement is not substituted into this experiment.

Export nine regional/role representations and fit eighteen fresh readers: all
120 task-reader conditions. Reuse verified existing v2 AEF and public references;
do not rerun those fits or use version 1 values. The public envelope may only
increase when the new matched public control is scored. Keep the six strict
AEF/public wins and each-region 0.5 percentage points mean AEF gain rule. Only an
eligible common candidate receives full-region queries with frozen v2 heads and
thresholds. No automated acceptance flag completes the whole paper goal.

This is a data-preserving fusion comparison. Existing normalization options and
reader code already support it; the experiment does not change package code.
The preceding 996-test/source/numerical gates continue to apply. Exact parameters,
source hashes, cohort identities, all detailed scores and bounded queue records
remain outside Git. No manuscript result has been claimed from this diagnosis.
