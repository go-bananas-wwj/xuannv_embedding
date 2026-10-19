# Time-conditioned local optical views: registered trial

The preceding common semantic residual trial still fails the three-region goal;
the conditional spatial mixer also failed. No region or head is substituted from
a different method. Earlier scores and the approved balanced calibration are kept.

`xuannv experiment time-modulated` reuses the matched residual training engine.
Every actual local observation has a frozen 1024-channel SAT-493M visual map and
fractional coverage. The frozen genuine-public temporal encoder supplies monthly
64-channel maps. Physical day/month/year intervals determine calendar support;
pixel coverage determines which matching public observations can contribute.
Year-only aerial imagery remains one view. Author analysis labels are preserved.
Public RGB controls bind their own actual monthly image support.

Public queries condition a zero-initialized featurewise affine adjustment of local
channels, behind a sigmoid channel gate. A coverage-masked learned attention pools
actual views; the existing detail and public branches remain. The public branch
is recovered exactly when both local branches are absent, including learned
biases. The context normalizer explicitly permits gradients from this new learned
pooling operation; ordinary frozen-context runs retain their detached behavior
and checkpoint parameter names. Standardized output is exposed for the optional
masked variance and covariance loss. This uses moments inspired by
[FiLM](https://arxiv.org/abs/1709.07871) and
[VICReg](https://arxiv.org/abs/2105.04906); it does not implement or claim the full
VICReg twin-view algorithm.

Two common configurations differ only in variance/covariance weights (0/0 and
0.2/0.05). Each is fitted independently in every region with native, public-only
and degraded-local input controls, giving 18 fits and 36 fresh MLP/CNN readers.
All retain the same WorldCover and OSM weak targets, physical reconstruction,
training budget and 64D output. CLCD, DW and LCDB labels are never encoder targets.
Public-reference floors never decrease. Selection uses the worst of six head/
region margins first; all six wins plus each region's mean AEF gain of 0.005 are
required before frozen full-region query evaluation. This is exploratory regional
map agreement, not unseen-area or blind-test generalization.

The trial is registered before new model weights or primary scores. Real maximum-
view and maximum-tile batch/memory gates, three-region recovery/export gates,
regression tests and a pinned source snapshot precede formal fitting. Device 0 is
reserved for the unrelated service. A bounded external controller monitors real
process identities, checkpoints and disk reserve; no chat polling is needed.

Implementation checks: 1094 tests passed; Black, Ruff, repository policy, build
and exact wheel/sdist source checks passed. The real Haidian two-tile BF16 gate
also passed: four updates, recovery after update two, strict cold load and finite
64D export. Maximum resident-memory and remaining regional gates are pending;
these checks establish executability, not downstream improvement.

A pre-training correction intersects the fixed auxiliary context-teacher loss
domain with the teacher's actual coverage. Missing zero vectors are unknown;
new local observations do not turn these into zero-valued reconstruction labels.
The targeted regression fails on the preceding immutable source and passes after
the correction. The updated full suite passes 1097 tests. Original gate outputs
remain archived; new source-bound real gates use a separate output directory.

Monthly public queries and the annual anchor come from separate learned encoders.
Each now uses statistics from its own observed public feature coordinates; query
statistics remain exactly shared across all roles and candidates within a region.
A label-free diagnostic found 17.4–20.0% clipping with unrelated anchor statistics.
This input correction was recorded before formal weights or primary scores.
