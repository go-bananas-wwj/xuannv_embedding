# Time-guided result and next shared spatial/geometry trial

The full time-guided study finished eighteen regional fits and thirty-six readers,
covering all240 registered binary conditions. An independent saved-prediction
check reproduced every F1/IoU, confirmed identical AEF support/initialization/update
schedules and verified the global selection rule. Neither shared candidate passed.
All negative scores, weights, observation identities and old reference products
remain available. No full-region query or Overleaf success claim was triggered.

| Region | Better common time-guided native MLP | Native CNN | AEF MLP | AEF CNN |
| --- | ---: | ---: | ---: | ---: |
| Haidian | 0.847731 | 0.852542 | 0.838754 | 0.843817 |
| Harbin | 0.577809 | 0.594525 | 0.583955 | 0.587958 |
| West Auckland | 0.621518 | 0.650611 | 0.634338 | 0.646834 |

The stronger public envelope is retained. Its CNN scores are0.854545,0.607554 and
0.647244 respectively; new public MLP references are0.846595,0.579986 and0.633369.
Thus beating AEF in four reader/region slots does not satisfy the common-method
goal. Urban parkland and manuka/kanuka remain important Auckland MLP gaps; grass,
trees and crops remain Harbin MLP gaps. The complete task list is retained.

The preceding always-on spatial mixer improved Auckland but harmed other regions.
The next preregistered trial replaces that compulsory update with a zero-projection
spatial correction behind a learned sigmoid gate. The same module, initialization
and rule apply in all regions. A second common setting also preserves pairwise
local feature similarities, emphasizing distinctions inside the same known coarse
WorldCover class. The fixed teacher uses the full1024 visual coordinates, rather
than only a64D PCA reconstruction target. This is motivated by the still-large
normalized visual reconstruction losses and fine-category gaps, not proof of a
causal explanation for the failed scores.

The cosine-Gram loss is inspired by
[relational knowledge distillation](https://arxiv.org/abs/1904.05068); it is not the
original paper's distance/angle loss. Teachers are detached and label-free visual
features. Only already-authorized WorldCover weak classes weight pairs; primary
CLCD/DW/LCDB references remain downstream-only. Relational statistics explicitly
disable autocast. The earlier variance regularizer was observed to return BF16
under training autocast, so it must not be described as fully FP32; that completed
study was not silently changed or relabeled.

Both new settings retain the same observations, genuine public anchor, original
weak-map objectives,64D static output, independent regional learning and B64/2000
update budget. Native/public/degraded roles stay matched. Selection still requires
all six wins against AEF and the nondecreasing public envelope, plus each region's
mean AEF gain of0.005. New source tests, three-region recovery/export checks and
actual saved-warmup B64 gates precede formal fitting. The paper body stays unchanged.

A separate rebuild-dependency inventory protects small fitted transforms and
provenance while the trial runs. The public64D anchor is a fixed16+16+32 block
projection of public temporal, public visual and physical-statistic features;
its block projections are required to reconstruct the original coordinates.
The inventory also retains fine/context visual PCA parameters, public temporal
checkpoints, source recipes, radiometry, geographic and physical-time indices.
Twenty-seven fitted PCA files were found and preserved. Frozen global visual
weights remain in their existing protected location; their producer hash is
recorded, with final-bundle hash verification still required. Raw observations,
existing weights and arrays were not deleted. This prepares reproducibility;
it is not an accepted-model release or performance evidence.
