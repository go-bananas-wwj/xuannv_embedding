# Calendar encoder: complete original-cohort calibration result

All nine regional/role fits completed the fixed 1,200 updates and visited all
registered tiles. Final weights loaded strictly, and all mean embeddings exported.
The regional manifest verifies the same original public weak-label domain and
class weights across roles in each region. Training success did not establish
classification superiority.

All 120 task-reader conditions completed and passed the existing support, target
and calibration geometry contract against the original baselines. The table
reports macro F1 percentages on the original exploratory calibration cohort.
The public envelope is the strongest genuine public calibration reference retained
under the registered rule, including the current public fit. It is not selected
from full-region query scores.

| Region | Reader | Native local | Matched public | Degraded local | AEF | Public envelope | Native minus AEF (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Haidian | MLP | 71.564 | 65.748 | 69.735 | 75.139 | 73.121 | -3.575 |
| Haidian | CNN | 64.392 | 62.337 | 63.866 | 76.892 | 77.297 | -12.500 |
| Harbin | MLP | 54.198 | 53.366 | 54.283 | 56.120 | 55.979 | -1.922 |
| Harbin | CNN | 57.323 | 58.129 | 58.884 | 58.168 | 58.288 | -0.845 |
| Auckland | MLP | 53.029 | 52.722 | 52.519 | 58.045 | 58.244 | -5.016 |
| Auckland | CNN | 53.788 | 53.675 | 53.356 | 59.314 | 57.381 | -5.526 |

The common candidate fails all six AEF comparisons and the overall selection gate.
No full-region query was launched for this candidate. Native local input improves
over the matched public model in five slots, with Harbin CNN declining. Compared
with degrading the same local imagery to 10 m, native imagery improves both
readers in Haidian and Auckland but declines in both Harbin readers. Thus this
experiment does not establish consistent three-region high-resolution gains.

The full reference, per-task F1/AP/IoU and boundary metrics, support identities,
training configurations, weights and provenance remain in the external run
records. No negative task or comparator was dropped. None of these results is a
blind test, same-weight unseen-region transfer, or proof of the paper objective.

Proceed with the single registered [visual and observation skip](temporal-visual-skip-experiment.md).
Reuse these fixed-budget encoders without another encoder training, keep the
original cohort and all three roles, and evaluate all twenty tasks with both
readers. Changes to calibration sampling remain unapproved and unapplied.
Original best model bundles and manuscript introduction remain preserved. No
paper body was changed to claim this unsuccessful experiment as a win.
