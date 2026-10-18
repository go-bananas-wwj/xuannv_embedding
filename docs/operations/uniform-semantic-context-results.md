# Uniform semantic-context result and shared variance budget

Three public-only regional models completed the same1,200-update training rule,
visited all320/422/320 registered tiles and exported verified auxiliary
probabilities. All nine fusion arrays use identical public-only probabilities
between each region's native, public and degraded roles. Context provenance,
source hashes, 64D outputs and all120 task-reader sampling contracts were audited.

Calibration macro-F1 in percent:

| Region | Reader | Native high resolution | Matched public | Same local images at10m | AEF |
| --- | --- | ---: | ---: | ---: | ---: |
| Haidian | MLP | 70.965 | 73.121 | 70.112 | 75.139 |
| Haidian | CNN | 77.149 | 77.297 | 76.651 | 76.892 |
| Harbin | MLP | 55.362 | 53.807 | 53.102 | 56.120 |
| Harbin | CNN | 59.887 | 56.756 | 55.980 | 58.168 |
| West Auckland | MLP | 56.782 | 51.811 | 51.152 | 58.045 |
| West Auckland | CNN | 57.950 | 52.246 | 52.424 | 59.314 |

This candidate fails the common-method criterion. Haidian improves substantially
over its observation-only predecessor, but its MLP remains below AEF and both
heads remain below the matched public representation. Harbin's favorable CNN
does not offset its MLP deficit. Auckland also remains below AEF. No new
full-region queries were run. Preserve all tasks and both readers.

The newly trained semantic models fitted their WorldCover training maps; a fixed
twenty-tile diagnostic gave training accuracies79.82%,75.04%and90.77%respectively.
Those are training-map diagnostics, not downstream validation or independent
generalization results. They must not be substituted for the primary scores.

The eleven probability coordinates have substantial redundancy:

| Region | Variance retained by first4 coordinates | Variance retained by first6 coordinates |
| --- | ---: | ---: |
| Haidian | 99.7094% | 99.9997% |
| Harbin | 94.0479% | 99.9967% |
| West Auckland | 99.3996% | 99.9739% |

Preregister one inexpensive follow-up before its scores: choose the smallest
**single common** width that retains at least99.9%of the semantic probability
variance in every region. The existing label-free projection spectra select
six dimensions. Use the same neural weights and observations; return the freed
five dimensions to the annual statistics. Native/degraded are then
public-RGB16/local-RGB16/annual22/terrain4/context6, while public controls use
public-RGB32/annual22/terrain4/context6. All remain64D with the same normalization
and reader rules. Do not choose a different width by region or task.

Variance retention does not guarantee task accuracy, especially for rare classes.
The dimension decision uses projection spectra, not primary labels, and all
twenty primary classes remain scored. Keep the previous eleven-coordinate
public results in the reference envelope. Only a common eligible candidate can
proceed to new full-region queries. This follow-up requires no neural retraining.
