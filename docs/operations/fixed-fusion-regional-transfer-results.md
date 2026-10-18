# Fixed regional fusion: completed primary Haidian comparison

This checkpoint reports the complete ten-task Haidian primary set. CLCD 2025 and
the Harbin comparisons remain in progress. No destination scores were used to
choose a new recipe. F1 values below are averages of binary task F1 scores, not
Auckland multiclass Macro F1. Full-region scoring includes support/calibration.

| Representation | MLP F1 % | CNN F1 % |
|---|---:|---:|
| AEF | 57.1550 | 58.3784 |
| Retained regional high-resolution base | 59.4301 | 59.7433 |
| Frozen DINO only | 48.8243 | 52.4409 |
| 40+24 fusion, no context | 57.5266 | 60.2170 |
| 40+24 fusion, 130 m affinity context | 53.2944 | 56.5622 |
| Regional base + matched context | 56.5577 | 56.0208 |
| AEF + matched context | 53.0186 | 53.0068 |

Local fusion changes CNN by +0.4737 pp and MLP by -1.9030 pp versus the original
base (rounded; complete task-level records remain in the experiment workspace).
The Auckland 130 m fusion recipe decreases both primary readouts. Matched
context also decreases the original AEF/base scores, so context cannot be assumed
to improve fine urban mapping. This is a fixed-recipe transfer result, not
evidence of universal improvement from DINO or neighborhood processing.

The actual source information remains approximately 3 m, sampled to a 0.5 m
encoder grid without adding observed detail. Both regions passed the ordered
shard/native gate comparison exactly (maximum feature difference zero). All
negative conditions remain available. The directory-initialization recovery
changed no feature values, support positions or readout algorithms. The related
regression and full suite passed: 887 tests, Black, Ruff, repository gate, build.
