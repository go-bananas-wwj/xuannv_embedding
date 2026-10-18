# Fixed regional fusion: completed two-region results

All 44 registered readout jobs and 436 binary task/head/model conditions completed.
Support descriptors and scored observation counts match across methods. Both
ordered-shard native-feature checks reproduce the original two-tile gates exactly.

Values are mean binary task F1 percentages. Each head uses the original twenty
support tiles, fixed 100-update budget, seed and calibration pool. These figures
are not numerically interchangeable with Auckland multiclass Macro F1. Full-region
scoring includes downstream support/calibration locations. OSM and WorldCover
may have been used in regional encoder training; CLCD 2025 and Dynamic World
are separate references, not encoder targets. This remains exploratory mapping
agreement on previously inspected regions, not a new blind evaluation.

## Complete registered primary comparisons

### Haidian: ten OSM/WorldCover tasks

| Representation | MLP F1 % | CNN F1 % |
|---|---:|---:|
| AEF | 57.1550 | 58.3784 |
| Original regional base | 59.4301 | 59.7433 |
| Frozen DINO only | 48.8243 | 52.4409 |
| 40+24 fusion, no context | 57.5266 | 60.2170 |
| 40+24 fusion, 130 m affinity context | 53.2944 | 56.5622 |
| Regional base + matched context | 56.5577 | 56.0208 |
| AEF + matched context | 53.0186 | 53.0068 |

### Haidian: four CLCD 2025 tasks

| Representation | MLP F1 % | CNN F1 % |
|---|---:|---:|
| AEF | 82.4999 | 83.0672 |
| Original regional base | 76.1696 | 81.2016 |
| Frozen DINO only | 71.0792 | 78.3268 |
| 40+24 fusion, no context | 79.2880 | 80.2710 |
| 40+24 fusion, 130 m affinity context | 79.8556 | 80.0305 |
| Regional base + matched context | 77.3428 | 80.1645 |
| AEF + matched context | 82.3807 | 82.0515 |

### Harbin: fifteen OSM/WorldCover/Dynamic World tasks

| Representation | MLP F1 % | CNN F1 % |
|---|---:|---:|
| AEF | 58.0175 | 60.0462 |
| Original regional base | 60.2757 | 60.2791 |
| Previous regional high-resolution model | 59.7834 | 59.5173 |
| Frozen DINO only | 53.2272 | 58.5546 |
| 40+24 fusion, no context | 60.7538 | 63.0266 |
| 40+24 fusion, 130 m affinity context | 59.2198 | 60.5813 |
| Regional base + matched context | 57.6846 | 57.7496 |
| AEF + matched context | 55.8632 | 56.2148 |

### Harbin: Dynamic World only

| Representation | MLP F1 % | CNN F1 % |
|---|---:|---:|
| AEF | 57.8339 | 60.8573 |
| Original regional base | 55.1580 | 55.5167 |
| Previous regional high-resolution model | 55.3326 | 55.8663 |
| Frozen DINO only | 52.7094 | 59.3387 |
| 40+24 fusion, no context | 56.4435 | 59.1824 |
| 40+24 fusion, 130 m affinity context | 56.8617 | 58.7206 |
| Regional base + matched context | 53.2835 | 54.8096 |
| AEF + matched context | 56.1165 | 57.4414 |

## Interpretation

The Haidian base already includes regional high-resolution optical/SAR training;
the Harbin fusion base is the retained public-only model. The retained Harbin
high-resolution model is a separate control. Consequently the difference between
regional gains cannot be attributed solely to geography or image resolution.
DINOv3 SAT-493M adds frozen global satellite pretraining. No region encoder was
retrained, and the three protected original bundles remain hash-identical.

The selected Auckland 130 m recipe is not a universal replacement. Its registered
no-context ablation is better in both regions. Local fusion changes primary F1
relative to the original base by:

- Haidian: MLP -1.9034 pp; CNN +0.4738 pp.
- Harbin: MLP +0.4781 pp; CNN +2.7475 pp.

Harbin local fusion improves all three source-group CNN averages over its public
base; neither fusion reading exceeds original AEF on the Dynamic World average.
Haidian CLCD MLP improves over the original base, while CNN decreases and both
remain below AEF. No superiority across all datasets, tasks or heads is established.
The DINO-only controls and all unfavorable conditions remain in the tables.

The two components may be complementary: Harbin OSM CNN averages are 63.8557%
for public features, 57.8877% for DINO alone and 67.6268% for local fusion. These
observations support complementarity but do not isolate every effect of PCA,
normalization, source quality or finite readout optimization. In particular, the
40D public projection retains over 99.8% of unlabeled variance in both regions;
variance retention alone does not demonstrate preservation of task information.

The next inexpensive validation would compare smaller/no spatial neighborhoods
and less destructive fusion against the unchanged original representation. A
residual or gated high-resolution branch is a candidate, not a demonstrated win.
Simply increasing regional encoder training duration is not supported by this run.

## Reproduction and recovery

Execution uses the immutable numerical implementation archive of `9d536cb`. A
job-directory initialization fix and failing-then-passing regression are committed
in `3b84804`; creating the expected runtime directories allowed the original
feature job to resume without recomputing native features or changing scores.
The legacy Harbin registry received a separate geometry-complete copy, verified
against retained patch IDs/bounds, with label/mask/support arrays unchanged.
Preparation errors and scheduling amendments are retained, not silently erased.

887 tests, Black, Ruff, repository gate and package build passed after the fix.
Configuration, raw receipts, projections, heads, predictions, paired-support
audits, all per-task F1/AP/IoU values and negative results stay outside Git.
All 436 conditions are retained in the experiment CSV and full result reports.
No Overleaf edits occurred. No further training or parameter search is queued.
