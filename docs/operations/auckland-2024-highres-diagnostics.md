# Auckland 2024: urban visualizations and high-resolution diagnostics

The completed trial adds three inspection figures: urban aerial/legacy-footprint/built-up
comparisons, urban P90 surface-height comparisons, and an eleven-class F1 chart with Chinese
labels. The two urban scenes are tiles 187 and 58, already selected by reference class counts
in the registered visualization procedure. Predictions did not determine scene selection.
Existing held-out predictions are reused; no encoder training or readout fitting was repeated.
Figures and their rendering script remain in the data workspace rather than Git.

## Task definitions and observed results

LCDB built-up classification includes settlement areas and does not measure individual building
footprint segmentation. Historical LINZ footprints date to 2017/2022; their patch coverage and
density diagnostics cannot establish 2024 footprint accuracy. Surface height includes both trees
and buildings, rather than building height alone.

| Metric | AEF | Public embedding | LINZ embedding |
|---|---:|---:|---:|
| Built-up F1 (%) | 92.70 | 91.33 | 90.60 |
| Urban parkland/open-space F1 (%) | 54.15 | 42.37 | 36.37 |
| Exotic-forest F1 (%) | 23.44 | 14.62 | 8.94 |
| Historical building coverage MAE (percentage points, lower better) | 0.872 | 0.653 | 0.761 |
| Historical building density MAE (buildings/km2, lower better) | 79.31 | 74.19 | 79.05 |
| Mean surface-height MAE (m, lower better) | 2.236 | 2.570 | 2.601 |

F1 uses the annual mean, MLP, 200 training labels/class and pooled held-out confusion matrices.
Historical building MAEs are the mean over five equal-size 64-tile test folds. Height MAE is
pooled over valid held-out pixels. The building coverage diagnostic improves over AEF for both
regional models, while the public model remains better than LINZ. It would be incorrect to
describe every downstream task as worse than AEF.

## Checks completed

- All 320 cached high-resolution masks were inspected: 305 tiles are fully valid; the remaining
  15 are partially or wholly unavailable. The median and fifth-percentile valid fractions are
  both 1.0. Widespread missing RGB does not explain the overall deficit, although individual
  missing tiles still matter.
- The preparation audit checked 114 raster headers with zero bounds discrepancy in EPSG:2193.
  This verifies grid extents, not measured subpixel image alignment.
- Final checkpoints register the same production source revision and 5,334 scheduled updates.
  LINZ has its own high-resolution optimizer group. Injection output matrices are finite and
  nonzero (RMS approximately 0.01071 and 0.00659 at blocks 2 and 4); both were zero-initialized.
  The high-resolution pathway therefore did learn parameters. This does not establish that
  the learned features are useful.
- The active regional builder re-enables gradients on the public backbone. The wrapper's
  historical `freeze_base` constructor argument does not mean this regional trial froze the
  backbone; both models trained from the shared untrained initialization.
- Registered objectives reconstruct public satellite inputs and use WorldCover/OSM weak
  semantics. There is no aerial-RGB reconstruction head or independent building-boundary
  objective. LiDAR and LCDB are evaluation-only references.
- Aerial RGB is static and shared as encoded context across twelve monthly queries. Input
  preparation uses approximately 3 m imagery and downstream embeddings are on a 10 m grid.

## Interpretation and next diagnostic priorities

The experiment establishes a performance decrease for this fusion/training configuration;
it does not isolate a causal explanation. The main hypotheses are insufficient supervision
for useful aerial detail, interference from joint training of the backbone and added branch,
and a mismatch between local appearance and coarse land-use/forest categories. Coarsening
the aerial input and reading 10 m outputs also limits what can be claimed about small roofs.
Single-seed training variation remains unmeasured.

Raw-statistics RF improves from 51.76% to 52.47% Macro F1 when aerial statistics are added.
This suggests some available RGB information, but does not prove the neural fusion mechanism
is the cause: the feature representations and readouts differ.

Useful next diagnostics would measure local image registration and learned gate/feature
behavior, then compare inference with and without RGB using the same fixed checkpoint.
If further training is approved, preserving a trained public backbone while introducing the
RGB branch would help separate fusion interference from information benefit. Independent
footprint evaluation requires labels with a declared reference year. These are proposed
diagnostics, not additional experiments started in this turn. No Overleaf changes were made.
