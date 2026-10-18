# AEF goal: wave one completed, goal remains active

Seventeen predeclared 64-D representations were screened with fold-zero training/validation labels. Selection maximized the smaller of MLP/CNN validation margins over the corresponding AEF validation score. The selected method is `semantic_blocks_16`: 48 public PCA coordinates plus 16 aerial DINOv3 SAT-493M PCA coordinates, normalized as branch vectors and concatenated. Projections use unlabeled regional features only.

## Primary pooled five-fold test

| Representation | MLP Macro F1 (%) | CNN Macro F1 (%) |
|---|---:|---:|
| AEF | 57.000 | 56.106 |
| Public | 54.373 | 53.527 |
| Public + aerial semantics | 56.634 | 55.404 |

The candidate improves over the public reference by 2.261 / 1.877 percentage points for MLP / CNN. It remains below AEF by 0.365 / 0.702 points. The registered goal is not achieved and has not been marked complete.

## Supporting label-budget results

These settings were not used for model selection. Same nested support locations and baseline protocol.

| Labels/class | AEF MLP | Candidate MLP | AEF CNN | Candidate CNN |
|---|---:|---:|---:|---:|
| 50 | 54.312 | 53.757 | 52.533 | 50.321 |
| 200 | 57.000 | 56.634 | 56.106 | 55.404 |
| 1000 | 58.632 | 59.033 | 59.723 | 58.152 |

## Attribution control: validation only

Using the exact selected public 48-D projection, padding its remaining 16 coordinates with zeros, gives validation MLP/CNN F1 of 57.116% / 56.570%. The selected fusion gives 59.330% / 56.319%. Thus semantics improves MLP validation over this projection control but slightly lowers CNN validation; some improvement over the original public embedding is due to projection itself. This posthoc diagnostic was not used for wave-one selection and is not a separate full-area test.

## All validation candidates

| Recipe | MLP F1 (%) | CNN F1 (%) | Smaller margin vs AEF (pp) |
|---|---:|---:|---:|
| blend_prior_050 | 54.164 | 54.438 | -2.851 |
| blend_prior_025 | 54.349 | 55.111 | -2.666 |
| blocks_prior_32 | 55.966 | 52.195 | -4.649 |
| blend_long_025 | 54.227 | 54.405 | -2.788 |
| blend_long_050 | 54.038 | 52.630 | -4.214 |
| blocks_long_32 | 56.234 | 54.076 | -2.768 |
| blend_frozen_025 | 54.324 | 54.295 | -2.691 |
| blend_frozen_050 | 53.965 | 53.819 | -3.050 |
| blocks_frozen_32 | 56.536 | 54.678 | -2.166 |
| blend_anchor_025 | 54.356 | 53.029 | -3.814 |
| blend_anchor_050 | 54.398 | 52.196 | -4.648 |
| blocks_anchor_32 | 56.031 | 54.016 | -2.828 |
| semantic_blocks_16 | 59.330 | 56.319 | -0.525 |
| semantic_blocks_32 | 58.627 | 56.148 | -0.696 |
| semantic_blocks_48 | 58.016 | 54.857 | -1.987 |
| semantic_joint64 | 54.602 | 52.940 | -3.904 |
| semantic_only64 | 50.094 | 48.753 | -8.091 |

Semantic-only is a diagnostic, ineligible as a regional-fusion goal winner. It performs worse than public on validation, so this is not simply a better external encoder replacing the local model. DINOv3 is additional global pretraining and must be disclosed.

## Next registered direction

Prepare actual 1 m crops from the same 2024 LINZ acquisition, rather than upsampling the existing 3 m mosaic as if it added information. Patch size 16 corresponds to about 48 m at the first input resolution; stronger native resolution may preserve details that are currently lost. Use bounded overlapping encoder windows, explicit crop geometry/masks, source hashes and unchanged 10 m output/scoring grids. The hypothesis is unverified until evaluated. Smaller semantic allocations are also reasonable validation-only refinements.

All class/regression outputs and negative results remain in the data workspace. This is adaptive analysis of an already inspected regional benchmark, not a new blind evaluation. No Overleaf changes.
