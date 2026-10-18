# Manuscript alignment before a three-region claim

The current Overleaf remote was freshly fetched and read at commit
`a63fa74d5c4ed9e23c92d287530daa6a6a35ff44`. No manuscript working files were
changed. The current methods, consolidated experiments and discussion describe
the earlier neural regional models; their results must not be silently assigned
to the new observation-only fusion candidate.

| Existing manuscript element | Current shared-method experiment | Requirement before updating the paper |
| --- | --- | --- |
| Public and local observations form a reusable 64D regional representation | Three regions use one 64D fusion rule and separate regional projection fits | Verify one recipe passes every regional acceptance slot |
| Monthly neural outputs averaged for static mapping | The current candidate directly constructs a static descriptor from image encoders and temporal statistics | Describe the actual static descriptor; do not call it a mean of monthly embeddings |
| Learned attention, reconstruction, map supervision and uniformity losses | Frozen satellite visual weights and observation-only PCA fits | Rewrite the method if selected; do not retain training objectives that were not used |
| Local optical and local SAR branches | The current candidate adds local RGB, with public S1 represented in the temporal statistics | Distinguish public radar from local SAR; do not claim a tested local-SAR contribution |
| Two-region averages over eight classifier/budget conditions | Three-region primary classification uses fixed MLP/CNN, twenty support tiles and one hundred updates | Report the new protocol explicitly; do not combine unlike averages |
| OSM, WorldCover and selected external applications | Primary references are CLCD2025, DW2025 and LCDB2023/24, none used as regional projection targets | Retain all twenty registered primary tasks and both heads |
| Historical neural training cost versus AEF global pretraining | Frozen visual pretraining plus regional image extraction, projection and readout costs | Measure the selected pipeline and disclose inherited global pretraining; do not reuse the old neural training ratios |
| Earlier retrieval, regression, object and map figures | These belong to different historical representations | Regenerate for the final selected representation or remove/label the historical evidence |

The original introduction and related work remain the starting point. No new
universal-win language, method figure or performance table is justified by this
audit. Match all new claims to the final selected model's actual input graph,
geographic grid, fitting rule and result artifacts.

The locked primary classification inventory is:

| Region | Registered task identifiers |
| --- | --- |
| Haidian, CLCD2025 | `clcd_crop`, `clcd_forest`, `clcd_water`, `clcd_impervious` |
| Harbin, DW2025 | `dw_trees`, `dw_grass`, `dw_crops`, `dw_built`, `dw_bare` |
| West Auckland, LCDB2023/24 | `lcdb_1`, `lcdb_2`, `lcdb_22`, `lcdb_30`, `lcdb_33`, `lcdb_40`, `lcdb_52`, `lcdb_54`, `lcdb_69`, `lcdb_70`, `lcdb_71` |

These inventories are fixed by the registered datasets before this candidate's
scores. Harbin's data bundle also contains historical OSM and WorldCover tasks;
the primary comparison explicitly selects its five DW tasks. This is not a
post-score removal of unsuccessful categories. Additional primary categories
would require a separately registered comparison for every representation.

Keep support and calibration pools, pixels and original seeds identical to the
fixed baselines. Full-region queries include support and calibration locations;
the repeatedly inspected comparisons are exploratory regional map agreement.
Do not describe them as an independent held-out test or direct transfer of the
same fitted weights to an unseen region.
