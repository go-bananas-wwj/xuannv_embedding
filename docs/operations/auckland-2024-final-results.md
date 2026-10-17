# Auckland west 2024: completed paired experiment

Both encoders and all registered evaluations are complete. Training finished on 2026-10-17 at 07:01 (public) and 13:18 (LINZ), Beijing time. Export, 215 classification fits, five regression feature families and visualizations finished at 13:36. The Watchdog completed normally with zero training restarts. No Overleaf changes were made.

The primary classification result does not establish an advantage over AEF: annual-mean MLP Macro F1 at 200 labels/class is 57.00% for AEF, 54.37% for the public embedding and 52.92% for the LINZ embedding. Adding aerial RGB reduces this score by 1.45 percentage points relative to the public model. Height regression also trails AEF. These are single encoder-seed results; spatial folds do not substitute for repeated encoder training.

## Protocol

Two encoders were trained from scratch on the same 320 Auckland-west tiles, with seed 41, 5,334 optimizer updates and effective batch 48. Public inputs are twelve monthly 2024 Sentinel-2, Sentinel-1 and Landsat composites. LINZ additionally uses one static aerial RGB observation at approximately 3 m resolution. Both use WorldCover and OSM weak supervision; LCDB and LiDAR are reserved for downstream evaluation.

The encoder sees the whole study area (transductive). Readout labels are held out using five spatial folds, with every tile evaluated once; supports, valid masks, folds and validation selection are shared. LCDB v6 2023/24 has eleven eligible classes. Primary embedding representation is the normalized twelve-month mean. Regression errors below are pooled from absolute/squared error sums over the same held-out pixels, rather than averaging fold RMSEs.

## Main LCDB classification

200 training-label points per class; all numbers are percentages. RF and U-Net consume temporal statistics, with RGB statistics added for the raw high-resolution baseline; they are practical baselines rather than matched encoder architectures.

| Feature | Readout | Macro F1 | mIoU | Accuracy |
|---|---|---:|---:|---:|
| AEF 2024 | Linear | 56.64 | 47.65 | 74.72 |
| AEF 2024 | MLP | 57.00 | 48.03 | 74.35 |
| AEF 2024 | CNN | 56.11 | 47.34 | 74.42 |
| Public embedding | Linear | 50.23 | 43.10 | 72.87 |
| Public embedding | MLP | 54.37 | 46.32 | 74.86 |
| Public embedding | CNN | 53.53 | 45.32 | 73.50 |
| LINZ embedding | Linear | 48.89 | 41.96 | 71.68 |
| LINZ embedding | MLP | 52.92 | 45.24 | 74.63 |
| LINZ embedding | CNN | 51.46 | 43.48 | 71.94 |
| Public temporal statistics | Random forest | 51.76 | 42.37 | 69.57 |
| Public temporal statistics | U-Net | 46.53 | 37.61 | 61.85 |
| Public + aerial statistics | Random forest | 52.47 | 43.01 | 70.49 |
| Public + aerial statistics | U-Net | 47.11 | 37.99 | 65.10 |

## Label-budget sensitivity

Macro F1 (%). Budgets are requested points/class, capped by eligible spatial support. All methods use the same registered support locations.

| Feature | Readout | 50 | 200 | 1,000 |
|---|---|---:|---:|---:|
| AEF 2024 | Linear | 53.53 | 56.64 | 58.50 |
| AEF 2024 | MLP | 54.31 | 57.00 | 58.63 |
| AEF 2024 | CNN | 52.53 | 56.11 | 59.72 |
| Public embedding | Linear | 47.97 | 50.23 | 50.97 |
| Public embedding | MLP | 52.46 | 54.37 | 53.64 |
| Public embedding | CNN | 51.09 | 53.53 | 55.35 |
| LINZ embedding | Linear | 46.58 | 48.89 | 49.38 |
| LINZ embedding | MLP | 51.56 | 52.92 | 52.59 |
| LINZ embedding | CNN | 49.49 | 51.46 | 53.87 |
| Public temporal statistics | Random forest | 47.68 | 51.76 | 54.40 |
| Public temporal statistics | U-Net | 40.18 | 46.53 | 50.11 |
| Public + aerial statistics | Random forest | 48.11 | 52.47 | 55.17 |
| Public + aerial statistics | U-Net | 39.95 | 47.11 | 51.08 |

## Temporal representations

MLP, 200 labels/class; Macro F1 (%). Annual mean remains the registered primary result.

| Feature | April | March–May mean | Annual mean |
|---|---:|---:|---:|
| Public embedding | 51.23 | 53.17 | 54.37 |
| LINZ embedding | 49.69 | 51.80 | 52.92 |

## Per-class F1

Annual mean, MLP, 200 labels/class; values and differences are percentage points. All eligible classes are reported, including declines.

| LCDB class | AEF | Public | LINZ | Public − AEF | LINZ − AEF |
|---|---:|---:|---:|---:|---:|
| Built-up Area (settlement) | 92.70 | 91.33 | 90.60 | -1.38 | -2.11 |
| Urban Parkland/Open Space | 54.15 | 42.37 | 36.37 | -11.79 | -17.78 |
| Estuarine Open Water | 99.07 | 98.96 | 98.94 | -0.11 | -0.12 |
| Short-rotation Cropland | 6.09 | 7.00 | 6.97 | +0.92 | +0.88 |
| Orchard, Vineyard or Other Perennial Crop | 16.99 | 11.98 | 11.57 | -5.01 | -5.42 |
| High Producing Exotic Grassland | 66.83 | 69.53 | 68.77 | +2.69 | +1.94 |
| Manuka and/or Kanuka | 66.78 | 66.41 | 65.23 | -0.38 | -1.55 |
| Broadleaved Indigenous Hardwoods | 16.34 | 9.70 | 8.31 | -6.64 | -8.04 |
| Indigenous Forest | 89.03 | 89.30 | 89.45 | +0.27 | +0.42 |
| Mangrove | 95.56 | 96.92 | 96.99 | +1.36 | +1.43 |
| Exotic Forest | 23.44 | 14.62 | 8.94 | -8.82 | -14.51 |

## LiDAR regression

Ridge regression, same 5,000 training points per fold. Surface height includes vegetation and buildings; it is not building height alone. Lower errors are better.

| Feature | Target | MAE | RMSE | Units |
|---|---|---:|---:|---|
| AEF 2024 | Terrain elevation | 22.477 | 31.388 | m |
| AEF 2024 | Terrain slope | 3.550 | 5.013 | degrees |
| AEF 2024 | Mean surface height | 2.236 | 3.336 | m |
| AEF 2024 | P90 surface height | 3.002 | 4.341 | m |
| Public embedding | Terrain elevation | 39.742 | 55.184 | m |
| Public embedding | Terrain slope | 4.915 | 6.782 | degrees |
| Public embedding | Mean surface height | 2.570 | 3.770 | m |
| Public embedding | P90 surface height | 3.373 | 4.822 | m |
| LINZ embedding | Terrain elevation | 41.295 | 57.360 | m |
| LINZ embedding | Terrain slope | 4.928 | 6.781 | degrees |
| LINZ embedding | Mean surface height | 2.601 | 3.825 | m |
| LINZ embedding | P90 surface height | 3.407 | 4.882 | m |
| Public temporal statistics | Terrain elevation | 48.238 | 70.865 | m |
| Public temporal statistics | Terrain slope | 5.484 | 8.168 | degrees |
| Public temporal statistics | Mean surface height | 2.890 | 5.143 | m |
| Public temporal statistics | P90 surface height | 3.754 | 6.453 | m |
| Public + aerial statistics | Terrain elevation | 46.661 | 69.467 | m |
| Public + aerial statistics | Terrain slope | 5.473 | 7.926 | degrees |
| Public + aerial statistics | Mean surface height | 2.883 | 4.966 | m |
| Public + aerial statistics | P90 surface height | 3.729 | 6.388 | m |

## Artifacts and interpretation

Machine-readable results, selected weights, embeddings and the three figures remain in the data workspace; none is committed to Git. Figures cover land-cover maps, per-class F1 differences and surface-height maps. Historical LINZ building references are diagnostic only because their acquisition dates do not establish 2024 truth.

The current experiment supports neither an overall New Zealand advantage over AEF nor a general benefit from adding LINZ imagery with this configuration. Some individual classes improve, while urban parkland, exotic forest and broadleaved hardwood classes account for substantial deficits. A causal explanation requires further diagnostics; this report does not change targets, select new test-driven training settings or start another run.

AEF attribution: The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind (CC-BY 4.0).
