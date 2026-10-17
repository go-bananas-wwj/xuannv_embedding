# Auckland 2024: complete CNN breakdown

All eleven eligible LCDB classes are included below. These are class breakdowns of one land-cover dataset, not eleven independent downstream datasets. The full-area visualization combines test-only predictions from the five registered spatial folds across all 320 tiles; it is not a prediction map from a single downstream head.

All models use annual embeddings, the same CNN architecture and the same registered label locations. No training or evaluation was repeated to make these figures. Eligibility and excluded categories remain recorded in the prepared evaluation metadata.

## Per-class F1 at 200 labels/class

| Class | AEF | Public | LINZ | LINZ minus public |
|---|---:|---:|---:|---:|
| Built-up Area (settlement) | 93.11 | 92.25 | 92.13 | -0.13 |
| Urban Parkland/Open Space | 55.49 | 40.26 | 27.34 | -12.92 |
| Estuarine Open Water | 98.96 | 98.55 | 98.81 | +0.26 |
| Short-rotation Cropland | 7.52 | 4.35 | 7.97 | +3.62 |
| Orchard, Vineyard or Other Perennial Crop | 14.77 | 13.16 | 14.23 | +1.07 |
| High Producing Exotic Grassland | 69.30 | 68.96 | 59.31 | -9.64 |
| Manuka and/or Kanuka | 63.40 | 65.98 | 62.57 | -3.41 |
| Broadleaved Indigenous Hardwoods | 11.47 | 10.06 | 6.37 | -3.69 |
| Indigenous Forest | 86.98 | 85.95 | 87.74 | +1.79 |
| Mangrove | 95.54 | 95.49 | 95.14 | -0.35 |
| Exotic Forest | 20.63 | 13.79 | 14.44 | +0.65 |

## Macro F1 by label budget

| Requested labels/class | AEF | Public | LINZ |
|---|---:|---:|---:|
| 50 | 52.53 | 51.09 | 49.49 |
| 200 | 56.11 | 53.53 | 51.46 |
| 1000 | 59.72 | 55.35 | 53.87 |

## Spatial and semantic interpretation

Each fold uses 192 training, 64 validation and 64 test tiles. Adjacent 2x2 tiles stay together; the five groups are distributed blocks, not five contiguous large subregions. The encoder has seen imagery from all tiles. Pixel labels are spatially held out; this does not establish generalization to imagery unseen by the encoder.

Full-area map grey regions are excluded from scoring: patch borders, LCDB class boundaries, unsupported classes or invalid observations. All four maps use the same mask and colors. Built-up is settlement land cover rather than individual building footprints. The area is 20.48 by 25.60 km in EPSG:2193.

Figures, rendering source and a complete 99-row CSV (three models x three budgets x eleven classes, including precision/recall/F1/IoU/support) are retained in the data workspace. Runtime arrays and images are not committed. No Overleaf changes or new training were started.
