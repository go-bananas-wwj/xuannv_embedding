# Independent land-cover product evaluation

`xuannv experiment prepare-native-landcover --spec SPEC.json` aggregates
already exported features onto a categorical reference's native grid. The
encoder and its historical artifacts remain frozen. The new product labels
are only consumed by downstream predictors.

The specification registers parent cohort/shared arrays, a source raster and
its provenance by SHA256, original feature CRS, product/year/class legend,
storage window size, support budgets/seeds, and a new output directory.

Features use area-average reprojection. Labels are read directly from native
cells without interpolation. Cells crossing an original tile boundary or
containing any invalid feature context are excluded. Every retained native
cell has exactly one original tile owner; duplicates raise an error. The
original train/calibration/query/buffer assignment is preserved.

Fixed square arrays may include masked padding. Their dimensions are storage
dimensions, not an increase in labeled support area. Missing context is masked
for all representations identically. The audit records native-cell counts,
resolution, CRS, class availability and original source identities.

Task classes must have enough eligible training tiles for the largest requested
support budget, plus both classes in calibration. This list is locked before
query product labels are read. Query-only classes cannot alter eligibility.
The generic primary cohort requires C/R/Q task schemas, but only explicitly
scheduled families should be reported; preparing a schema does not execute a
regression or retrieval experiment.

For 30-metre products, a boundary tolerance of one output pixel means 30 metres,
not the 10 metres used in the original benchmark. Report target and feature
years separately. Agreement with a historical automated land-cover product is
not equivalent to contemporaneous field-survey accuracy.

Example first comparison: three feature sources, MLP/CNN/RF/compact U-Net,
5/20 original support tiles, one fixed support draw. Record full valid neural
support pixels and the RF sampling cap separately, with each head's actual
training budget. Freeze all heads before querying; compare only matched rows.
