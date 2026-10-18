# Shared wave 1: two regions pass, unified goal remains unmet

All 72 calibration jobs completed (480 task fits, all twenty registered classes).
The per-task receipt verified exactly the same support tiles, sampled pixel and
target hashes, and calibration observation counts against both AEF and public
controls. All eleven candidates and the public-statistics diagnostic are retained.
No high-resolution candidate passed all six gates, so no new full-region query
was selected. These are calibration results, not full-region test improvements.

The best shared recipe by the registered worst-gain rule is separate PCA blocks
with 16 public, 16 frozen high-resolution semantic and 32 spectral coordinates.
It uses the identical rule in all three separately fitted regions.

| Region | Reader | AEF F1 % | Public F1 % | Shared F1 % | Gain vs AEF, pp | Gain vs public, pp |
|---|---|---:|---:|---:|---:|---:|
| haidian | mlp | 75.14 | 66.04 | 69.72 | -5.41 | +3.68 |
| haidian | conv3x3 | 76.89 | 66.05 | 71.38 | -5.51 | +5.33 |
| harbin | mlp | 56.12 | 55.48 | 59.58 | +3.46 | +4.11 |
| harbin | conv3x3 | 58.17 | 55.88 | 61.97 | +3.80 | +6.10 |
| auckland | mlp | 58.04 | 56.18 | 61.16 | +3.11 | +4.98 |
| auckland | conv3x3 | 59.31 | 56.81 | 62.18 | +2.86 | +5.37 |

The same recipe clears AEF/public in Harbin and Auckland. Haidian fails, so the
goal is not complete and the manuscript cannot claim a three-region success.
Haidian's forest F1 explains much of the deficit: public 36.33/32.94%, shared
52.65/59.17%, AEF 66.67/71.40% (MLP/CNN). Cropland also worsens under this recipe;
those classes must not be removed. The spectral-only diagnostic is weak in all
regions and does not explain away the need to measure high-resolution contribution
with matched public-plus-statistics refits.

## Evidence determining the next action

Haidian's public embedding and wave-1 statistics use December 2025 through May
2026; Harbin uses May through October 2025, Auckland all twelve months of 2024.
Locally processed public rasters include Haidian's unused summer/autumn months.
This is a plausible seasonal-information limitation, not a demonstrated cause.
A common twelve-month window ending at each existing registered last month can
be constructed from local data, with genuinely absent months explicitly masked.
Before fitting, map archived raster grids to registered geometry; Harbin's old
processed patch identifiers are not the reindexed retained patch identifiers.
Never join those inputs by patch number alone.

A separate source audit found that all 320 rows of the Haidian manifest used by
both the restored public and protected best-high-resolution configurations map
the legacy target named `worldcover` to `labels/osm_landcover` rasters. That target
is OSM land cover, not evidence of ESA WorldCover supervision. Preserve its
historical name for checkpoint compatibility and its original scores; use an
explicit provenance-verified ESA WorldCover target for subsequent common training.
Do not infer label provenance from a target or directory name.

The next stage should resolve time coverage and target provenance before adding
model capacity: audit observation-only inputs for a common annual window, prepare
true WorldCover weak supervision for each region, and then register one lightweight
learned residual-fusion architecture and matched no-imagery controls. Evaluation
CLCD/Dynamic World/LCDB targets remain excluded from embedding fitting. Any new
recipe must still pass all three regions; these partial wins do not narrow the
objective. No Overleaf body or protected model bundle was changed.
