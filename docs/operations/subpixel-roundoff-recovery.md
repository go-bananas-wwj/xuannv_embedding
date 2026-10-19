# Recover the subpixel trial from float32 RGB endpoint roundoff

The first subpixel execution failed before any downstream reader or primary score.
Two degraded-cache workers exhausted their bounded retries; the parent completed
its remaining workers, then exited with an actual failure. Seven complete caches,
seven projection bundles and fourteen embeddings had been generated. They are not
a completed scientific comparison and do not enter candidate ranking.

A source diagnostic found six samples at 255.00001525878906 in Haidian tile 2,
view 4, a float32 resampling overshoot of one ulp at the declared 255 display bound.
The initial cache validator rejected every value outside [0,255]. The fix permits
only 8 * float32 epsilon * 255 endpoint tolerance, then clips to the declared
range. Nonfinite and larger radiometric errors still fail. Tests reproduce this
actual boundary condition, retain exact normal-range bits and reject -1, 256 and
1000. The same function applies to all regions and roles.

The recovery is the same registered scientific trial. Raw source and geographic
references, two common strengths (0.1/0.5), public source, cohort, readers, budgets,
thresholds and acceptance rules are unchanged. Only execution output paths and
this numerical guard differ. Every cache and covariance bundle is regenerated
under one corrected pinned source; old and new versions are not combined by
region, reader or task. No primary score was seen from the failed execution.

A role/hash/hardlink/open-file/mapping audit retired only the aborted execution's
reconstructible cache, embedding and deterministic readout arrays: 56 paths,
44 physical files, exactly 29.937595 GiB observed release matching allocation.
All seven parameter bundles, sample positions, source versions, failure logs,
receipts, original specifications and raw/public/reference inputs remain retained.
This does not repeat previous completed cleanups. Retired output completion
receipts are preserved; rebuilding an original output requires staging its old
array-specific completion marker outside the generated directory first, without
removing model fit status or changing its immutable specification.

The corrected code passed 39 focused tests. Nine real region/role gates now read
32 actual source tiles each, including the observed failure position, then verify
partial parameter reload, missing-RGB identity, protected directions and bounds.
These remain partial QA artifacts, with the full original registered index and no
formal model/score claim. All 1,210 full-suite tests, style, repository, source lineage and byte-exact
package checks have now passed. The corrected source must be pinned before
formal processing. The original source and its failure
remain reproducible, and all negative task results from completed earlier trials
remain unchanged. No manuscript body or success claim follows from the repair.
