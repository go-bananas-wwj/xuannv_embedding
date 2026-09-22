# Identity-bound input grid audit

`xuannv experiment audit-input-grids --spec /path/spec.json` checks physical reference
raster metadata against registered cache layouts and official product reference grids.
It does not read pixel arrays, generate embeddings, prepare labels, score a model or
certify an entire final evaluation from a partial geographic check.

The specification has exactly these fields:

- `protocol`: `input-grid-audit-v1`.
- `caches`: named `{path, sha256}` records for the cache JSON files to compare.
- `official_manifest`: `{path, sha256}` of the product's reference-grid manifest.
- `splits`: explicit canonical train/validation/test/buffer partitions, without aliases or duplicates.
- `references`: exactly the selected patch IDs, each with a current raster `{path, sha256}`.
- `output`: a new output directory.

Every cache must have the same complete ordered grid, patch size and partition. Each
selected physical raster must have the exact byte digest recorded in the official
manifest's `reference_grid.sha256`; selecting a merely similar-looking replacement is
rejected. This permits recovery of a moved reference whose archived staging path no
longer exists, while preserving the original bytes. The raster's actual CRS, dimensions,
affine transform and bounds must match the recorded official grid and the cache bounds.
Coordinates are not used to guess the CRS. Affine coefficients use absolute tolerance
1e-9, bounds 1e-5, and CRS equivalence uses rasterio's CRS equality.

Only the explicitly selected reference files are opened; unrelated test files may be
unavailable. Hashing reads file bytes, but raster pixel arrays are not decoded or analyzed.
The result records checked canonical indices, exact split coverage, whether all cache tiles
were covered, whether any test tile was accessed, reference hashes, CRS/units/resolutions,
missing original-path count, source/implementation identities and elapsed time. Resolution
values use each raster CRS's units; they must not be called metres without checking units.
Failures retain completed records and a failure status; existing outputs are never replaced.

This is upstream evidence for the eventual geographic contract. A train/validation-only
run cannot certify test or buffer tiles. It also does not verify corrected official
embedding values, native high-resolution resampling, raw feature channel definitions,
reference-label dates/classes or final exported model manifests. Those must be bound and
verified separately. Observe the final recipe lock before registered formal test access;
the audit itself does not create or select a recipe.

Synthetic tests use a different projected CRS from the current experiment. They cover
missing old paths with exact matching bytes, selected-only file access, mismatched cache
bounds/size, duplicate patch IDs, reference changes, recorded CRS/shape/affine mismatch,
and a physical raster shift despite internally rehashed metadata. This prevents treating
consistent JSON descriptions alone as evidence of correct physical geography.
