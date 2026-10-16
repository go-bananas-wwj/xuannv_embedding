# Artifact retention and regional data preparation

Keep data, credentials, model bundles, manifests and execution journals outside Git.
A checkpoint filename is not evidence that a file contains a training checkpoint:
exported tensors and tensor caches may also use `.pt`.

## Verified retention

`export.standalone.load_bundle()` verifies a complete final model and configuration
against bundle hashes, constructs the system from scratch and loads both model and
criterion states strictly. It never follows an initialization checkpoint. Validate
representative real inputs before removing an ancestor dependency. Keep the
selected mean embedding, geographical feature ordering, configuration, source
revision and audit inputs with each primary bundle.

`data_process.checkpoint_retention` reads recorded update indices before considering
filename indices. `compact_checkpoint()` writes a separate inference-only copy,
removes optimizer/recovery state without quantizing tensors, reloads the copy and
compares all model/criterion tensor digests. It does not delete the source. A raw
state dictionary with no update metadata must not be assigned an invented step.

The destructive command accepts an explicit file-role manifest:

```bash
xuannv data storage prune --manifest /data/task/delete.json \
  --journal /data/task/deletions.jsonl --validate-only
xuannv data storage prune --manifest /data/task/delete.json \
  --journal /data/task/deletions.jsonl
```

The `artifact-retention-v1` document contains `allowed_roots`, `protected`,
`requirements`, and `files`. Each requirement/file records `file_identity()`;
files also require an approved `role`. Symlinks, escaped paths, protected files,
missing dependencies and changed identities fail before deletion. Each unlink has
fsynced intent/completion records. Reclaimed space counts allocated blocks only
when the last hardlink disappears. Archive compact results, including negative
results, sampling definitions, thresholds, configs and provenance before pruning.

## Bounded acquisition

```bash
xuannv data acquire --spec /data/task/acquisition.json \
  --jobs /data/task/jobs.json --workers 4
xuannv data acquire --spec /data/task/acquisition.json --materialize
```

Raster jobs use isolated spawned processes so GDAL state is never shared between
concurrent jobs. Preparation reads windowed COGs, keeps unsigned STAC metadata, signs Planetary
Computer assets only while reading and retains validity masks. S1 RTC is available
through anonymous signing; no subscription key is needed. Radar values remain
linear gamma-naught. S2 uses product BOA quantification/offset metadata; Landsat
uses the published reflectance scale and offset. SCL and QA_PIXEL/QA_RADSAT reject
invalid observations before per-pixel monthly medians. No other month fills a hole.

The spec declares a `/data` root, projected bounds/CRS, months, patch size, WGS84
search bounds and budgets. Defaults allow 150 GiB of new artifacts and require
100 GiB of unreserved available filesystem space. Jobs have unique IDs, confined
output/catalog paths and explicit kinds: `linz`, `raster`, `monthly_optical`,
`monthly_radar`, `vector`, or `osm`. Unknown job options are rejected. Receipts
allow completed jobs to be reused; a partial TIFF alone is not completion evidence.
Do not run the same job concurrently in separate processes.

Acquisition and finalization never call training. Finalization requires every
public monthly composite and the static inputs. It writes exact-boundary patches,
normalization statistics, explicit companion masks, an ordered manifest, reference
labels and audits for public-only/static-high-resolution data configurations.
The static aerial file is stored once per patch. A nominal 3 m input uses 427
pixels over a 1280 m patch, retaining the exact geographic bounds. Reconstruction
quality carriers are zero slots, with validity in companion masks. WorldCover
class zero is reserved for invalid/ignored pixels; its eleven real classes use
indices 1–11 and require twelve decoder slots, including the ignored slot.

OSM supervision has separate `masks/` and `valid_masks/` rasters. Known positives
and declared weak negatives from incompatible WorldCover classes are supervised;
other pixels stay unknown. A missing/invalid validity raster fails explicitly.
Datasets without `valid_masks/` retain their previous scalar mask behavior.

Independent references must stay outside training input/source maps. LCDB product
period, source scale and any geometry simplification must be reported. DSM minus
DEM measures surface height, including vegetation; it is not automatically building
height. A building outline's capture year is not its last-modified year. Older
outlines remain historical diagnostics and cannot substantiate current-year
building accuracy. Optional tasks require a verified source, license, coverage and
adequate samples before inclusion.
