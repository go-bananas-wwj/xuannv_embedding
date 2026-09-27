# External WorldCover evaluation and period-mean embeddings

`training.semantic_probe_month_index` selects the output read by the fine semantic
training head. It defaults to `-1` for compatibility. An integer in
`[-model.num_months, model.num_months)` is required. For December 2025–May 2026,
index `4` selects April. Changing this loss requires a new training run; choosing
a different exported feature month does not change the learned supervision.
Loading a parent checkpoint preserves the new configuration's month selector.

`xuannv experiment prepare-worldcover --spec <json>` prepares nearest-neighbor
categorical references on a registered cache grid. The source raster is bound to
its ESA product provenance and digest. The producer preserves existing OSM labels,
freezes eligible WorldCover tasks using training/validation availability before
query-label access, records unsupported classes, and writes binary label bundles.
Product year, grid CRS, sample budget, source paths and hashes are explicit in the
specification. Renaming an ESRI raster does not create WorldCover evidence.

An optional `task_schema` in a primary evaluation specification declares each
family's source/task groups. For example:

```json
{
  "C": {"osm": ["osm_building"], "worldcover": ["worldcover_tree"]},
  "R": {"worldcover": ["worldcover_tree"]},
  "Q": {"osm": ["osm_building"]}
}
```

Custom bundles contain `indices`, `cache_sha256`, and one `{-1,0,1}` map per
registered task. Regression/retrieval tasks must refer to registered classification
maps. Classification source groups receive equal weight in aggregate comparisons.
Omitting `task_schema` preserves the historical OSM/ESRI schema for old artifacts.

`xuannv experiment export-temporal-mean --spec <json>` reads every month of a
registered monthly export, accumulates the arithmetic mean in float64, and writes
one float32 vector per position. Its manifest has `kind: temporal_mean`, an explicit
`observation_months` list, and a `mean_START_END` stored period. Feature selection
uses `period: START/END`; it never fabricates monthly timestamps. Validity is the
intersection of feature validity over the entire window. No extra normalization
is applied here; existing downstream head and cosine preprocessing remain intact.

Primary readouts accept one to eight CPU workers. Report resampling accepts up to
32 available Numba threads. These settings change execution concurrency, not
support selection, task metrics, or the paired bootstrap schedule. Choose a
representation on validation, lock it before query scoring, and retain all
registered comparison results.
