# Fixed fusion check in two existing regions

The 2026-10-18 experiment transfers the selected Auckland feature recipe without
searching either destination's downstream scores. Encoders remain frozen; no
joint city training, new embedding encoder training, or Overleaf edits occur.

## Inputs and processing

- Haidian: all 320 retained tiles, six months 2025-12 through 2026-05. The retained
  best regional high-resolution embedding is the base. This is not a public-only
  base, and already includes high-resolution optical/SAR training history.
- Harbin New Area: all 422 buffer-intersecting tiles, six months 2025-05 through
  2025-10, scored on the existing monitoring-area core mask. The retained public
  embedding is the fusion base; the retained best high-resolution model is also
  compared separately. Neither region is reduced to the Auckland 320-tile count.
- Frozen DINOv3 ViT-L/16 SAT-493M adds global satellite pretraining. Optical
  observations within each existing training window form a masked temporal RGB
  median, not a new monthly observation. Haidian's stored blue/green/red bands
  are reversed; fixed DN 0–3000 maps to 0–255. Harbin uses RGB DN 0–255. No label
  or model score determines radiometry, observation dates or valid pixels.
- Existing companion masks are preserved. A temporal median does not establish
  cloud-free observations where no authoritative cloud QA is available.
- Source images remain approximately 3 m / 2 m. Bilinear sampling to the selected
  0.5 m encoder grid adds no observed detail. Overlapping 512-pixel DINO windows,
  prefix-token removal, satellite normalization and exact 10 m output geometry
  use the same exporter as Auckland.
- PCA is fitted independently on each region's unlabeled features, seed 41,
  at most 65,536 samples. Public/regional 40D plus semantic 24D yields 64D;
  cosine-affinity aggregation uses 13x13 output cells and temperature 1.

## Fixed comparisons

Evaluate the original base, original AEF, semantic-only, fusion without spatial
aggregation, fusion with aggregation, base with the same aggregation, and AEF
with the same aggregation. Harbin additionally includes the original high-res
base. No candidate is selected from the destination full-area scores.

Use the existing binary MLP and shallow CNN implementations, 20 support tiles,
100 optimizer updates, unchanged original support/calibration pools and seeds.
Do not compare these numbers numerically with Auckland's multiclass 200-points
per-class protocol. Preserve per-task F1/AP/IoU, positive and negative outcomes,
and verify support identity and scored observation counts across methods.

Haidian includes the ten original OSM/WorldCover tasks and the four independent
CLCD 2025 native-grid tasks. Harbin includes the fifteen existing OSM/WorldCover/
Dynamic World tasks; DW is not an encoder reconstruction target. The original
OSM/WorldCover exposure is disclosed. Scores cover each full registered region,
including downstream support/calibration pixels; they measure overlapping
regional map agreement, not blind cross-region accuracy.

## Execution and outputs

`xuannv experiment fusion-transfer --spec SPEC --phase prepare|semantic|features|report`
prepares registered RGB inputs, extracts frozen features, constructs fixed
representations and job specifications, or summarizes completed readouts.
Run each generated job using `full-region-readout` calibration then full_region.
Every fitted head receives its own feature representation; no old head is reused
against changed inputs. Runtime orchestration and all rasters/arrays/weights/logs
stay in the data workspace. NPU 0 remains untouched; jobs use free devices 1–7.
Local orchestration waits between meaningful state changes, with bounded retries
and a 100 GiB free-space reserve; no conversational polling loop is required.

Execution registration, source hashes, geographical order, true input resolution,
RGB provenance, projections, supports, scores and failure records remain in the
experiment workspace. New input adapter regression checks cover masks, virtual
raster dependencies, band order, fixed radiometry and geographic/identity errors.

## Execution checkpoint

All 742 temporal RGB inputs are prepared. The implementation passed 886 CPU
tests, Black, Ruff, the repository gate and package build; execution uses an
immutable archive of commit `9d536cb`. Real two-tile Haidian NPU extraction passed
shape, finite-value, satellite-normalization, prefix-token and unit-norm checks.

After measuring throughput, a scheduling amendment split each region into three
contiguous inference shards on devices 1/3/4 and 2/5/6. Thirty flushed Haidian
tiles from serial extraction are reused. Shards are restored to the original
regional order before fitting one regional PCA; per-shard fitted projections
never enter the final representation. The merged native features must match the
two-tile gate before downstream evaluation. NPU 0 is not used. This amendment
changes scheduling only, and was recorded before destination downstream scores.
