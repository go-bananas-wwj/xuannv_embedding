# Auckland west 2024 execution status

Updated 2026-10-16. Approved protocol: [paired experiment](auckland-2024-experiment.md).

- All 320 tiles and 12 monthly public observations are prepared; static aerial input
  remains a single observation. Both strict initialization checks passed from scratch.
- Five spatial folds are frozen. Each uses 192/64/64 train/validation/test tiles, with
  2x2-tile blocks kept together. All 320 tiles receive exactly one held-out test role.
- Both real-input NPU forward gates passed. Two successful distributed optimizer updates
  passed separately on three NPUs per model, with effective batch 48 and micro batch 4.
- Micro batch 8 passed for both models. The highres two-update check took 55.6 seconds,
  compared with 25.6 seconds at micro batch 4, and approached device memory capacity.
  Formal runs therefore use public micro batch 8 and highres micro batch 4, both at
  effective batch 48. Both three-card training jobs are running. Device 0 is preserved.
- Both final runs are registered for 5,334 optimizer updates, seed 41, learning rate
  1e-4 for all groups, and the same weak supervision. Step 1,334 is a fixed preview.
- Official AEF 2024 discovery selected two intersecting COGs. Regional acquisition is
  complete with 100% valid coverage on the output grid. Signed-square dequantization
  precedes interpolation and normalization; south-up native COGs are supported.
- Eleven LCDB classes satisfy the frozen pixel-count and spatial-block requirements.
  Common classification coverage is 2,618,840 pixels, and LiDAR coverage is 3,611,594
  pixels after evaluation masks and patch borders. Full exclusion reasons are retained.
- The first actual AEF MLP evaluation completed on NPU 7. The persistent followup queue
  is running there with CPU random forests in parallel. It will export each completed
  encoder, run all registered readouts, render fixed reference-selected scenes, and push
  compact stage reports. The 215 readout settings do not represent additional encoders.
- Static-input, common-initialization, spatial-fold, checkpoint-milestone, CLI and reference
  regression checks passed. The full suite passed 818 tests; subsequent empty-tile,
  figure-selection and historical-reference cases passed targeted regression tests.
  Black, Ruff, repository validation and package build passed. Runtime payloads remain
  outside Git. AEF data attribution: The AlphaEarth Foundations Satellite Embedding
  dataset is produced by Google and Google DeepMind (CC-BY 4.0).

No downstream accuracy or superiority is claimed at this stage. No Overleaf changes.
