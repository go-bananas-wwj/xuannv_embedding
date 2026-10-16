# Auckland west 2024 execution status

Updated 2026-10-16. Approved protocol: [paired experiment](auckland-2024-experiment.md).

- All 320 tiles and 12 monthly public observations are prepared; static aerial input
  remains a single observation. Both strict initialization checks passed from scratch.
- Five spatial folds are frozen. Each uses 192/64/64 train/validation/test tiles, with
  2x2-tile blocks kept together. All 320 tiles receive exactly one held-out test role.
- Both real-input NPU forward gates passed. Two successful distributed optimizer updates
  passed separately on three NPUs per model, with effective batch 48 and micro batch 4.
- Public micro batch 8 also passed. Full-model micro batch 8 calibration is in progress;
  micro batch 4 is the verified fallback. The existing device-0 service is preserved.
- Both final runs are registered for 5,334 optimizer updates, seed 41, learning rate
  1e-4 for all groups, and the same weak supervision. Step 1,334 is a fixed preview.
- Official AEF 2024 discovery selected two intersecting COGs. Regional window acquisition
  is running. Signed-square dequantization precedes interpolation and normalization.
- Static-input, common-initialization, spatial-fold, checkpoint-milestone, CLI and reference
  regression checks passed. Raw data, models, embeddings and logs remain outside Git.

No downstream accuracy or superiority is claimed at this stage. No Overleaf changes.
