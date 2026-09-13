# Regional upgrade experiments

Use `xuannv experiment prepare` to materialize a CPU sample cache from an explicit public-source
configuration, and `xuannv experiment run` for an independent CPU/NPU training process. All sample
caches, configurations, checkpoints and runtime logs belong outside the source repository.

The baseline excludes high-resolution inputs and reconstruction targets. Its regional map prior is
retained explicitly; this is not an image-only self-supervised model. The target-only map source must
be included in the source mapping. Preparation checks that every reconstruction target has valid
pixels and hashes each sample. Cache data/source contracts are validated before training.

Five geographic groups are built from projected patch centers using KMeans (seed 42, 20 initializations),
ordered west to east. The western group is the initial test group, the eastern group is validation,
and remaining patches adjacent to either held-out group form a one-patch Chebyshev buffer. The first
calibration phase uses 64 training and 16 validation patches, sampled with seed 20260913; no test
predictions are used for learning-rate selection. Full spatial cross-validation remains a subsequent
stage, not a property claimed by the calibration run.

Calibration compares learning rates 0.0001 and 0.0003 with seeds 41, 42 and 43 for 20 epochs. Models
start from scratch. AdamW, the 800-epoch schedule with 30-epoch warmup, and the original masking recipe
are shared. Gradient norms are clipped at 5; AMP-overflow updates are skipped and eight consecutive
overflows stop the process. The incomplete accumulation window uses its actual number of batches.

Validation uses unmasked inputs and deterministic inference, with the objective weights fixed at
the final scheduled values. Selection averages each run's best validation total loss across the
three seeds at each learning rate; it does not compare a changing warmup-weighted training objective.
This calibration is not downstream validation and does not establish mapping accuracy.

Every completed epoch appends component metrics and durations, and atomically writes best/latest
checkpoints. Resume verifies the configuration, source schema, cache, split and Git provenance;
optimizer, scheduler, AMP scaler and CPU/NPU random states are restored. Run paths are never silently
overwritten. `--epochs` is the total epoch endpoint for this experiment command.

Example (paths refer to externally prepared inputs):

```bash
xuannv experiment prepare --config /data/experiment/base.yaml --output /data/experiment/cache
xuannv experiment run --config /data/experiment/base.yaml --cache /data/experiment/cache \
  --output /data/experiment/run --device npu:1 --epochs 20 --pilot
```

High-resolution adapters, alignment-tolerant losses and their formal evaluations are separate
experiments. No baseline checkpoint may be labeled as having validated those upgrades.
