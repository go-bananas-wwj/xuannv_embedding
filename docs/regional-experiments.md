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

`xuannv experiment follow --root /data/experiment` advances a registered six-run calibration to
three independent 800-epoch public baselines. It requires every pilot to complete, selects the
learning rate by the predeclared paired-seed mean, waits for devices 1–3 to have no processes, then
starts each full baseline from scratch using the complete development training split. A file lock
prevents concurrent controllers; existing formal registrations are never silently relaunched.
Training uses the immutable code snapshot and cache checksums in the registry. Any failed pilot
stops advancement. `controller_status.json` records waiting, running, completion or failure.
Each full baseline runs in its own `runtime/<run-name>` directory beneath the experiment root.
Device compiler reports therefore stay outside the immutable source tree and do not overwrite
reports from other runs. The registry records this working directory alongside each run.
The controller does not implement or launch high-resolution adapters after the baselines finish.

`xuannv experiment fold-cache --cache /data/experiment/cache --output /data/experiment/fold1
--test-group 1` creates an independent cache manifest for a further spatial fold. Existing sample
paths and checksums are reused without modifying the parent. Group k is held out for testing,
group (k - 1) modulo 5 validates, and adjacent training tiles are removed using the original
one-tile buffer. Group 0 reproduces the original full development split. Fold manifests omit
pilot subsets; each resulting baseline is trained from scratch with the frozen hyperparameters.
Results from these folds must not feed back into development model selection. They provide
additional baseline checkpoints, not evidence that upgraded methods have been evaluated.
