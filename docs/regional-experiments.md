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

When a completed stage must remain independently reproducible, resume into a **new output
directory**. Copy its `run.json`, `config.yaml`, `metrics.jsonl`, and `best.pt`; change only the
new registration's `epochs` to the next endpoint. Pass the previous stage's immutable numbered
checkpoint to `--resume`, with the original configuration, code, initialization/adaptation flags,
and distributed world size. The runner requires the copied registration to validate provenance.
Record the source registration, terminal status, and resume-checkpoint hashes before launch and
verify that the source directory stays unchanged. Do not hard-link mutable checkpoint destinations.
Use a separate follow-up plan/output for each stage so later training cannot invalidate an earlier
report's terminal-epoch check. Keep the original scheduler horizon in the configuration;
`--epochs` only sets the stopping point. Elapsed time resets for each resumed segment, so report
segment time and cumulative training time separately rather than interpreting the last segment as
the cost of the entire run. The CPU resume regression checks identical model, criterion, optimizer,
and scheduler states against uninterrupted training while preserving the previous directory.

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

## Continuous device queue

`xuannv experiment queue --plan /data/experiment/continuation/plan.json` watches registered
existing runs and dispatches additional independent baseline jobs as individual devices become
available. The plan records devices, explicitly allowed resident process IDs, existing jobs,
and an ordered list of new jobs with code/config/cache hashes and unique output/log/runtime paths.
Optional `depends_on` names gate a job on successful completion of its dependencies.

`queue_state.json` persists each launch and completion; `queue_status.json` records counts and
failures or ten-minute progress stalls. A lock prevents duplicate controllers. Completed jobs
are not relaunched on restart, and uncertain interrupted launches remain held for inspection.
Existing source snapshots, configurations and caches are checked before launch. Each child
uses spawn workers and an isolated runtime directory. Unknown device processes block dispatch;
the controller does not terminate any training or resident process. Failed or stalled runs are
recorded for inspection while independent jobs on other devices continue. Restart uses the same
immutable plan; editing a live plan is rejected rather than silently altering an experiment.


## 共同解码器的外部目标缓存

`xuannv experiment reconstruct-ridge --target-cache /path/to/target-cache` 允许冻结编码器
读取自身训练时登记的观测缓存，同时从独立缓存取得重建目标。例如，没有高分重建头的基座
也可用同容量线性读出器预测高分来源的模型网格目标。其余参数（`--cache`、`--config`、
`--checkpoint`、`--target`、`--months`、`--context`、`--alpha` 和 `--sample-stride`）沿用
共同重建命令；不传 `--target-cache` 时仍使用模型自身缓存中的目标。

两份缓存必须有相同的登记 manifest 身份、月份顺序、图块尺寸、记录顺序、图块 ID、边界及
完整训练／验证／测试／缓冲划分。目标必须声明连续类型和通道数。评价前重新校验两份缓存的
全部训练／验证样本哈希，逐样本复核地区、图块、时间戳及目标尺寸，不读取测试样本文件。
外部缓存的观测、标签和输入掩码不会合并进编码器输入；只有其重建目标与有效目标掩码用于
训练支持拟合、朴素基线及验证评分。

目标源在模型输入中存在时先遮住目标月；本来不存在时不注入该源，身份记录明确列出缺失源。
前缀模式仍删除模型所有未来输入及无日期静态输入。额外别名必须是模型已登记的实际输入源，
不能用缺失源选项忽略拼写错误。结果记录两份缓存身份、目标通道数和缺失遮挡源。
跨模型比较仍须统一目标缓存、采样位置、读出容量和超参数；本功能不提供原始分辨率恢复、
真实时间外推或独立变化检测的证据。正在执行的实验继续使用各自固定的代码快照。
