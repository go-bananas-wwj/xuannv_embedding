# Regional training Watchdog

The Watchdog monitors the two registered regional training processes locally. It does
not call a model API, retrain healthy runs, change input data or alter the training budget.

## Recovery policy

- Inspect progress every 30 seconds. Treat 15 minutes without an optimizer-step advance
  as a stall, with a 10-minute grace period after process launch.
- On an unexpected exit or stall, validate available checkpoints against the registered
  configuration, source schema, region, finite tensors and optimizer state. Select by
  saved update count, not modification time. Fall back if `latest.pt` is corrupt.
- Stop only process groups whose exact training spec matches this run. Recheck ownership
  before escalation, and do not start a replacement while old workers remain.
- Restore model, auxiliary heads, AdamW moments and global update count. The learning-rate
  schedule continues at that update. Keep device groups and effective batch unchanged.
- New checkpoints also store each rank's sampler, CPU/NPU RNG and AMP scaler state.
  Historical checkpoints lack these random states: their first recovery explicitly records
  `legacy_rng_reinitialized`. Subsequent checkpoints support full state restoration.
- Permit five automatic restarts per model, with increasing delays after repeated failure.
  Do not start over from random weights when no valid checkpoint exists. Wait if the disk
  reserve is breached or runtime source changes have not been committed.
- Preserve healthy/completed training. If the downstream scheduler exits during a training
  recovery, wait for its existing readouts to finish, then resume its pending job queue.

The original active jobs save every 100 updates, so recovery can roll back the unsaved
tail since their last checkpoint. Installing the Watchdog does not interrupt these jobs.

## Run and inspect

```bash
xuannv experiment watchdog --trial TRIAL --repository REPOSITORY
xuannv experiment watchdog --trial TRIAL --repository REPOSITORY --once
```

`TRIAL/watchdog/state.json` contains the heartbeat, per-model state and restart counts.
`events.jsonl` records recoveries and checkpoint identities. Restart output goes to
`MODEL-restart-N.log`. All files remain outside Git, under the data volume.

To disable automatic recovery without terminating training:

```bash
xuannv experiment watchdog --trial TRIAL --repository REPOSITORY --disable
```

The daemon notices the `TRIAL/watchdog/STOP` marker and exits. Remove that marker before
explicitly starting the Watchdog again. Completed training is never restarted.
