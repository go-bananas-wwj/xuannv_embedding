# Shared-domain review readouts

The review workflow evaluates fixed features with independently scheduled neural
jobs. It preserves the original paired evaluator and frozen neural implementation,
so compatible completed experiments can be reused without re-fitting.

`xuannv experiment review-prepare --spec COHORT --output SHARED --phase calibration`
materializes the all-model intersection once for training and calibration. The
same command with `--phase test` prepares query features after prediction-head
selection has been frozen. Array digests and spatial indices are recorded.

`review-calibrate --spec JOB` and `review-score --spec JOB` handle one feature,
one head, and a registered subset of budgets/seeds. A task record is saved after
each condition; completed phases retain their identities on resume. Model,
array, support and prediction digests are verified before reuse. Archive-only
jobs can run on CPU without opening an accelerator context. A failed archive
match in a `reuse_only` job raises an error rather than silently starting a fit.

The neural implementations are the existing pixel MLP and two-layer 3×3 CNN.
The latter has batch normalization and dropout. Separate NPU training runs are
not assumed bitwise reproducible, even with matching initialization and sampled
batches. A convergence check therefore captures 100, 300 and 1000 updates from
**one continuous trajectory**. The fixed-budget result uses that trajectory's
100-update snapshot. A snapshot's saved predictions must replay exactly.

Jobs optionally register `trajectory_checkpoints: [100, 300, 1000]` and a subset
of `support_seeds`. `review-convergence --spec CHECK --phase calibration` consumes
the saved trajectory and freezes its best calibration AP checkpoint, breaking
an exact tie toward fewer updates. Its test phase scores only that checkpoint;
it does not perform another fit. This permits fine-grained independent device
jobs without splitting a small head across devices.

`review-retrieval --spec JOB` adds registered positive-component budgets while
reusing compatible completed predictions. Source labels, model contracts,
canonical positions and frozen prototypes remain traceable.

`review-cluster --spec JOB` fits a fixed K-means configuration on unlabeled
positions from the support region. Query map labels are used only for ARI/NMI
after the cluster parameters are saved. The feature encoder may itself have
used map supervision; label-free clustering does not imply a label-free encoder.
The same samples provide covariance spectra, effective rank, a 95% variance
dimension and a train-fitted three-channel PCA display transform.

The scheduler is external experiment orchestration: one process per device job,
shared read-only feature matrices, bounded CPU-only archive imports and an
explicit free-device check. Frozen source snapshots are used for long runs.
Result producer code remains region independent; data, credentials, weights,
predictions, logs and large reports remain outside Git.
