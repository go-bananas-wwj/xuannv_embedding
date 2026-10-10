# Paired uncertainty for the five strong classification heads

`xuannv experiment summarize-strong` reads completed G9 calibration/test artifacts and
produces AP uncertainty for RF, SVM, kNN, MLP and convolution separately. It never chooses
a best head, pools heads into a new selection score, refits a model or certifies the full
paper claim. F1/IoU/balanced accuracy/boundary point estimates remain in the G9 results;
this component computes intervals for the registered classification primary metric AP.

```bash
xuannv experiment summarize-strong --spec /path/strong.json \
  --test-identity-sha256 REGISTERED_TEST_IDENTITY_SHA256 \
  --output /path/new-strong-uncertainty --threads 2
```

The CLI fixes 2000 paired tile draws and seed 20260921, matching the primary C/R/Q schedule.
The Python API permits smaller synthetic checks but records `registered_schedule: false`
when they differ. Every task, head, support seed and model uses the same drawn tile
multiplicities, including registered tiles with no labeled observations. Pixels within
a tile stay together; this is not a pixel-independent bootstrap.

Before computing metrics, the reporter verifies the strong/primary candidate contracts,
complete calibration/test identities, implementation hashes, common-domain files, every
support and retained-position record, saved validation predictions, all model payload
bytes and test predictions. It hashes numeric NPZ and joblib bytes without deserializing
models, invoking an optimizer or initializing the registered inference device. Thus an
NPU-produced prediction archive can be summarized on CPU without loading NPU models.
The producer backend/runtime remains recorded in each identity; reporting does not assume
cross-backend prediction equivalence or demand the producer backend for arithmetic on
already saved predictions.

The saved test common labels and all-model feature-validity mask independently define
the expected truth, canonical tile identity and flattened valid pixel positions for each
task. Every head/model/support/budget prediction must match that domain exactly; matching
prediction hashes alone cannot excuse a wrong pixel order. Original source feature/label
arrays are not reopened or used. Archived AP is independently recomputed from each saved
prediction before any family summary is accepted, within absolute tolerance 1e-12.

For each draw, compute pooled AP separately for each model realization and support seed,
then average these APs, not the predictions. Within each head, average tasks and budgets
within OSM and ESRI and give the two sources equal weight. Keep every ordered method pair,
head, source and task/budget result. Realization names and support seeds accompany the
full individual metric draws; ordinal realization indices used by the numeric routine
must not be mistaken for independently pretrained seed IDs.

Intervals are pointwise 95% percentile intervals, without multiplicity correction. A
condition with no positive truth has undefined AP. Undefined draws propagate through
ordinary means; any undefined draw withholds the corresponding unconditional interval.
Do not use `nanmean`, delete an unfavorable task or substitute zero. Other fully defined
sources/heads retain their own intervals. A positive interval is supplementary evidence,
not proof that primary regression/retrieval, auxiliary tasks, multiple training seeds,
geographic provenance or the complete improvement claim have passed.

Tests cover source balance, all-head completeness, paired cancellation, undefined draws,
invalid matrices, unequal realization counts, exact archived domains, independently
recomputed AP and modified payload rejection. The file-backed synthetic integration runs
G9 calibration/scoring first, then makes all original feature and label arrays unavailable;
summarization succeeds with model loading and source preparation forbidden and leaves the
producer artifacts unchanged. Synthetic shortened resampling is explicitly labeled.
Actual locked-candidate comparisons and their provenance/recipe audits remain required.
