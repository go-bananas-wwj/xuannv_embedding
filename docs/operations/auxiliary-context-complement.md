# Preserve observation branches and supplement predicted semantics

The prior shared learned adapter failed the common-method criterion. Replacing
the whole representation with its learned vector substantially weakened the
Auckland calibration tasks. This motivates a separate hypothesis: retain frozen
observation branches and supplement them with predicted auxiliary semantics.
It is not a demonstrated improvement.

Recover the archived model's twelve-class WorldCover auxiliary head and apply it
to that model's verified, archived 64-dimensional embedding. Export probabilities
conditional on the eleven foreground classes; ignore index zero. A missing zero
embedding remains zero. This operation needs neither primary evaluation maps nor
the original WorldCover targets. It reads only observations, auxiliary head
weights, geographic index and provenance. Identity checks bind the training spec,
checkpoint, completed embedding and input role, preventing a high-resolution
checkpoint from being presented as a public-only context producer.

Use the public-only regional models trained by the same earlier rule on genuine
WorldCover targets, and give their identical probabilities to both fusion roles.
Do not reintroduce local-image information through the public context branch.
That rule used the original short-window and earlier annual
statistics, including the annual quality limitation documented subsequently.
Reusing its predicted probabilities does not repair that historical training or
create evidence about a retrained model. No new encoder training is required for
this screen, and its possible benefit must be judged empirically.

Register one shared candidate, `context_complement`, before new scores:

| Source | Dimensions | Normalization |
| --- | ---: | --- |
| Public regional embedding | 12 | Unit projected branch |
| Local visual embedding | 16 | Unit projected branch |
| Corrected annual public observations | 25 | Standardized PCA, unit projected branch |
| Predicted auxiliary probabilities | 11 | PCA, retain projected magnitude |

The output remains 64-dimensional, without final unit normalization. This retains
the observation **branches**, not all coordinates of an earlier 64-dimensional
embedding: their allocation changes and may discard useful information. Keep
that limitation explicit. The public counterpart has no local visual inputs;
its context is identical to the high-resolution role's public-only context. Both use the exact
same fusion rule and geographic/task contracts. No primary task labels enter the
projection. This initial candidate does not add terrain or another recipe sweep.

The three public-only probability arrays were prepared and checked against the
archived head's convolution on nine actual tiles, with maximum absolute error
below 2e-7. However, the upstream public baselines have different historical loss
weights and map targets. This inherited difference prevents this candidate from
proving one uniform end-to-end training rule. Its reader queue is **disabled**;
retain the arrays and tested producer as diagnostics and prioritize the uniform
observation-only backbone experiment instead.

If this context is later regenerated from uniformly trained public models, wait
for previous NPU queues to release devices rather than competing with them.
Audit all eighty task-reader sampling contracts; retain
all twenty tasks and both heads. Select by the same six-slot worst-gain rule and
regional mean improvement gate, including the strongest existing public control.
Only an eligible shared candidate can proceed to new full-region queries. A
complete high-resolution contribution claim additionally needs the matched
pretraining-capacity and degraded-local-resolution controls for the final recipe.
