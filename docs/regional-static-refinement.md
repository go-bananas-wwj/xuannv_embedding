# Full-region static-map refinement

`xuannv experiment prepare-static-worldcover --spec FILE` aligns the official
WorldCover reference to **every** record in an existing image cache. It retains
11 valid classes (indices 1–11) plus ignored index zero, and stores a separate
small target manifest without copying any image tensors.

`xuannv experiment refine-region --spec FILE --output DIR --device DEVICE`
loads a registered transformer-fusion parent and adds a training-only static
decoder. The decoder reads the arithmetic temporal mean of the monthly maps.
The objective adds a separately weighted cross-entropy to the unchanged parent
loss. A head-only phase is followed by joint optimization of the public encoder,
fusion branches, and reconstruction heads; the original semantic probe remains
frozen. Reconstruction aliases are migrated explicitly, preserving original
parameter values. In particular, an old target called `worldcover` must not be
mistaken for a genuine ESA target if its data is derived from another map.

The strict recipe binds parent configuration, checkpoint and registration,
image cache, static labels, target aliases, all training indices, loss weights,
parameter-group learning rates, exact update budgets and hardware geometry.
The original downstream partitions remain unchanged in the image cache.
The required disclosure is
`full-region-map-supervision; downstream-query-labels-seen`.

All-region optimization is a distinct protocol from unseen-label spatial
evaluation. Downstream heads may still have separate support and query sets,
but this does not undo label exposure during representation learning. Exports
carry both the representation-training index list and original downstream
partitions. Results on the training reference describe map agreement, not an
independent ground-truth assessment.

The runner shuffles complete cycles, pads to the distributed world size, and
continues accumulation across cycle boundaries to preserve the exact effective
batch. A checkpoint records successful optimizer updates, phase, sampler
cursor, rank-specific RNG/scaler state, and visited indices. Resume rejects a
different recipe, code identity or runtime world size. Stage-boundary resume
recreates the joint optimizer and its deterministic RNG just as an uninterrupted
run does. Invalid gradients do not count as successful updates.

`--stop-after-updates N` provides a bounded preflight/pause. A preflight uses its
own recipe and directory; it must never be silently resumed as the official
experiment. The final checkpoint is selected by the registered update budget.
`xuannv experiment export-refinement --spec FILE --checkpoint FILE --output DIR
--device DEVICE --batch-size N` strictly reloads the recipe and exports all
monthly embeddings. The static map is never an encoder input.

Validation includes CPU gradient, class mapping, complete spatial scope,
sampling, exact resume at and within stages, and checkpoint-to-export tests.
An actual device preflight must cover the joint phase before expensive training;
it checks memory, finite losses, changed public and fusion parameters, unchanged
semantic-probe parameters, distributed registration and unique patch coverage.
