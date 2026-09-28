# Additional RF and U-Net comparisons

`xuannv experiment review-baseline --spec JOB.json --phase calibration|test`
adds isolated readouts to a previously registered review cohort. It does not
change the frozen encoder, the ongoing MLP/CNN implementation, or old predictions.

Each strict job specification contains `protocol: review-additional-baseline-v1`,
the registered `cohort` path/hash, `shared` array root, `model`, `head` (`rf` or
`unet`), explicit `device`, `budgets`, `support_seeds`, and separate `output`.
Calibration must finish before query scoring. All arrays, saved heads, records,
and predictions have verified identities; completed conditions can be resumed.

RF reuses the existing registered implementation: 200 trees, square-root feature
sampling, minimum leaf size 2, balanced class weights, up to 4096 pixels per
class sampled from the same support tiles. These pixel counts are reported;
they are not equated with the full-tile label consumption of neural models.

The U-Net is a compact adaptation, not an exact reproduction of the original
biomedical network. Three pooling levels use widths 32/64/128/256, two 3x3
convolutions per block, GroupNorm, bilinear upsampling and skip connections.
It reads the same 10-metre aligned feature maps as the CNN, and predicts one
binary task per fit. Training uses all valid labeled pixels in the support
tiles, training-only standardization, zero invalid context, AdamW and weighted
binary cross entropy. A single trajectory saves the best validation AP among
100/300/1000 updates (earliest exact tie), then freezes weights and threshold.

For a bounded first comparison, register 5/20 support tiles and the first
existing support seed *before* scoring these new models. Compare its scores
with the matching seed/budget rows of the existing MLP/CNN matrix, rather than
their five-seed average. This is a single-support-draw comparison, not evidence
of variance over support draws. Three representations, two heads, two budgets,
one draw and ten tasks produce 120 conditions. Model-training seeds are not
repeated. An NPU save/reload gate must pass before scheduling U-Net jobs.

Report AP, IoU, F1, boundary F1, training label counts and actual timing. The
100-update CNN is a fixed-compute control; the separately selected longer CNN
training and U-Net are labeled as different training budgets. Runtime from
concurrent jobs must not be used as a same-hardware speedup benchmark.

The full-region encoder has seen the OSM/WorldCover query labels. This comparison
measures regional map agreement and readout utility, not unseen-label spatial
generalization. Splitting the downstream head alone does not remove exposure.

References: [U-Net](https://arxiv.org/abs/1505.04597),
[AlphaEarth evaluation](https://arxiv.org/html/2507.22291v1#S18).
