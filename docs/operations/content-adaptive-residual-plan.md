# Learn a bounded correction gate from the existing weak labels

The fixed mixtures improved the closest common result but did not meet acceptance.
All forty task-reader cells of the 15% case were retained and examined. Major
AEF deficits persist for Harbin grass MLP and Auckland classes 2 and 52. Changing
a single global coefficient trades MLP and CNN outcomes, motivating a content-
dependent test. This is a hypothesis; the current observations do not prove a
particular failure cause or that a learned gate will repair it.

The design adapts the multiplicative gated-fusion idea in
[Gated Multimodal Units for Information Fusion](https://arxiv.org/abs/1702.01992).
Its usefulness for these regional representations remains to be measured.

Keep the same two frozen regional source methods and verified shared public
anchor. Train a small region-independent network to predict one mixture coefficient
per output channel and pixel. Its inputs are the standardized public anchor,
each source's correction, their absolute difference, and the existing observed
95-channel public physical statistics with their observation mask: 446 channels.
Two 3x3 convolution, GroupNorm and GELU blocks use width 32, followed by a zero-
initialized 1x1 projection to 64 coefficients. A sigmoid starts every coefficient
at 0.15. Each output coordinate is a convex mixture of the two source coordinates;
there is no direct addition of unbounded hidden CNN features.

Use the earlier semantic model's frozen WorldCover and OSM heads to supervise
only this new gate. All nine existing head capsules passed shape, finiteness and
original checkpoint/export/spec binding checks. Load public and physical
normalization buffers from that same retained checkpoint. The head input must be
the clipped standardized public anchor plus the scaled output correction, matching
the old model; simply standardizing the exported output would lose that clipping
contract. Unknown labels remain ignored in the unchanged physical domain.
No CLCD, Dynamic World, LCDB or AEF values enter gate fitting.

Two cases share the entire architecture and schedule. Their only difference is
a penalty coefficient of 0.05 or 0.20 for the standardized displacement from the
semantic source. WorldCover, OSM area and OSM geometry loss weights remain 1, 1
and 0.5. Source encoders, source embeddings and auxiliary heads are frozen. Both
cases use 2,000 updates, AdamW at 0.0003 with weight decay 0.01, gradient clipping
1, seed 41, 100 warm-up updates and the same cosine floor. There is no burn-in
because the initialized gate reproduces the useful fixed mixture. Gate-body AMP
is BF16; alpha calculation, mixing, frozen logits and losses explicitly use FP32.

Before formal fits, choose one batch size from 64, 128 and 256 using only actual
Harbin maximum-residency throughput and a 54 GiB peak-allocation limit. Discard ten
warm-up updates and time ten further updates. Use the fastest admissible size in
every region, role and case, then freeze its numerical value before formal
training. No primary scores influence this hardware choice.

Registration precedes new gate weights and scores. Implement the model and
trainer in the package with failing-first tests, then full CPU/style/lint/package
checks, three real regional mini train/recovery/export gates and actual selected-
batch resident memory/recovery checks. The implementation is not yet present and
no new gate fitting has started. Extra storage must be reclaimed from identified
obsolete evaluation grids before launch; all source embeddings, weights,
predictions, results and current public-reference dependencies remain protected.

The intended study has eighteen independent regional/role fits, thirty-six fresh
readers and all 240 fixed balanced-v2 conditions. Select one common penalty by
the unchanged worst-six-slot rule, keep the nondecreasing public envelope and
require every region's mean AEF gain of at least 0.5 points. Only an eligible
case gets frozen full-region queries. Complete source/rebuild, contribution,
compute-cost and remaining paper-task evidence is still required after any
primary success. All negatives remain; no Overleaf success claim is supported yet.
