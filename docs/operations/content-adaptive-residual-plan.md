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

The package implementation is now present with a dedicated strict training and
export command. Nineteen behavioral tests cover clipping, convex bounds, frozen
heads/source gradients, exact CPU recovery, invalid identities, duplicate fields,
CLI routing, terminal-payload integrity and batch/output lifetime. All 1,132 CPU
tests, formatting, static checks, repository policy and package builds passed;
wheel and sdist copies of the new production modules exactly match the source.
All nine real region/role source and weak-domain contracts passed.

Three real NPU checks used the full original regional index with two-update
partial training, two further updates through recovery, and cold inference on
two audit tiles. No four-update checkpoint was labelled a terminal fit. All
heads/statistics stayed unchanged, FP32 alpha/output/logits and componentwise
bounds passed, and output arrays were finite. These checks do not score tasks.

| Candidate batch | Measured samples/s | Peak allocated GiB |
| --- | ---: | ---: |
| 64 | 771.47 | 15.57 |
| 128 | 801.38 | 22.97 |
| 256 | 818.03 | 37.78 |

The predeclared hardware rule selected batch 256. The full 422-tile resident check
then trained 100 updates, recovered to update 110 and retained the same 37.78 GiB
peak with finite source-bound weights. The formal eighteen specifications now
freeze batch 256 for every case, role and region; none uses the preflight weights
as an initialization. These are hardware and integrity checks, not accuracy or
statistical-significance evidence. Formal fitting and the unchanged complete
primary evaluation are the next stage.

A further 29 unique obsolete evaluation grids from two completed negative studies
were retired after enumerating 52 hardlink paths. Actual free-space growth equals
30.4612 GiB, leaving about 149.35 GiB. All 23 protected current-reference/source
arrays, every source annual embedding and weight, original scores, predictions,
labels, masks and figures remain. Per-file identity/deletion records are retained
outside Git; the removed grids can be rebuilt without refitting old readers.

All eighteen formal gate fits have reached update 2,000; remaining exports and
primary readers are still pending. A separate reconstruction-readiness audit
located every declared implementation fingerprint for all eighteen frozen
upstream source checkpoints. The semantic source maps to archived commit
`6096c14`, and the geometry source to `2e3e735`. This proves availability of the
checkpoint-declared code dependencies, not complete raw-data/PCA closure or a
full inference replay. It performs no new model/head fitting or task scoring.
