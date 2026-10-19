# Prithvi study results and the next supervised fusion hypothesis

All seven extraction shards, three regional projections, eighteen fusion exports
and forty-two readers completed. All280 conditions retained the approved balanced
cohort, exact reference labels/masks/geography, supports, budgets and reader code.
The two candidate families passed their shared48-coordinate audit. Neither passes
scientific acceptance; no full-region query was launched. Software verification
and correct inference did not establish useful task improvements.

Calibration macro F1 percentages, MLP/CNN:

| Region | Regional NN + Prithvi + detail | Prithvi + detail | AEF |
| --- | ---: | ---: | ---: |
| Haidian | 81.952 / 84.671 | 80.228 / 82.889 | 83.875 / 84.382 |
| Harbin | 57.838 / 58.236 | 55.521 / 55.332 | 58.396 / 58.796 |
| Auckland | 57.583 / 57.216 | 55.732 / 54.509 | 63.434 / 64.683 |

The standalone Prithvi64 public control is also weaker than AEF in all six slots.
Public reference floors do not decrease. The complete280-row comparison CSV and
all raw records are retained outside Git. The earlier context candidate remains
the best common exploratory recipe by worst-slot gain, but is not accepted.
A label-free40-fitting/20-other-tile diagnosis finds relative-position templates
explain approximately12%of held-tile feature variation. That does not establish
the cause of the task deficit or demonstrate that subtracting a template helps.
No normalization result has been fabricated from that audit.

## Evidence for a different learning objective

The current regional calendar backbone was trained primarily with WorldCover and
observation/visual reconstruction. WorldCover's broad classes alone do not give
separate land-use tags for parks, farmland, forest and wood. The largest remaining
Auckland gaps include Urban Parkland/Open Space and Exotic Forest. This suggests
a finer weak-supervision experiment, not proof that omitted supervision caused
the failures. Original OSM caches already exist in all three regions, and the
full1024D local-context tensors are retained. The next trial can test learned
selection of useful directions without rerunning image encoders or using primary
reference maps as training labels.

A read-only geometric audit clips the existing caches to all320/422/320 image
footprints, never primary reference masks. Mapped category union areas in km2:

| Region | Forest tag | Wood tag | Park | Farmland | Grass | Construction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Haidian | 38.263 | 85.521 | 43.309 | 11.450 | 2.596 | 13.455 |
| Harbin | 11.692 | 8.748 | 22.358 | 1.966 | 16.340 | 9.635 |
| Auckland | 4.881 | 177.895 | 15.906 | 114.912 | 3.474 | 4.565 |

Forest/wood are literal OSM tag classes, not biological plantation/native labels.
Overlaps are not counted as independent land area. Leaf-type annotation is too
sparse/nonuniform for the first common trial, so no leaf-type head is selected.
No explicit LCDB/CLCD/DynamicWorld provenance was found in this audit; most OSM
features omit provenance, so do not claim fully independent human truth. Exclude
features explicitly naming those primary products during label preparation and
record unknown lineage. OSM supervision and cross-product exploratory evaluation
must remain visible in the paper's exposure table.

## One common learned family, independently fitted in each region

Register before targets/weights/scores: use the same genuine-public context
anchor64 and full original physical95 within each region. Supply fineDINO64 and
uncompressedDINO1024 context (with actual coverage) in native, publicS2 and
same-local-image degraded roles. All nine native tensors exist. Do not discard
the1024D context with another fixed PCA cut before the new semantic learner.
Public input arrays, target arrays, class weights and label domains are identical
across each region's three roles. No primary CLCD/DW/LCDB value enters fitting.

Learn zero-initialized public/local corrections in per-coordinate public-standard-
deviation units. The final output is the original public vector plus its standard
deviations times learned corrections. It remains64D with no final unit norm.
A no-op/detail/context gate uses real availability; missing local sources produce
exactly the learned public branch, including biases. Penalize the final combined
correction, not just the public branch before local injection. These hypotheses
address inappropriate magnitude handling and forgotten public information; they
need numerical and task evidence.

Train the same width128 architecture for2,000 updates, seed41, AdamW3e-4 and
weight decay0.01, batch64 after a real memory gate. First200 updates train decoder
heads on the unchanged anchor; then update the fusion. Use WorldCover12, OSM
area-use7 (zero unknown), geometry4 and normalized physical/visual reconstruction
heads. Unknown OSM locations are ignored, not negative background; overlapping
geometry categories are unknown. Area priorities, road corridor width and exact
loss weights are fixed in the outside-Git registration. No artificial image or
feature masking is introduced. Every region uses its own observations and labels,
not a combined regional dataset. Keep terminal weights and bounded recovery.

The nine matched fits produce eighteen fresh primary readers/all120 conditions
under the unchanged v2 protocol and nondecreasing public envelope. Only a common
eligible model receives frozen full queries. Conditional native-only ablations
can isolate OSM supervision and full-context information after primary success.
Label rasterization/model/trainer code, tests and real NPU gates must precede
formal training. This registration is not evidence those models already trained.
Original model bundles, all failures, introduction/related work and the Overleaf
body remain protected. The overall three-region goal is still unachieved.


## Implemented pre-training gates

The generic `osm-weak-targets` and `semantic-refinement` experiment commands now
prepare known-only weak labels, train the residual learner and export a terminal
64D embedding. Configurations reject evaluation fields and unknown or duplicate
keys. Training serializes recovery state atomically, checks output ownership,
cycles through every tile and records weak-label domains and class weights.
Per-source unlabelled statistics reset the same seed independently so a missing
optional source cannot change the public statistics. Native context alignment
matches the frozen exporter, including padding, and remains differentiable without
an export-module dependency in the model.

CPU tests cover nonunit-anchor identity, exact missing-source fallback, final
preservation, frozen-input gradients, matched label exposure, deterministic
interrupted recovery, geographic mismatches and cold export. Real NPU gates use
two actual tiles in each region, four updates with a restart after update two,
and terminal export. Full-fusion BF16 batch64 on actual observations passes at
13.8 GiB peak allocation; a three-update synthetic batch64 also passes. These
checks validate execution, not downstream improvement. All numerical receipts,
source identities, maps and weights remain outside Git.


## Frozen-input residency optimization

The first full streaming attempt exposed substantial host batch overhead: two
actual batch64 preparations took31.4–39.8 seconds under concurrent load. That
attempt ended before any terminal model or primary reader score; its checkpoints,
curves, source and process records were archived. No failed primary score was
removed. The scientific recipe and all nine regional/role specifications stay
unchanged.

Canonical streamed float32 features, Boolean masks and integer targets are now
loaded onto the assigned device once, then sampled by the same deterministic
tile indices. This changes data transport only. CPU four-update model parameters
remain bitwise equal to streaming. Actual NPU batch64 values, dtypes and strides
match exactly; all three real mini-run exported embeddings are byte-identical.
NPU parameter tensors are not all bitwise equal (largest observed absolute
difference4.99e-7), although shared normalization buffers match exactly. Those
measurements are retained rather than described as exact parameter equality.

The largest real region,422 tiles, passes ten full-fusion BF16 updates at batch64
with24.6 GiB peak device allocation. Its11.5 GB canonical input cache loads once;
subsequent batch gathering takes0.060 seconds, and ten complete updates take4.13
seconds in the isolated gate. These timings exclude full queue contention and
reader fitting. Three-region restart/export and recovery gates also pass. This
performance improvement does not establish an accuracy gain or complete the goal.
