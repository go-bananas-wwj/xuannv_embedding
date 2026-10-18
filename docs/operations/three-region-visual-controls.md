# Matched visual controls for the shared regional method

The shared method has not met the three-region AEF acceptance criterion. Its local
visual branch also introduces a globally pretrained image encoder. A positive
comparison against a public model without that encoder does not, by itself,
establish a benefit from local spatial detail.

Two controls are registered before their downstream scores:

| Control | Changed input | Fixed components | Interpretation |
| --- | --- | --- | --- |
| Public RGB | Replace local RGB with a public Sentinel-2 RGB composite | Image encoder weights, crops, encoder grid, regional PCA rule, fusion, reader and sampling | Tests access to the same pretrained visual capacity; sensor, dates and compositing still differ |
| Degraded local RGB | Average the original local image to 10 metres and interpolate to the original encoder grid | Original images, dates, radiometry, fine validity mask and all model components | Tests spatial detail at fixed sensor and observations |

Use all three registered regions, all twenty primary binary tasks, both MLP and
shallow CNN readers, twenty support tiles per task and one hundred reader updates.
Each representation has 64 channels. Apply the existing common block fusion with
16 public, 16 visual and 32 standardized annual observation channels. Fit one
regional PCA after merging native image-encoder shards; never concatenate
independent shard projections.

Compare against the already evaluated native-local representation using exactly
the same corrected annual observation inputs. This is a diagnostic of that fixed
fusion recipe. If a later recipe changes its inputs or dimensions, the pertinent
controls must use that later recipe before claiming a high-resolution benefit.
Do not reuse these comparisons as evidence about a different fusion method.

Audit output identities, geographic order, support positions, reference values
and calibration counts against the fixed AEF and public contracts. Retain every
task score and both heads, including decreases. These are exploratory calibration
comparisons, not blind evaluation. They do not establish the main goal alone and
do not trigger additional full-region queries.

The finite local controller waits for validated feature exports, prepares each
control on CPU, and waits for the existing visual extraction and terrain reader
queues to terminate successfully before scheduling reader fits on devices 1–7.
It leaves device 0 alone, reserves 100 GiB, observes STOP files, bounds waits and
retries, and records child identities and terminal failures. Background waiting
uses local timers rather than repeated conversational polling.
