# Matched visual encoder extraction

The public S2 RGB inputs are ready in all three regions. Local-image 10 m controls
retain the original RGB dates, values convention and exact fine observation mask.
Encode both controls with the same SAT-493M weights, 512-pixel crop, crop batch 8,
0.5 m sampling grid, output width 64, seed 41 and regional PCA fitting rule as the
original local-image features. No control is a primary downstream victory yet.

Extract seven nonoverlapping native-feature shards on free NPUs 1–7. Each shard
uses extraction-only mode; no shard fits PCA. Validate identical encoder parameters,
source references, geographic positions, completeness and native precision. Merge
all 1024D features back into the original full regional order, publish the feature
and mask files atomically, and then fit exactly one unlabelled regional PCA. Unknown
fields, overlapping/missing indices, changed parameters and unowned destinations
are errors. Keep the original images, weights and existing feature bundles intact.

Only after the final 64D output passes its identity/shape checks may the controller
remove its own regeneratable native caches. Record every removed path, checksum and
physical byte count; retain PCA parameters, source manifests and all compact results.
Process one region/control group at a time to bound peak disk use. Preserve the
100 GiB reserve, bounded retries/timeouts and explicit STOP behavior. Existing
unchanged completed outputs are verified and reused rather than scored again.

AEF's original Table S2 and supplement S15.8 list Copernicus GLO-30 elevation and
terrain quantities as training targets. GLO-30 is a surface-height model, including
vegetation and buildings. Do not describe it as a listed raw inference input or
silently claim equal input availability if adding public terrain directly to the
regional representation. The candidate public source tiles were located for all
three regions; no terrain model trial has been started. LINZ LiDAR evaluation
references remain excluded from upstream inputs/training. [AEF original paper](https://arxiv.org/html/2507.22291v1)
