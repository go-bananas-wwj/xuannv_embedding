# Preserve observation magnitude and check local image resolution

The corrected annual screen completed all eighty paired task-reader conditions.
Local imagery improves its matched public representation in all six slots, but
Haidian still fails against AEF. The common goal remains unachieved; do not select
three different methods or publish a three-region success.

| Region | Highres MLP/CNN F1 % | Matched public MLP/CNN F1 % | AEF MLP/CNN F1 % |
|---|---|---|---|
| Haidian | 67.83 / 72.06 | 65.60 / 71.00 | 75.14 / 76.89 |
| Harbin | 57.97 / 59.59 | 55.55 / 56.86 | 56.12 / 58.17 |
| Auckland | 61.16 / 62.18 | 57.62 / 56.31 | 58.04 / 59.31 |

Haidian's forest F1 remains the largest deficit: 39.65/53.33% versus AEF's
66.67/71.40%. Water is already 81.09/82.22%, slightly above AEF. Do not remove the
forest/cropland tasks or substitute the stronger region-specific setting.

## Two shared normalization variants, registered before their scores

The existing block operation normalizes each point's PCA coordinates and then the
concatenation to unit length. This can remove observation magnitude. In Haidian,
features are subsequently averaged onto the native 30 m CLCD grid; nonlinear
point normalization and spatial averaging also do not commute. These are reasons
to test information preservation, not evidence yet of a performance improvement.

Keep all corrected inputs, PCA fit positions/seed, 16 public / 16 local semantic /
32 spectral allocation, output width 64 and float16 storage unchanged. Compare:

1. Preserve all projected component magnitudes; no branch or output unit norm.
2. Keep public/semantic branch unit norms, preserve spectral magnitude, and omit
   the final output unit norm.

No spatial pooling is added. Reject non-Boolean/ambiguous normalization options and
coordinates outside the finite storage range. Original defaults retain their exact
normalization behavior. Fit and evaluate both roles with the same setting in all
regions. Run all twenty classes with the fixed MLP/CNN, 20 support tiles and 100
updates. Keep all negative task records. Candidate ranking remains worst-slot gain,
then mean. References include AEF, original public, the matched no-imagery variant,
and the stronger previously measured public variants; an easier public baseline
cannot create success. The same candidate must win all six slots and meet the
0.5 pp per-region mean AEF margin before full-region scoring is permitted.

## Matched local-image degradation

Prepare a separate image control from the exact same registered local RGB inputs.
Average only observed pixels into 10 m cells, then interpolate to the original
encoder sampling grid. Keep the original fine observation mask, dates, footprint
and radiometric convention exactly. Store both coarse and encoder-grid rasters
with content hashes and an explicit effective resolution. Upsampling restores no
observed detail. Original inputs must remain byte-identical; unknown configuration
fields, overwrite attempts and nonintegral grids are errors.

Use the identical SAT-493M weights, crop geometry and normalization for extraction.
Native feature extraction and regional PCA projection are separate explicit phases:
shards may extract in parallel, but projection fits one shared basis for the entire
region. Never concatenate independently projected shards. Verified final features
can be reused after deleting regeneratable native caches. Data/weights remain
outside Git. NPU jobs wait for the main screening queue to release its cards.

Also prepare a public S2 RGB encoder control before attributing gains to local
spatial detail. Degraded local imagery isolates detail at the same dates/colors;
public RGB controls added pretrained visual capacity but has sensor/date differences
that must be disclosed. Neither preparation nor software tests prove downstream
superiority. Keep the goal active until the complete common acceptance and
contribution checks pass. The Overleaf body remains unchanged during screening.
