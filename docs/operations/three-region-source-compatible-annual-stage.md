# Source-compatible annual inputs and retained observation features

The common learned residual trial finished six independent 1200-update fits and
all eighty paired task-reader conditions. Every registered tile participated;
paired weak-label domains, class weights and public/spectral normalization matched
exactly. It did not clear the shared gate, so no full-region query was selected.
These calibration F1 percentages must not be promoted as a three-region victory:

| Region | Highres MLP/CNN | Matched public MLP/CNN | Fixed AEF MLP/CNN |
|---|---|---|---|
| Haidian | 69.90 / 70.26 | 66.65 / 68.60 | 75.14 / 76.89 |
| Harbin | 56.19 / 57.09 | 55.09 / 58.29 | 56.12 / 58.17 |
| Auckland | 54.50 / 55.66 | 54.21 / 55.69 | 58.04 / 59.31 |

The earlier frozen 16/16/32 fusion remains stronger overall. Retain that observation
information in the next screen; fitting a coarse weak-map objective is not evidence
that the resulting 64 coordinates preserve every finer downstream distinction.
Semantic supplementation, rather than wholesale replacement of the observation
features, is a later hypothesis if corrected inputs alone remain insufficient.

## The original public caches remain available and valid as registered baselines

An all-record source-mask audit found S2 coverage of 96.60%, 99.97% and 92.44% of
the public monthly union in Haidian, Harbin and Auckland respectively. These are
monthly-composite mask counts, not per-scene clear-sky rates. In particular the
original public caches did not lose almost all optical data. Do not invalidate or
silently rerun those baselines based on the new annual-branch failure.

The added annual branch mishandled legacy quality storage: S2's archived SCL field
is nodata at most pixels although its independently stored validity mask is good;
China Landsat's archived QA field was scaled as reflectance. Requiring those broken
quality fields discarded usable observations. Consequently the failed annual screen
confounded temporal coverage and source availability; it cannot show that adding
seasons is inherently harmful.

## Explicit compatibility, with independently checked mask provenance

Default annual processing continues to check named integer SCL/QA flags and the
companion mask. A source may instead use its separate mask only through an explicit
`validated_companions` reference to a completed source-specific proof. A missing
companion or failed proof is an error. The mask must match the image's CRS, bounds
and dimensions; physical-band finite/nodata checks still apply. Never silently
ignore malformed quality flags for unverified data.

For Haidian S2, regeneration from original raw observations matches all 16,384
mask pixels of a complete archived tile, including its 16,360 valid pixels. For
Harbin S2, independent official per-asset quality masks were merged by any-clear
acquisition, matching the valid-scene contributions to the daily reflectance mean.
Eight tiles agree at 131,036 of 131,072 pixels; all 36 differences (0.0275%) lie in
one-pixel categorical boundaries after independent reprojection. The zero-tolerance
mismatch report is retained; the separate accepted proof explicitly records the
one-pixel boundary tolerance and zero interior disagreements. Do not claim exact
Harbin pixel equality or confuse this union with the older conservative all-clear
quality merge used in its public cache.

Landsat uses regenerated integer QA from original NetCDF, with reflectance-only
scaling and nearest-neighbor quality reprojection. All 163 daily China jobs have
completed with per-file checksums and per-valid-pixel quality checks. The corrected
assignment-mask implementation and actual raw-coordinate pilot passed; 926 tests
passed for that fix. The S2 compatibility extension adds two further regressions.

## Ordered shared comparison

Generate new, separately named annual inputs with the validated S2 masks, restored
Landsat and unchanged SAR. Keep all original data and historical results. Assert
that each requested source has observed pixels and report real monthly gaps;
never fill absent months or use evaluation masks in observation preparation.
Auckland's already correct twelve-month inputs remain unchanged.

First test the same previously selected 16 public / 16 local semantic / 32 spectral
rule in every region, replacing only the added annual spectral source by these
corrected inputs. Include a matched no-local-imagery version using an explicitly
zero semantic source generated from public geometry, without reading local imagery.
Use the original AEF/public anchors, all twenty classes, MLP/CNN, twenty support
tiles and one hundred reader updates. Reuse Auckland's unchanged full-fusion results
only after byte and sampling-contract checks. Select no different city recipes.

The shared method must strictly beat AEF, the original public baseline and its
matched public-plus-spectral counterpart in all six slots, with the original
0.5 pp per-region mean gain requirement over AEF. Only that eligible shared method
receives new full-region scoring. All negative class results remain in the record.
Pretrained-encoder/public-RGB and local-image-resolution controls remain required
before attributing a successful result specifically to spatial detail. No paper
body changes or completed-goal claim follow the preparation or software checks.
