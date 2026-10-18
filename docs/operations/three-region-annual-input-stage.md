# Common annual observation and real weak-target stage

Wave 1 completed all 480 calibration task fits and failed the Haidian gate. Do
not query its full-region scores or select different city recipes. Its shared
16/16/32 public/semantic/spectral recipe improves Harbin and Auckland, but those
partial wins do not complete the registered three-region objective.

## Input hypothesis registered before new scores

Haidian's largest gap is forest classification, while its current observations
cover winter/spring. Existing local rasters contain unused summer/autumn data.
This motivates testing temporal information before adding network capacity. It
is a hypothesis, not a causal explanation already proved by wave 1.

Use the same rule everywhere: twelve consecutive months ending at the last month
of the existing regional input registration. No later observation is introduced:

| Region | Annual window | Registered geometry |
|---|---|---:|
| Haidian | June 2025–May 2026 | 320 tiles |
| Harbin New Area | November 2024–October 2025 | 422 tiles, original core mask for scores |
| West Auckland | January–December 2024 | original 320 tiles |

Haidian/Harbin native rasters match every registered tile by CRS and bounds.
Their archived numeric patch IDs are not generally the retained IDs: join by
geometry, never by number. Register raster and companion-mask content hashes
before compositing. Calendar months with no observations stay explicitly missing;
in particular the Harbin archive does not invent November/December 2024 inputs.
Auckland's original twelve-month public cache is reused unchanged.

The generic exporter checks named bands, exact grid footprints and quality masks.
For each public source, form per-pixel monthly means from finite valid observations
intersected with companion masks. S2 accepts the pre-existing SCL set 4/5/6/7/11;
Landsat excludes QA bits 0–4. Reject nonintegral quality values rather than truncate
interpolated flags. Quality channels are excluded from physical outputs. Empty
months have zero storage values and false masks; no acquisition is copied into a
missing month. Save physical archive-unit arrays so the existing named-band v2
summary applies the same archived normalization/clipping and explicit radiometry.
The new spectral branch spans the registered annual window; historical public
encoder embeddings retain their original observation windows and identities.

An initial input-only screen uses the already registered shared 16/16/32 fusion
with the annual spectral branch, the same MLP/CNN, all twenty tasks and fixed
support/calibration domains. This is six new calibration jobs; reuse a result only
when its feature bytes and full sampling contract are unchanged. No region-specific
recipe change is allowed. Keep the original public-only/AEF anchors, and measure
the annual public-plus-spectral no-imagery control before crediting high resolution.
If the annual screen still fails, retain it and proceed to one common lightweight
learned residual-fusion method, with matched public-only fits and the same training
loss/normalization/selection rule in all regions. No primary evaluation maps enter
any encoder or feature fitting.

## Correct weak-supervision provenance

The legacy Haidian `worldcover` head reads OSM masks in all 320 manifest records.
Keep that compatibility name and all old model bytes; it is not evidence of ESA
WorldCover training. New targets use the explicit name `esa_worldcover`, official
ESA 2021 v200 sources, a provenance-bound raster, nearest-neighbor reprojection,
and twelve output classes (zero ignored, eleven actual land-cover classes).

The Haidian and Harbin official source rasters have archived checksums and official
bucket URLs. Auckland is a previously prepared local crop whose metadata lists the
official source URL; record its actual digest and derived-crop status rather than
pretend the crop has the global source's checksum. Cross-check its palette and
sample geography against the source before relying on it for new training. These
are weak training maps, not independent evaluation references. CLCD, Dynamic World
and LCDB remain excluded from embedding fitting. No manuscript changes or new
three-region claim follow data preparation alone.
