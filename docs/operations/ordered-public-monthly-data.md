# Preserve ordered public observations for the next regional encoder

The existing95-channel observation summaries contain per-band mean, standard
deviation, minimum, maximum, availability and optical-index summaries. They are
invariant to month permutation, so their values do not retain seasonal order.
This motivates preparing the original ordered observations before testing a new
temporal encoder; it does not demonstrate a performance improvement.

`xuannv experiment public-monthly-tensor --spec /path/prepare.json` produces:

- Float32 frames shaped `[tiles,12,19,height,width]` in canonical band order:
  eleven Sentinel-2 bands, VV/VH and six Landsat bands.
- Boolean masks shaped `[tiles,12,3,height,width]`; unavailable observations stay
  zero without removing a calendar position or inventing an observation.
- Twelve consecutiveYYYYMMtimestamps, explicit band slices, source/index hashes
  and fresh normalization moments fitted only from valid public observations.

The strict specification binds a named public-observation registration, its
period provenance and a geographic index, plus an output directory and disk
reserve. Physical NPZ and archived normalized public tensors are supported. The
latter are decoded with their archived statistics; additional availability
channels are excluded from the physical bands. Apply declared band radiometry,
reorder by band name and fit regional mean/deviation before clipping normalized
values to[-6,6]. Entirely absent modalities retain zero masks and safe statistics.

The producer reads public frames, public reconstruction-validity masks and
timestamps only. It does not use local-image or map-label values. It verifies
intrinsic tensor timestamps where available, source record order, all band
names, valid-value finiteness and source identities. Existing completed outputs
are hash-checked, and cache identity includes the imported radiometry code.

Real preflight checks passed at the first and last record of every region:

| Region | Tiles | Ordered observation window | Physical bands |
| --- | ---: | --- | ---: |
| Haidian | 320 | June2025–May2026 | 19 |
| Harbin | 422 | November2024–October2025 | 19 |
| West Auckland | 320 | January–December2024 | 19 |

All three outputs follow the same preparation rule with separate regional
statistics. This changes neither the pending calibration-cohort choice nor any
active benchmark labels, thresholds or readouts. No new encoder has yet been
trained from these tensors. A future temporal/multisource architecture must be
registered, fitted independently in every region under one rule, and assessed
against the full fixed AEF/public comparisons before any paper claim.
