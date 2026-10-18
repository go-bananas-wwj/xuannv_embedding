# Completed spatial-detail and pretrained-capacity diagnostics

The fixed three-branch fusion diagnostics completed all eighty registered
task-reader conditions. Supports, reference values and calibration counts match
both original baseline contracts. Results below are calibration macro-F1 in
percent across the four CLCD, five DW and eleven LCDB tasks respectively.

| Region | Reader | Native local RGB | Same local RGB degraded to 10 m | Public S2 RGB, same image encoder | Native minus degraded, pp |
| --- | --- | ---: | ---: | ---: | ---: |
| Haidian | MLP | 67.827 | 66.592 | 65.839 | +1.235 |
| Haidian | CNN | 72.058 | 70.918 | 70.760 | +1.141 |
| Harbin | MLP | 57.975 | 55.603 | 55.979 | +2.372 |
| Harbin | CNN | 59.587 | 57.141 | 57.492 | +2.446 |
| West Auckland | MLP | 61.158 | 56.565 | 56.229 | +4.593 |
| West Auckland | CNN | 62.178 | 57.281 | 56.362 | +4.898 |

Native images exceed their degraded counterparts in all six slots. That contrast
holds image identity, dates, fine validity masks, frozen encoder and the regional
projection/readout rule fixed; it supports a spatial-detail benefit for this
specific fusion. Native images also exceed the public-RGB replacement in all six
slots, but different sensors, dates and compositing limit causal interpretation
of that second contrast. These are single registered support draws, not evidence
of statistical significance over repeated independent training runs.

These diagnostics retain the historical regional public embeddings. Their
upstream neural losses and map targets differ by region. They therefore do not
establish the new common end-to-end method, and must not be reported as the
uniform observation-only candidate's results. That candidate has its own native,
public-capacity and degraded-local controls.

The separate terrain-augmented historical screen also completed eighty audited
conditions:

| Region | Reader | Local-image fusion | Matched public fusion | AEF |
| --- | --- | ---: | ---: | ---: |
| Haidian | MLP | 67.883 | 65.153 | 75.139 |
| Haidian | CNN | 74.063 | 76.330 | 76.892 |
| Harbin | MLP | 57.486 | 55.113 | 56.120 |
| Harbin | CNN | 60.026 | 57.869 | 58.168 |
| West Auckland | MLP | 61.108 | 58.244 | 58.045 |
| West Auckland | CNN | 59.795 | 55.940 | 59.314 |

This candidate fails the shared acceptance criterion; no new full-region queries
were run. Haidian still trails AEF in both heads, and its CNN also trails the
matched public fusion. Auckland's mean AEF gain does not rescue Haidian. Retain
all per-task rows and both reference conditions rather than selecting favorable
readers or regions.

Input-availability audits further distinguish buffer coverage from scored
coverage: Harbin's native local visual vectors cover about 86.55% of all tile
pixels but 100% of its valid primary reference pixels. Missing buffer imagery
does not explain a primary-score deficit. Auckland coverage is audited against
each task's valid reference mask; a coarse native reference mask is never naively
resized to the embedding grid for this diagnostic.
