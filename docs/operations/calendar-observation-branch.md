# Keep public observation seasonality in the physical branch

The four-source study completed both profiles, all three roles and both readers:
240 original-cohort task-reader conditions passed the support, target and geometry
contract. Both profiles still fail the common three-region acceptance gate. No
full-region query was launched. The following is the profile with the better
worst-slot calibration gain, selected once for all regions, not a final winner.

| Region | Reader | Native local | Public | Degraded local | AEF |
| --- | --- | ---: | ---: | ---: | ---: |
| Haidian | MLP | 68.616 | 65.203 | 67.048 | 75.139 |
| Haidian | CNN | 74.379 | 64.417 | 72.849 | 76.892 |
| Harbin | MLP | 57.325 | 54.923 | 54.160 | 56.120 |
| Harbin | CNN | 59.459 | 57.106 | 56.934 | 58.168 |
| Auckland | MLP | 59.647 | 54.021 | 52.944 | 58.045 |
| Auckland | CNN | 61.530 | 56.635 | 54.261 | 59.314 |

These are exploratory calibration macro F1 percentages, not blind or full-region
scores. The second profile also fails: its native MLP/CNN scores are 68.575/73.895
in Haidian, 56.997/59.193 in Harbin and 59.415/60.089 in Auckland. It additionally
loses both Haidian native-versus-degraded comparisons. All per-task and comparator
records remain retained. Older, stronger configurations are not overwritten.

## Evidence and one paired change

All 277 original Haidian forest calibration positives have nonzero learned,
public-visual, local-visual and physical branches. Nonzero branch presence does
not establish cloud-free observations or good features, but rules out an entirely
absent branch at those points. The selected profile's forest MLP has 138 true
positives, 193 false positives and 139 false negatives; AEF has 171/65/106.
Of its false positives, 98 are reference cropland and 95 reference impervious
surface, compared with AEF's 25 and 40. CNN has a different precision/recall tradeoff.
This is diagnostic evidence, not a causal explanation or changed scoring rule.

The existing physical descriptor is an orderless collection of annual statistics.
The calendar encoder uses dates, but those physical inputs currently have no direct
calendar-preserving route. Test one paired change: replace the 95 physical channels
with mean and annual/semiannual sine/cosine coefficients for the same nineteen
public bands. Five coefficients approximate the twelve monthly observations;
they do not retain every temporal detail or establish change-detection ability.

For each source and pixel, fit its observed months using actual YYYYMM dates from
the monthly cache. Fit an unpenalized constant and ridge-penalized annual and
semiannual terms, with one fixed penalty 0.1 in every region. Actual invalid months
are excluded rather than treated as zero observations. One observation yields its
constant value; an entirely absent source yields zero coefficients. The nineteen
bands are eleven S2, VV/VH and six Landsat bands, in their canonical named order.
The input is the same already clipped, region-normalized public monthly cache.
Output units are coefficients of those normalized observations, not reflectance.

Keep the selected four-source profile fixed: learned/public-visual/local-visual/
physical dimensions 8/8/24/24. The genuine public-only control uses 8/32/24 and no
local input. Native and degraded roles use their respective existing encoders and
frozen local optical features. No encoder retraining, new visual teacher, local SAR
or primary reference target is added. This is one source-description ablation,
not a new regional hyperparameter search. See the [four-source contract](common-four-source-fusion.md).

Fit the 95 coefficient maps once per region and share exactly those maps across
all roles. Bind inputs and outputs to the existing geographic index and hashes.
Use the monthly cache dates rather than the legacy spatial-index month list.
Preserve the original observation masks and all fixed evaluation masks. Save
source observation-count histograms and coefficient provenance. The exporter
requires no ancestor observations, local image values or map reference files once
the trusted public cache is available.

Then refit the same label-free projections and eighteen downstream readers for all
120 task-reader conditions. Keep every task, original support/calibration cohort,
seed, twenty-tile support budget and one hundred reader updates. The balanced
cohort proposal remains unapproved and unapplied. Require the same six strict
AEF/strongest-public wins and at least 0.5 percentage points mean AEF gain in each
region. Earlier public references remain in the envelope. Only an eligible common
candidate proceeds to full-region queries; no automated goal completion follows.

Regression tests cover known harmonic recovery, correct date association, joint
scene/date permutation, partial and missing sources, ignored masked values,
invalid ridge/calendar/shape/masks, geography and band metadata, map-free export,
repeatability, output corruption, unknown configuration fields and disk reserve.
Numerical reconstruction checks are separate from primary classification results.
No manuscript body or original introduction has been replaced.

Validation before launch: all 996 tests passed; Black, Ruff, repository validation
and both package builds passed. Wheel and source archives contain the exact new
module. Six real tiles across the three regions produced finite 95-channel maps
in 0.32–0.35 seconds per tile with two CPU threads, with lower observed-signal
reconstruction error than a constant baseline. This numerical gate does not prove
any classification improvement. The fixed reader and encoder weights are unchanged.
