# Preserve public and local visual features separately

The completed three-branch visual skip improves over AEF in both readers in
Harbin and Auckland, but still fails in Haidian. All 120 original-cohort
task-reader conditions were retained, including degraded-local controls.

| Region | Reader | Native local | Matched public | Degraded local | AEF |
| --- | --- | ---: | ---: | ---: | ---: |
| Haidian | MLP | 69.379 | 66.618 | 67.129 | 75.139 |
| Haidian | CNN | 72.305 | 68.812 | 73.691 | 76.892 |
| Harbin | MLP | 56.734 | 54.720 | 55.293 | 56.120 |
| Harbin | CNN | 59.348 | 57.117 | 57.739 | 58.168 |
| Auckland | MLP | 61.057 | 56.707 | 56.599 | 58.045 |
| Auckland | CNN | 62.324 | 57.192 | 57.381 | 59.314 |

Values are exploratory calibration macro F1 percentages, not full-region or blind
test scores. Native local imagery also loses to degraded imagery for Haidian CNN.
The fixed overall gate fails, and this candidate receives no full-region query.

The direct skip retained local visual features and public physical observations,
but public visual features still passed through the learned encoder. Label-free
reconstruction audits found weak recovery of those visual inputs. Furthermore,
the fitted sixteen-component learned projection retains over 99 percent of
learned feature variance, whereas sixteen local visual components retain only
66–72 percent of their input variance. Variance is not task accuracy; these
observations motivate a measured change rather than establish a fix.

## Fixed follow-up

Keep four observation branches: role-matched learned calendar representation,
frozen annual public S2 visual features, frozen local visual features, and public
physical observation summaries. Add a distinct `public_visual` source to the
existing label-free exporter. It must contain 64 channels and bind to exactly the
same geographic index. Missing regions, incompatible width and mismatched index
are rejected before scoring. No reference map is read by the exporter.

Register exactly two dimension allocations before their exports or reader scores:

| Common profile | Learned | Public visual | Local visual | Physical | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| More physical observations | 8 | 8 | 24 | 24 | 64 |
| More local visual detail | 8 | 8 | 32 | 16 | 64 |

All three regions use both registered profiles. Each profile uses the same
dimensions in its native-local and degraded-local role. The public-only control
uses its genuine public-only calendar encoder and reallocates the entire local
visual budget to the public visual branch: 8/32/24 or 8/40/16. It has no local input,
no duplicate public block filling nominal dimensions, and the same total budget.

Use already trained fixed-budget regional encoders without additional neural
training. Public visual features come from all twelve months with the same frozen
SAT493M weights and extraction rule in every region. Fit regional projections
with seed 41 and at most 65,536 observation samples, normalize branches and the
64D output, and standardize physical observations before their projection.
Training weak targets remain genuine WorldCover 2021; primary CLCD, Dynamic World
and LCDB references are never upstream targets. Local SAR is absent. Terrain is
an existing calendar-encoder input in all roles, not an additional direct branch.

In the engine the legacy name `public` is the role-matched calendar representation.
Native/degraded representations include their local input and are not public-only
features. The new name `public_visual` specifically denotes the frozen public
imagery route. These meanings must remain explicit in provenance and paper text.

Keep the original twenty tasks, masks, support/calibration positions, reader
architecture, seeds, twenty-tile budget and one hundred updates. Both profiles,
all roles and both readers produce 240 task-reader calibration conditions. The
balanced-cohort proposal remains unapproved and is not applied.

Select one common profile by its worst of six calibration gains, then its mean.
The public envelope includes both new public profiles and the previously strongest
genuine public references. Require all six strict AEF/public wins and at least
0.5 percentage points mean AEF gain within each region. Only the selected eligible
profile may receive full-region queries. Do not assemble a different profile per
region or head, drop tasks, rank from full-query scores, or automatically declare
the goal complete from a calibration result.

Regression tests first reproduced unsupported-source rejection and silently
ignored incompatible public-visual provenance, then cover both public/native
layouts, map-free export, repeatability, missing-region rejection, width and
geography checks. Runtime registrations, logs and data remain outside Git. Original
best bundles, negative results and the manuscript introduction stay preserved.

Validation before launch: all 980 tests passed; Black, Ruff, the repository gate
and source/wheel builds passed. The change does not modify neural architecture,
checkpoint loading or the fixed reader implementation. Actual regional exports
and their downstream performance still require the registered experiment.
