# Frozen common-candidate full-region diagnostic

One common candidate was frozen by calibration minimax ranking before these new
query results: the eleven-coordinate semantic complement. The same recipe and
regional fitting rule are used in all three areas. No per-region candidate,
reader, task or threshold was changed after freezing it.

This is an explicit screening-protocol deviation. Earlier screens queried only
calibration-eligible candidates. This one calibration-ineligible candidate was
additionally queried to diagnose the calibration/full-region coverage difference
under the user's requested complete regional mapping scope. Its negative
calibration status and original acceptance flag remain unchanged. These queries
do not become a blind test or a new parameter-selection set.

All22 new query jobs completed. Original AEF/public query results were reused for
all12 region-reader-baseline combinations. The added public references were
chosen by calibration only. Audits verified every primary reference pixel count,
unchanged support descriptors, identical saved readout identities and unchanged
calibration thresholds. No readout training was repeated for these queries.

Full-region macro-F1 in percent:

| Region | Reader | Frozen high-resolution candidate | Matched public | AEF | Candidate minus AEF, pp |
| --- | --- | ---: | ---: | ---: | ---: |
| Haidian | MLP | 80.721 | 79.694 | 82.500 | -1.779 |
| Haidian | CNN | 84.436 | 84.406 | 83.067 | +1.369 |
| Harbin | MLP | 56.862 | 55.374 | 57.834 | -0.972 |
| Harbin | CNN | 62.294 | 59.230 | 60.857 | +1.437 |
| West Auckland | MLP | 55.853 | 52.186 | 59.706 | -3.853 |
| West Auckland | CNN | 59.753 | 51.816 | 61.973 | -2.219 |

The diagnostic fails the three-region goal too. Calibration coverage matters,
but cannot by itself explain the remaining deficits. Haidian and Harbin CNN
comparisons do not offset the MLP and Auckland decreases. All twenty primary
classes, original public comparisons, calibration-selected stronger public
references and degraded-image results remain in the result artifacts.

The outputs include downstream support and calibration locations and therefore
describe exploratory in-region map agreement. The original320/422/320tile grids,
reference masks and annotation budgets were preserved. The proposed balanced
calibration cohort has not been applied and still requires the user's choice.

All-year public image descriptors are also complete: the Chinese regions were
encoded from their twelve-month composites and Auckland's identical existing
annual representation was reused. Those new descriptors were not substituted
into this frozen diagnostic. Their PCA bases differ from the former Chinese
inputs, so old regional auxiliary weights cannot silently be treated as trained
on the new descriptors.
