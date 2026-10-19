# Approved balanced calibration, version 2

On 2026-10-19 the user approved adding the previously proposed balanced cohort
and preserving all original results. This supersedes earlier pending-approval
notes. The exact tile positions were generated before approval using seed 41 and
reference class counts only; they are now frozen without looking at version 2
model scores. Existing version 1 artifacts are not overwritten or relabelled.

| Region | Registered tiles | Calibration tiles | Training support pool |
| --- | ---: | ---: | --- |
| Haidian | 320 | 67 | Unchanged |
| Harbin | 422 | 63 | Unchanged |
| Auckland | 320 | 64 | Unchanged |

The class-count balancing rule targets the whole reference area's class
prevalence, with at least five positive tiles per class where feasible. It is
not an equal-pixel-count resampling of every class. Empty reference/buffer tiles
are excluded; all selected tiles are outside the unchanged training support pool.
Haidian forest calibration coverage changes from 277 to 18,289 positive pixels.
All proposed class counts, reference hashes, tile counts and pool separation were
recomputed before creating any version 2 scores.

No reference masks, class codes, feature arrays, embeddings or encoders change
because of this new split. New data files share the immutable original feature,
label and mask arrays and differ in calibration positions and version identity.
They keep the original sample seeds, twenty-tile support budget, both readers and
one hundred reader updates. Every affected reader is fitted afresh with empty
archive settings. Original heads, thresholds, predictions and full-region scores
cannot serve as newly calibrated version 2 baselines.

## Fixed candidate and reference panel

Freeze three common candidates before any version 2 scores:

1. Calendar representation with local visual and public physical skips.
2. Separate public/local visual skips, using the previous common 24-dimensional
   local profile in every region.
3. The same four-source profile with calendar-associated physical coefficients.

Each candidate includes its genuine public-only and degraded-local controls in
all three regions. The third candidate's completed version 1 experiment failed:
its native MLP/CNN scores were 66.744/70.738 percent in Haidian, 56.426/56.949 in
Harbin and 56.350/59.436 in Auckland. It remains in this fixed panel as a paired
source ablation; numerical reconstruction improvement was not classification
improvement. The other two candidates and all their negative version 1 results
also remain preserved.

Add AEF, the original genuine public baseline and a previously selected strong
public reference for each regional reader. These references are all re-fitted
under the same new cohort. Their version 1 numbers only identify configurations;
no version 1 number is used as a version 2 comparison value. The new public
envelope is computed from the freshly scored registered public references and
the three matched public controls.

This gives 72 fresh reader jobs and 480 task-reader calibration conditions across
the same twenty primary tasks. There is no per-region candidate substitution.
Choose one common candidate by worst six-slot gain, then mean gain. Require every
region/reader to beat AEF and the registered public envelope, plus at least 0.5
percentage points mean AEF gain within every region.

Only an eligible common candidate receives full-region queries, together with
freshly calibrated AEF, the original public baseline, its own public/degraded
controls and public references selected using calibration alone. These queries
use frozen version 2 heads and thresholds; they do not select another candidate.
The full comparison must separately pass before considering a paper claim.
No controller automatically completes the overall goal.

All reference maps, new cohort positions, approval evidence, immutable source
identities, job manifests and detailed results stay outside Git. The existing
data-only experiment is allowed to finish in its original namespace; the new
queue waits for its actual owner to release the devices. Bounded jobs, disk
reserve checks and free-device leases cover devices 1–7; device 0's unrelated
service remains untouched.

The new cohort is exploratory. Repartitioning previously inspected areas does
not create a blind test. Full-region scores include support/calibration locations
and describe regional map agreement, not same-weight unseen-region transfer.
No manuscript result has been replaced by an unsupported improvement claim.
