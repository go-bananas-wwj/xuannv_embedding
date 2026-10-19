# Keep public coordinates and add within-cell visual detail

The public/visual coordinate-allocation study failed. Its visual branch has a
32 m native token grid even though its prepared RGB inputs are on a 2 m working
grid. The next registered test uses the RGB values inside each public 10 m cell
and keeps the original full public embedding. It tests a concrete information
bottleneck and public-coordinate preservation; classification improvement remains
unproven. Local sensor resolution and a resampled working grid are distinct.

Keep the same common genuine-public time-modulated expert, independently fitted
in each region and frozen. All three output roles use exactly its same regional
64D features. Reuse the original physical masks, source index and actual RGB
manifest/calendar. Average registered observations/composites once at each known
working-grid pixel, with masks, then pack each 5x5 RGB group into 75 coordinates
and 25 coverage values. Public and degraded RGB also lie on a 2 m working grid;
this does not claim they contain independent 2 m observations. No monthly copies
of Auckland aerial imagery or artificial changes to source observations are made.

Subtract each cell's coverage-weighted RGB mean, creating a mean-free within-cell
highpass descriptor. Unknown centered values are zero with their coverage kept
separately. Fit three observed highpass RMS scales, with a one-digital-unit floor;
clip standardized values to [-6,6], then recenter known values. Fit a centered
75-to-16 covariance PCA from at most 65,536 label-free cells per region/role, using
seed 41 and equal per-tile quotas. At least two known subpixels are needed to
measure within-cell variation. Do not remove those cells from evaluation: an
absent visual correction returns the unchanged public embedding.

Fit the public channel mean/std and covariance basis from the same public-only
observations, identically across all three roles in a region. Use the bottom
sixteen public eigenvectors for the added correction, leaving the complementary
forty-eight directions protected. Whiten private PCA scores with a 0.01 standard
deviation floor, suppress zero-variance components, apply tanh and mean coverage,
then map the sixteen values into the public tail. Add the resulting bounded
correction, scaled by public channel std, to the original full 64D public vector.
The saved vector keeps the original public coordinates and does not truncate or
replace its tail. Protection is an FP32 linear-algebra property; exported float16
quantization error must be checked separately.

Only two common correction strengths, 0.1 and 0.5, enter selection. At strength
0.5 the standardized correction norm is at most two before coverage attenuation,
and the protected public projections remain unchanged in exact arithmetic. An
initial pre-execution draft used 0.02/0.1; it is retained with its hash and an
explicit amendment before any new cache, parameters or task score. The revision
adds a visibly larger bounded-signal probe, not a regional or task-specific choice.
No new neural encoder or primary/weak-map target is fitted in this covariance
trial. A convolutional reconstruction encoder would be a separate future test,
only if source evidence and full results justify it.

Nine source-cache specifications and the full common recipe are registered;
implementation and data preparation are pending. Add failing-first tests for
frame pooling, pixel packing, geography/calendar/source integrity, masks, missing
modalities, mean-free descriptors, zero variance, public-basis equality, identity,
residual bounds, protected projections and cold reload. Run full software/package
checks and real regional source/partial-projection/inference checks before formal
parameters. Retire only rejected study exports and deterministic readout grids
after role/hash/hardlink/open-file/mapping checks. Keep all projection parameters,
source, results, predictions, thresholds and active raw/reference dependencies.

Then generate nine pixel caches, fit nine projection bundles, export eighteen
64D representations and run all thirty-six fresh MLP/CNN readers and 240 unchanged
balanced-v2 conditions. Select one strength globally by worst-six margin then
mean. The public envelope can only increase; six strict AEF/public wins plus each
region's mean AEF gain of at least 0.5 points remain mandatory before frozen
queries. No favorable task/reader/region splicing or manuscript success claim is
permitted. New pixel caches are about 9.72 GiB and candidate/readout arrays about
31.71 GiB; budget 44 GiB peak and keep a 100 GiB reserve. No manuscript body edit
follows from this registration.
