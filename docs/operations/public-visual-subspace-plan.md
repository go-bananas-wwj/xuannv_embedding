# Preserve public and direct visual coordinates separately

The learned public-fallback gate failed the complete three-region acceptance.
The next bounded test asks whether local visual information is lost before the
reader can use it. It bypasses the trained semantic/geometry fusers in the new
visual branch and reserves output coordinates for frozen dense visual features.
This is a hypothesis, not a demonstrated explanation or a new success claim.

Use the same previously chosen genuine-public time-modulated expert in every
region and every output role. Retain its existing weights and outputs. Public
input masks derive only from observed satellite availability, with no CLCD,
Dynamic World or LCDB label read during projection fitting. Public sampling,
scalers and public projection must be identical across native/public/degraded
roles within a region. Each region fits its own parameters by the same procedure.

Pool the original 1024D DINOv3 SAT493M native token maps with their observed
coverage. Average each actual view once. Use all six available real Haidian views,
five Harbin views, and one Auckland aerial observation; the public role uses its
actual public annual RGB view and the degraded role its matching degraded views.
Keep the existing dates and geographic/mask contracts. No monthly copies of the
Auckland aerial image are introduced. There is no learned weak-label gate in this
new branch. Frozen upstream public fitting and the shared global visual backbone
remain part of the full method and its provenance/costs.

Fit channel scalers on at most 65,536 valid label-free samples per branch, using
seed 41 and equal per-tile quotas with a fixed ordered remainder. Clip standardized
values to [-6,6]. Fit the public top-48 basis by centered covariance eigendecomposition;
fit the visual top-32 basis with randomized PCA, seed 41, 16 oversamples, four power
iterations and no whitening. Both candidates reuse these same fitted bases:

| Common candidate | Public coordinates | Visual coordinates | Total |
| --- | ---: | ---: | ---: |
| Public-preserving | 48 | 16 | 64 |
| Equal allocation | 32 | 32 | 64 |

Project native visual tokens before masked, normalized patch-center interpolation
onto the public 10 m grid. A coverage value at or below 0.05 yields a zero visual
block; a wholly missing visual branch must remain loadable and preserve its public
block. Public-domain-invalid output is zero. Concatenate the two blocks without a
final unit normalization. The result is one static period embedding, with no claim
of aligned coordinates or shared fitted weights across regions.

This differs from the older 223D joint-compression/manual-block trials by using
the newer common learned-public expert and uncompressed 1024D actual-view tokens,
before trainable semantic/geometry compression. Earlier compression failures and
negative source diagnostics remain retained; PCA itself is not claimed as novel.

Nine self-contained region/role specifications and both common allocations were
registered before new fitted parameters, exports or scores. The package and single
CLI entrypoint are now implemented. Twenty-five focused tests cover observed/missing
modalities, mask-aware alignment, projection reuse, cache identity, strict schemas,
constant inputs and cold reload. The full 1,171-test CPU suite, Black, Ruff, tracked
repository policy and package build passed; wheel/sdist production bytes match.
All nine real region/role payload, geography and physical-calendar checks passed.
Partial eight-tile/512-sample QA bundles and cold outputs passed with exact public
projection equality across roles. These are partial QA fits and never initialize
formal fits. Prior source lineage and current implementation hashes also match.
After a pinned source snapshot, fit nine projection bundles,
export eighteen 64D representations and run thirty-six fresh MLP/CNN readers for
all 240 unchanged balanced-v2 conditions. No neural encoder is newly trained.

Select one allocation globally by worst-six margin then mean, using the unchanged
nondecreasing public envelope and AEF. All six strict wins and every region's mean
AEF gain of at least 0.5 points remain required before frozen full-region queries.
Do not select different allocations or tasks by region/head. Retain all negative
records. Retire only rejected, reconstructible gate exports/readout grids after a
role/hash/hardlink/open-file/mapping audit; keep weights, source, saved predictions,
thresholds, all active public/raw-visual dependencies and original best bundles.
Keep a 100 GiB reserve. No manuscript body or accepted-model claim follows from
registration; complete source/rebuild, contribution and cost evidence still matter.

Only rejected fallback-gate annual outputs and deterministic evaluation grids
were removed: 50 paths representing 36 physical files, releasing exactly
31.711212 GiB by both allocation inventory and observed free-space delta. Free
space is about 147.894 GiB. All eighteen terminal weights, negative results,
predictions, thresholds, source and active public/raw-visual inputs remain.
No formal subspace fit, export or primary score preceded the completed checks.
The extra visual branch currently uses local optical imagery; public Sentinel-1
information remains in the public expert, and no new local SAR branch is claimed.

Formal fitting has started from immutable source commit
`e9111814764afa25bdfbae1cdf535cf496fd2642`, synchronized to GitHub after all checks.
The first three regional native projection bundles have completed; their paired
64D exports are running. A bounded controller uses three CPU fit/export workers
and then leases unused devices 1–7 for the fixed readers. Quiet dependent audit
and deterministic-grid observers are live. Device 0 is excluded. No complete
primary-task score or accepted common model is available at this point.

All nine formal projection bundles and eighteen exports have completed. The
matched-projection audit passed: public PCA/scaler parameters are bitwise equal
across the three roles within each region, source tile counts match registration,
and observed sample-position hashes match the saved records. Public sampling
includes all 320/422/320 regional tiles. Native visual samples cover 320 Haidian,
422 Harbin and 316 Auckland tiles; four Auckland tiles have no usable native
visual samples, and retain their public block. These observations are retained
explicitly rather than claiming complete aerial coverage. Seven NPU workers have
started the unchanged thirty-six MLP/CNN reader jobs. No complete primary result
is available at this stage.

The study completed all 36 fresh readers and 240 conditions and failed the
complete acceptance. Independent saved-prediction verification passed. No frozen
query was eligible or launched, and the public envelope remains unchanged. The
48/16 allocation is the best common case by the fixed minimum-six-margin rule,
with worst margin −3.1697 points; the 32/32 allocation has worst margin −3.5076.

| Region | Reader | Native 48/16 | AEF | Strongest public |
| --- | --- | ---: | ---: | ---: |
| Haidian | MLP | 82.005 | 83.875 | 84.659 |
| Haidian | CNN | 85.202 | 84.382 | 85.455 |
| Harbin | MLP | 56.398 | 58.396 | 58.065 |
| Harbin | CNN | 59.592 | 58.796 | 61.080 |
| Auckland | MLP | 61.542 | 63.434 | 63.337 |
| Auckland | CNN | 61.555 | 64.683 | 64.724 |

These are exploratory calibration macro F1 percentages, not blind geographic
performance. Both readers' regional mean AEF changes for the 48/16 allocation are
−0.5250, −0.6008 and −2.5100 points. All negative tasks, public and degraded
controls, projection bundles and source records remain retained. The current
comparison now contains seventeen common cases from nine independently audited
balanced-v2 studies, all ineligible; it is not the full historical experiment set.

Source geometry also exposed a limitation worth testing: the direct visual
branch has 40x40 native DINO tokens over a 1,280 m footprint, or 32 m per token.
The prepared RGB grid is 2 m, but the 16-pixel token patch spans 32 m. Interpolation
to the public 10 m grid adds no native visual detail. This statement applies to
this trial's direct visual branch, not every historical branch or the full mixed
embedding. First-tile checks in all nine region/role sources confirm existing
640x640 RGB working grids and permit explicit 5x5 subpixel groups. They are
feasibility/geometry diagnostics, not whole-region quality or causal proof.
