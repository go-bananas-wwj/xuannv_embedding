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
registered before new fitted parameters, exports or scores. Implementation is
pending. Add failing-first coverage, missing-modality, alignment, projection-reuse,
strict-schema and reload tests; then run software/package and all real-data gates.
Check public projection equality across roles and truthful observed-sample counts.
Only after those checks and a pinned source snapshot, fit nine projection bundles,
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
