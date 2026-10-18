# One public-terrain-augmented shared fusion screen

Preregister before new terrain-augmented scores. AEF used GLO-30 and terrain
quantities as training targets; this candidate adds the public product directly to
regional inputs. This is a data-enriched product comparison, not equal-inference-
data proof of architecture superiority. The same public terrain is given to the
high-resolution and no-imagery versions in every region. LINZ LiDAR evaluation
references remain excluded. No primary classification labels enter feature fitting.

Use the corrected annual public statistics and original frozen public/local visual
features. Allocate exactly 64 coordinates by the same four-block rule everywhere:
12 public-temporal, 16 local-visual, 32 spectral, 4 public-terrain. Terrain comprises
public height and slope/aspect trigonometric coordinates; standardize by the same
unlabelled regional rule before fitting its four PCA directions. Keep projected
terrain magnitudes, and retain unit normalization of the other branches. Do not
unit-normalize the final concatenation. Seed 41, 256 unlabelled positions per tile
before the 65,536 cap; fit each region separately over all registered tiles.

The no-local-imagery version uses the already registered zero semantic source,
identical public/spectral/terrain arrays and identical fitting/scoring rules. All
three regions must have terrain with a matching geographic-index identity. Reject
a missing region or mismatched grid rather than testing terrain only in Haidian.

Run the original twenty tasks with unchanged source masks, label/support positions,
MLP/CNN, 20 support tiles and 100 updates. Select no region-specific setting. Require
strict gains in all six region/reader slots over AEF and the strongest registered
public control, including the matched terrain counterpart and previous public
variants. The per-region two-reader mean gain over AEF must reach 0.5 percentage
points. Record every negative class result. Only an eligible common candidate gets
full-region scoring; this source change is not permission to change test masks.

The visual-control queue currently owns NPUs 1–7. Prepare terrain fusion features
on CPU while that queue runs, then wait for its actual process to terminate before
starting the new readouts. Bound waits/retries/timeouts, preserve the 100 GiB reserve
and prior best bundles, and reuse unchanged results. No Overleaf edits or universal
win claims follow data preparation or software tests.
