# Shared fusion wave 1 (registered before candidate scores)

The completed baseline stage leaves the largest validation deficit in Haidian:
MLP/CNN public F1 66.04/66.05%, AEF 75.14/76.89%. Harbin's corresponding values
are 55.48/55.88% and 56.12/58.17%; Auckland's are 56.18/56.81% and 58.04/59.31%.
These are fixed calibration-region scores, not new full-region improvements.

This bounded wave evaluates eleven shared high-resolution candidates and one
public-statistics diagnostic. Every recipe produces 64 coordinates and uses the
same rule in all three independently fitted regions. Inputs are the genuine public
embedding, frozen SAT-493M local-RGB features, and corrected named-band public
seasonal statistics. No reference labels enter the feature exporters.

| Family | Preregistered settings |
|---|---|
| Public-preserving aligned residual | strength 0.25, ridge 0.01, kernel 1 or 3 |
| Public-tail residual, 16 semantic + 16 spectral directions | strength 0.1, 0.25 or 0.5 |
| Public-tail residual, 8 semantic + 24 spectral directions | strength 0.25 |
| Separate PCA blocks, public/semantic/spectral | 32/16/16, 24/16/24, 16/16/32, 40/8/16 |
| Standardized joint PCA | all three sources, 64 dimensions |
| Diagnostic, ineligible for high-resolution success | public statistics only, PCA64 |

Residual recipes retain the public vector when the optional source is absent.
Tail recipes inject into the lowest-variance 32 public directions; this algebraic
property does not guarantee better downstream scores. Component scaling and random
seed (41) are fixed. Fit positions use at most 65,536 unlabeled regional samples.

The statistics exporter explicitly names S2 bands. Historical raw-feature code
used red-edge/B8A positions for purported NDVI/NDWI; this wave uses B08/B04 and
B03/B08. Archived China S2 DN matches official asset samples; processing baselines
05.11 and 05.12 both specify quantification 10000 and BOA offset -1000. Auckland
acquisition already applied product scaling. Apply archived normalization/clipping
before the declared radiometry and retain all historical arrays/results unchanged.
These changes affect this new observation-statistics branch, not the protected
regional or AEF embedding arrays.

Run 72 fixed calibration jobs: twelve representations times three regions times
MLP/CNN, comprising 480 task fits. Keep all four CLCD, five Dynamic World and eleven
LCDB tasks. Choose one high-resolution recipe using the worst gain over the fixed
AEF/public anchors, then the mean. A candidate must win all six slots and exceed
AEF by at least 0.5 percentage points on each region's two-reader mean to receive
full-region evaluation. The raw-only diagnostic cannot be promoted. If no candidate
passes, save every result and register the next scientific hypothesis; do not pick
a different city winner or call the goal complete.

Numerical source is frozen in a Git snapshot for execution. CPU exporters precede
a dynamically balanced queue on free NPUs 1–7. The local finite controller handles
bounded retries, process identity, stop requests, progress receipts and the 100 GiB
disk reserve. No automatic Overleaf claim or paper change follows this screening.
Matched no-high-resolution and public-RGB pretrained controls remain necessary
before attributing a successful method to local high-resolution data.

Execution milestone: all 36 representations were exported before calibration.
The first reader startup failed because the operational snapshot launcher replaced
CANN's inherited Python module path, making `tbe` unavailable. No task scores or
checkpoints were produced. Preserve these failed startup logs, prepend the frozen
source path to the inherited environment, verify a real NPU convolution forward
and backward, then resume the original 72 registered reader jobs using the existing
arrays. This environment repair changes no recipe, split, target, budget or score.
The subsequent absent-source regression fix passed the complete 908-test suite;
its source is separate from the frozen numerical snapshot used for this wave.
