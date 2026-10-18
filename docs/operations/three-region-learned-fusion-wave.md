# Shared learned residual fusion (registered before new downstream scores)

The annual-input screen failed: Haidian MLP/CNN F1 67.04/68.13%, Harbin
56.67/58.76%, Auckland 61.16/62.18%. Relative to the previous short-window fusion,
Haidian lost 2.68/3.25 pp and Harbin 2.92/3.21 pp. Auckland's feature bytes were
unchanged; all 22 existing task-reader results were reused through the audited
archive API. All forty task-reader support contracts match both fixed baselines.
No annual candidate received a full-region query. Temporal coverage alone has
not explained or fixed the performance gap.

## One architecture and loss, six paired fits

Retain both the original 95D statistics and the annual 95D statistics, rather than
replace the earlier information. Every regional model receives its genuine frozen
64D public embedding, the 190D spectral stack, and optionally the same frozen 64D
SAT-493M local-RGB representation. Public backbone and satellite vision encoder
weights remain frozen. Fit source-specific normalization from unlabeled regional
positions by the same rule; public/spectral statistics never depend on imagery
availability. No city names or region-specific algorithm branches enter core code.

A small convolutional public adapter produces a zero-initialized 64D residual of
the public/spectral input. Add the existing availability-masked high-resolution
residual module. Its correction also starts at zero: the initialized output equals
the original public vector exactly. If local imagery is missing, the result equals
the learned public branch exactly, including after training. Output width stays 64.

Fit twelve-class ESA WorldCover weak supervision (zero ignored), standardized
spectral reconstruction, optional semantic-feature reconstruction and cosine
anchoring of the public branch. Loss weights are respectively 1, 0.2, 0.1 and 0.02.
WorldCover classes use square-root inverse-frequency weights clipped to 0.25–4;
absent classes have zero weight. Use the same public-observation mask for weak
labels in both fits: adding imagery cannot add labelled pixels. Local imagery may
add its own unlabeled reconstruction targets. OSM remains inherited in historical
public encoders; the new categorical target is explicitly true ESA WorldCover.
CLCD, Dynamic World and LCDB arrays never enter training or normalization.

| Setting | Shared value |
|---|---|
| Public adapter width | 128 |
| Bounded public residual scale | 0.25 |
| Spectral channels | 190, original + annual |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.01 |
| Updates / batch | 1200 / 16 complete native tiles |
| Schedule | 30-update warmup, cosine decay to 5% |
| Gradient norm limit | 1 |
| Seed / augmentation | 41, common deterministic horizontal/vertical flips |
| Normalization fit | at most 65,536 unlabeled positions, 256 per tile before cap |
| Selected checkpoint | final update 1200; no per-region checkpoint selection |

Each of the three regions has a high-resolution fit and a no-imagery fit. The
latter never reads local semantic arrays, uses all public inputs and identical
WorldCover labels, sampler order, normalization rule, optimization and model width.
Every registered tile must participate; save the actual coverage and common weak
label-domain digest. This is learned adaptation of frozen features, not end-to-end
retraining of either encoder. A synthetic NPU gate is software evidence, not a
regional model score.

## Evaluation and remaining contribution checks

Run the unchanged MLP/CNN readers and all twenty registered tasks. Preserve exact
support/validation masks and the fixed AEF/public anchors. The shared high-resolution
method must beat AEF, the original public baseline and its newly trained public
counterpart in all six region-reader slots; its per-region two-reader mean gain
over AEF must reach 0.5 pp. Use the same worst-slot rule, never a city-specific
winner. Only an eligible common candidate receives new full-region scoring.
Keep all negative task results and all failed methods.

Before attributing gains specifically to local spatial detail, also control for
SAT-493M's pretrained capacity using public RGB input and degraded local imagery.
Those controls remain required if the paired method wins; do not credit the extra
pretrained encoder entirely to image resolution. These repeatedly inspected
regional maps are exploratory agreement, not a new blind generalization test.

Checkpoint writes are atomic and resumable. Keep the latest recovery state during
training and only the final compact model once it is strictly loadable. Recovery
also handles interruption after the final weight write but before the status
receipt. Source/spec identities are immutable. Run six independent jobs on free
NPUs with bounded retries and a 100 GiB reserve; preserve existing best bundles,
source rasters, all earlier results and the current Overleaf body.
