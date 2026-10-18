# Registered goal: exceed AEF with a regional high-resolution representation

The user explicitly enabled goal mode on 2026-10-18. The goal is active until the
registered acceptance criteria are actually met, or the user pauses it. Launching a run
or improving a single class/split is not completion.

## Fixed acceptance protocol

Use the existing 320 Auckland-west tiles, 2024 data, 64-dimensional representations,
eleven eligible LCDB classes, unchanged spatial folds, common masks and label locations.
Primary readouts are the same MLP and CNN at 200 points/class. The target is to exceed
AEF (MLP 57.00%, CNN 56.11% pooled Macro F1), which also exceeds the original public
model (54.37%, 53.53%). Use exact machine-readable baseline values for comparisons.
Report class metrics and LiDAR regression without hiding declines. Do not alter
Overleaf during goal work.

The benchmark has already been inspected and adaptively tuned. Any resulting pooled
score is an exploratory regional result, not a new blind evaluation. Preserve this
qualification and the separately recorded fold-zero test outcome.

## Current gap and next wave

Longer training improves CNN but not MLP. Validation selection based on the mean of
the two can favor a CNN gain while sacrificing MLP. Subsequent representation searches
will maximize the smaller of the two validation improvements over the corresponding
AEF validation readout; ties use the validation mean. No test score selects a candidate.

The next inexpensive wave tests information preservation before further long training:

1. Late fusion of existing public/high-resolution features, using an explicit fixed
   64-D projection fitted only on unlabeled image features. Include original public as
   an eligible fallback and keep the underlying encoders unchanged.
2. Extract frozen DINOv3 ViT-L/16 SAT-493M features from the existing 2024 LINZ RGB,
   then test 64-D semantic features and public/semantic fusion. The local checkpoint
   is available; no AEF feature is an input or training target for our candidates.
3. Only if a representation shows validation gains, run the registered complete test
   and regression suite. Use the outcome to decide whether further projection/fusion
   training, stronger local image resolution or semantic distillation is warranted.

DINOv3 is additional global satellite pretraining, not a model trained from scratch
on these 320 tiles. Any successful method must disclose its use. The existing 3 m RGB
differs from SAT-493M's pretraining resolution, which is another diagnostic dimension.
Official references: [DINOv3 model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md),
[reference implementation](https://github.com/facebookresearch/dinov3).

## Execution discipline

Keep compact results, recipes, code and stage milestones on GitHub. Weights, caches,
native feature maps, projections, figures and detailed logs remain under the data
workspace. Maintain geographic indexes and source hashes. Preserve every negative
validation/test result. Use disjoint available NPU devices, retaining the unrelated
device-zero service. A local supervisor should poll quietly and resume completed jobs;
the language model should not repeatedly inspect unchanged training state. Do not
start another long sweep merely because hardware is idle.

## Wave-one registration

Seventeen fixed 64-D representations are registered: twelve late-fusion recipes from
existing encoders, four public/DINO semantic fusions and one DINO-only diagnostic. The
diagnostic cannot satisfy the regional-fusion goal on its own. This is a feature search,
not seventeen additional encoder training runs. PCA uses at most 65,536 unlabeled feature
samples and never reads LCDB or LiDAR labels. Readouts retain their original training
rules and support points.

The SAT-493M extractor strictly loads the existing local ViT-L/16 checkpoint. Restore
physical RGB from dataset standardization, divide by 255 and apply the satellite model's
own mean/std (not a display contrast stretch). Remove class/register tokens. Right/bottom
padding is accounted for using actual source-pixel centers when aligning native patches
to the common 10 m footprint. Missing high-resolution observations stay missing; they do
not change the common scoring mask. Two real urban tiles passed NPU extraction and PCA
checks before the full run. Optional installation is `pip install -e '.[downstream,vision]'`.

## Native aerial and spatial-context waves

The native aerial wave and all class/regression results are recorded in
[the native-wave result](auckland-2024-goal-native-results.md). It improves pooled MLP
beyond AEF but not CNN, so the goal remains active. The next six fixed candidates
aggregate fused features over 30 or 50 m using 16, 24 or 32 semantic coordinates.
Selection retains the previous validated candidate and uses only registered validation
metrics. This is label-free latent aggregation, not ground-truth or prediction smoothing.

## Spatial-context attribution controls

After validation selects the context recipe, evaluate the same fixed 5 by 5 aggregation
on AEF64, original public64, and public PCA48 with sixteen zero coordinates. Use the
same 64-D unit-vector averaging, support points, folds, readout training and validation
selection as the fused representation. These three controls are not eligible candidates
and cannot select or modify the fusion winner. Report them alongside the frozen
original baselines: a gain from generic spatial aggregation alone must not be attributed
to high-resolution imagery. This is a classification-only diagnostic, with no new raw
imagery, encoder training, labels, or reference-map inputs.

## Acceptance outcome, 2026-10-18

The registered classification target was met at 14:42 Beijing time. The selected
64-D native-aerial/public representation with 50 m context reaches pooled Macro F1
58.6834% (MLP) and 56.8089% (CNN), above original AEF by 1.6838 and 0.7027 percentage
points. It also exceeds AEF with identical 50 m aggregation by 1.0699 and 0.5241
points. The original public model and both context-matched public controls are lower.

[Complete results, all classes, spatial folds and regression](auckland-2024-goal-context-results.md)
include remaining weaknesses: several classes and all four LiDAR regression targets
remain below AEF for this classification-selected representation. This completes
the fixed classification goal, not a claim of superiority on every task. The benchmark
was used adaptively and is not a new blind test. No additional search is required for
this acceptance, and no Overleaf changes were made.
