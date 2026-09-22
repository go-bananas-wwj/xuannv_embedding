# Frozen neural classification readouts

`downstream.neural_readouts` calibrates MLP and 3×3 convolution readouts on explicit
training support and validation maps, saves numeric parameters, and replays inference
without labels or optimizer updates. It is a component for the final shared strong-head
controller; it does not select support tiles, bind external geography, lock candidates,
or perform the actual final comparison. Existing pinned experiments remain unchanged.

The caller supplies paired NCHW float32 feature maps, NHW boolean feature-validity masks,
and binary targets with -1 for missing labels. Keep the original feature dimension and
common tile order. Both calibration splits require both classes on their valid labeled
domain; every training support tile must contain a labeled valid pixel. StandardScaler
fits only labeled, feature-valid training support pixels. Feature-valid pixels with
missing labels can supply convolution context but do not contribute to the scaler or
loss. Missing-feature context is zero after standardization, identically for all methods.
Predictions at missing-feature positions are NaN; the caller must intersect output validity
and label validity before scoring. This explicit standardized/masked protocol is separate
from historical unstandardized fully labeled probes.

Architecture comes from `heads.build_head`: MLP D→128→1 with 1×1 convolutions and ReLU;
Conv D→128→64→1, two 3×3 layers, BatchNorm, ReLU and dropout 0.1, then a 1×1 output.
Training uses exactly 100 AdamW updates, learning rate 0.001, weight decay 0.01, batch size
2 (or all support if only one tile), float32 without AMP, and binary cross-entropy with
training negative/positive ratio clipped to [1, 50]. No validation early stopping or epoch
selection occurs. These retain the existing probe architecture and fixed optimization
budget. The head initialization seed is 41. A separate CPU generator with seed 41 draws
each batch without replacement, making the sequence independent of feature dimension,
architecture construction, ambient RNG and dropout. Dropout RNG is reset after construction.
All 100 losses, batch positions, initial/final weight hashes and fit time are recorded.

Validation chooses only the F1 threshold; AP and exact full-map prediction hashes are
recorded. Inference uses evaluation mode, frozen parameters and one tile per batch,
preserving BatchNorm statistics and making batch boundaries independent of cohort size.
The caller can select CPU or an explicit NPU device index. CPU thread count, ambient CPU
RNG and the selected NPU RNG/device context are restored. Calibration/replay must use the
same recorded backend and numeric runtime; cross-backend bitwise equivalence is not assumed.
Physical NPU reservation and CANN environment setup remain the launcher's responsibility.

`save_neural` writes a numeric NPZ (including BatchNorm counters, scaler statistics and
batch schedule) and JSON identity, with no pickle model payload. `load_neural` requires
an externally registered identity digest, exact implementation and runtime identity, and
payload digest before parsing; it checks tensor names, shapes, dtypes, finite parameters,
training batch identity and final weight digest. `verify_validation` rejects changed inputs,
labels, masks or predictions and never retrains. Use exclusive new directories for every
artifact. Batch positions are relative to the caller's support tiles; a shared controller
must still bind them to registered canonical spatial positions.

Synthetic tests cover learned weights, finite training, exact save/reload prediction replay,
no optimizer step during replay, unchanged BatchNorm buffers, independent batch RNG,
validation-label exclusion from fitting, support-only scaling, missing context, invalid
inputs and modified payload/runtime rejection. A separate synthetic NPU gate checks both
heads' forward/backward and saved prediction replay on a reserved idle device. These checks
are software evidence, not measured superiority of any geographic embedding. The shared
strong-head file workflow is provided by `strong_multitask`; actual locked-candidate
evaluations and their paired uncertainty report remain required.
