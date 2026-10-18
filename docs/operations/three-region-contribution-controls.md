# Contribution controls for a shared three-region winner

Register before observing wave-1 candidate scores. These controls follow a single
shared winning recipe; they are not additional city-specific selection options.
Keep the existing 20-support-tile, 100-update, MLP/CNN protocol, full reference
masks and all twenty tasks. Select no replacement recipe from control scores.

1. Genuine original public-only embeddings: the fixed AEF/public baseline anchors
   remain unchanged. They do not by themselves isolate the new spectral-statistics
   branch from high-resolution input.
2. Refit the same recipe without local imagery: replace the entire semantic source
   by zero before fitting, keep all allocated dimensions and use the same seed,
   fit positions, public embedding and corrected public statistics. Report its
   separate downstream calibration and full-region scores in all six slots.
   Public-aligned residual reduces exactly to the original public output when its
   only optional source is absent. Tail and joint recipes still have public
   spectral statistics; this is essential to distinguish spectral additions from
   high-resolution benefit. Blocks keep a neutral zero projection for an absent
   source, including when standardization was declared.
3. Frozen-model removal: zero only the semantic contribution after fitting the
   full model, keeping fitted public/spectral statistics. This answers reliance on
   the source at inference and is distinct from refitting the no-imagery ablation.
4. Control pretrained visual capacity: pass public Sentinel-2 RGB through the same
   SAT-493M encoder and projection rule instead of local high-resolution RGB.
   Freeze dates, quality masks, RGB band names, reflectance/display conversion,
   spatial footprint, resampling, crop geometry and output dimensions in an input
   registration before scoring. Upsampling public RGB provides no new observed
   spatial detail. Public imagery should cover the same available period; retain
   differences in sensor and observation dates as limitations rather than claim a
   perfectly controlled resolution-only experiment. Also compare local imagery
   degraded to the public grid, using the same encoder, to isolate its spatial
   detail more directly.

Run the six full-model queries only for the common eligible winner. If its full
scores fail a registered gate, keep that failure and do not promote another
previously inspected candidate using full-query scores. A subsequent method wave
requires a new explicit registration. Contribution controls cannot be omitted
because their scores weaken the desired interpretation. A raw-only diagnostic
winning against AEF would support useful public spectral features, not a local
high-resolution advantage.

Do not alter the manuscript's introduction/related work or claim a three-region
win during screening. The final method section must accurately describe whichever
shared fusion operation was actually validated; feature postprocessing is not
end-to-end retraining. All per-task results and negative controls remain in the
research record even when a shorter main table is prepared.
