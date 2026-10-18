# Public RGB control for the frozen satellite vision encoder

The local-image 10 m degradation control preserves its original dates, colors and
fine observation mask. A second control is needed to distinguish local observations
from the extra SAT-493M pretrained visual capacity.

Prepare Sentinel-2 RGB from public observations only, using named B04/B03/B02,
archived normalization, declared physical scale/offset, quality masks and the
original registered experiment months in every region. For China, select those
six months from the corrected annual cache; Auckland uses its existing twelve
months. Median only valid monthly observations. Map reflectance 0–0.3 to 0–255
with the same fixed rule everywhere, then resample to the same 0.5 m encoder grid.
Upsampling introduces no new observed spatial detail. Source hashes, selected
months, actual masks and both native/virtual raster dependencies are recorded.
No local imagery, OSM/WorldCover label values or downstream references enter this
preparation. The normalized-tensor regression contains no label targets or local
imagery fields and verifies the same physical RGB values as the NPZ path.

Encode with the identical SAT-493M weights/crop configuration used for local RGB,
fit one regional unlabelled projection and use the same downstream protocols.
This public RGB control has different sensor response and temporal compositing
from the local images; the fixed-local-image degradation control isolates spatial
detail more directly. Report both, including unfavorable results. Neither control
preparation nor its software tests proves a downstream win.

The two magnitude-preserving fusion candidates both failed the common acceptance
matrix. Do not extend normalization search without a new supported hypothesis.
Preserve all 160 paired task-reader results and the previously better observations.
The three-region goal is still active; there is no approved three-region paper
victory claim and no Overleaf body change.
