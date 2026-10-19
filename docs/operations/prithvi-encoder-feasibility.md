# From failed feature allocation to a temporal pretrained backbone feasibility gate

The common public-context/fine-detail study completed nine exports, its exact
shared-coordinate audit and all120 fresh primary conditions. Within every region,
the three roles have identical56 common coordinates; only eight fine coordinates
differ. This is a verified source control, not an accuracy success.

| Region | Reader | Native local | Matched public | Degraded local | AEF |
| --- | --- | ---: | ---: | ---: | ---: |
| Haidian | MLP | 82.933 | 82.750 | 82.849 | 83.875 |
| Haidian | CNN | 84.902 | 84.849 | 84.602 | 84.382 |
| Harbin | MLP | 59.248 | 57.648 | 57.594 | 58.396 |
| Harbin | CNN | 60.489 | 60.261 | 60.905 | 58.796 |
| Auckland | MLP | 56.295 | 56.095 | 55.709 | 63.434 |
| Auckland | CNN | 55.164 | 55.552 | 55.061 | 64.683 |

Values are balanced-v2 exploratory calibration macro F1 percentages. Harbin
exceeds AEF in both readers, but its CNN still trails the registered strongest
public reference. Auckland worsens substantially. The study fails acceptance;
no full-region queries were launched. The public envelope increases where
applicable and never falls. All task-level failures remain retained.

High retained variance did not ensure task quality. Do not continue arbitrary
PCA allocation grids or combine different regional winners. The next hypothesis
is that a globally pretrained multispectral temporal encoder can improve the
shared public representation before adding local detail. It requires new evidence.

## Primary-source compatibility research

[OlmoEarth's official minimal implementation](https://github.com/allenai/olmoearth_pretrain_minimal)
provides multimodal pretrained encoders. At the inspected source commit
`c3a7e852d4f9179372eafc0a2c709a6b1284f8ad`, its package requires PyTorch>=2.7
and an additional reshape dependency forbidden by this repository. Its full band
contracts also differ from our cached public observations. It is not a drop-in
for the existing PyTorch2.6 NPU environment; that environment was not upgraded.

[Prithvi EO2 300M's official model card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
describes masked-autoencoder pretraining on HLS with six spectral channels.
Pin revision`9eb1b1102806593963daa333bcc491b1c6f8562f` before compatibility work.
The existing S2 observations contain all six physical channels: blue B02, green
B03, red B04, narrow-NIR B8A, SWIR1 B11 and SWIR2 B12. The publisher config's HLS
B05/B06/B07 names must not be mistaken for S2's identically numbered red-edge
bands. Restore each source's declared physical scale/offset, then apply the
published normalization in scaled-reflectance units. Existing S2 imagery is not
claimed to be harmonized HLS; its working resolution also differs from HLS30m.

The [publisher's encoder source](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/blob/9eb1b1102806593963daa333bcc491b1c6f8562f/prithvi_mae.py)
is Apache2 licensed. The new native PyTorch encoder retains its names, positions,
normalization and attention mathematics, with copyright/license attribution.
It excludes the decoder, rejects silent cropping and transfers recomputed dynamic
positions to the encoder device. No new dependency is added. Every encoder key
must be consumed strictly; excluded decoder keys are counted, not silently used
as permission to ignore encoder mismatches. The fixed model is the non-TL300M
variant, not a reader-selected per-region model.

## Feasibility evidence and remaining gates

Failing-first tests cover dynamic/square/non-square shapes, invalid dimensions,
missing/extra checkpoint keys, bad shapes/weights and positional layout. All20
focused tests and1,028 full tests pass. A tiny implementation comparison loads
all30 publisher encoder keys and returns bit-identical CPU tokens in three shape
conditions. The reference only omits an unused optional decoder/image-reshape
import; its encoder forward and position functions are unchanged.

An observation-selected real S2 sample from every region has a recorded input
hash, calendar, named spectral mapping, radiometry and valid-pixel counts. Missing
inputs are normalized to zero (the publisher mean), and observation masks remain
separate. The frozen architecture has no native cloud-mask attention. Subsequent
features must disclose that limitation and weight temporal pooling by observed
patch fractions; missing values cannot be declared observations.

Three actual NPU gates with these inputs and a small synthetic encoder pass.
FP32 maximum errors are below0.00074; mixed-precision relative errors are
0.00345–0.00467 and mean token cosines exceed0.99998. This proves operator/layout
compatibility for the small architecture only. It does not validate the complete
pretrained checkpoint, its task quality or a high-resolution fusion model.

The fixed official checkpoint is downloading outside Git through the HF CLI.
After its checksum is verified, complete strict full-encoder loading and real
pretrained CPU/NPU comparisons before exporting regional features. Only then
register a bounded three-region public-feature comparison and corresponding
local-fusion follow-up. Reuse the existing observations, primary masks and
approved calibration cohort. Do not change targets, budgets or acceptance gates.
A public pretrained-model diagnostic alone cannot satisfy the high-resolution
method goal. The original best regional bundles and manuscript body remain intact.

## Complete pretrained checkpoint verification and next bounded study

The download completed and matches the official LFS SHA256
`faab2c8b776f0723f93431747603a7ec586a54fc45473613b4322c29752486e7`.
All294 encoder state entries load strictly;104 decoder entries are explicitly
excluded. Complete pretrained CPU comparisons on the three actual regional
samples, at four and twelve frames, are bit-identical to the publisher reference.
All six complete-model NPU cases pass the preregistered numerical thresholds.
FP32 relative errors are0.04–0.11%, and bfloat16 errors0.53–1.52%. No primary task
score was calculated by these checks. Frozen verified source is`ea103cf`.

Register the next whole-region study before regional projection fitting or task
scores: all320/422/320 existing tiles, their twelve calendar months and original
geography. Extract observed-patch-weighted temporal descriptors, preserving
missing masks; fit a single64-component regional PCA only after merging every
shard. Fit from at least0.9 observed-frame equivalents per native patch, capped
at65,536 samples with seed41. Align over the unchanged footprint to128-by-128;
mask annual coverage below0.05 and preserve PCA-score magnitudes. Native global
patch tokens remain160m at the existing10m input grid; interpolation is not newly
observed detail. No new image acquisition or reference-map input is introduced.

Score Prithvi64 alone as a genuine public control: six readers/forty conditions.
That alone cannot achieve the local-fusion goal. Then test two declared common
64D candidates, with one unchanging recipe across all regions and data roles:

- Regional public NN16, Prithvi16, fine visual16 and original physical16.
- Prithvi32, fine visual16 and original physical16.

The first preserves the existing regional WorldCover-trained backbone; the
second uses regional projection fitting without a locally trained neural
backbone. Do not conflate their training or losses. Physical observations retain
public S1 and Landsat even though Prithvi itself consumes S2 only. Existing fine
SAT-493M features supply native local, public S2 and degraded-local roles with
the same capacity. Within a candidate,48 public/physical coordinates must match
between roles; only the sixteen fine coordinates change. All branch/output
unit-normalization flags are false. These are fixed budget hypotheses, not a
per-region or primary-label dimension search.

Both candidates receive all240 matched conditions: thirty-six fresh MLP/CNN
reader jobs plus the six public-control jobs, totalling280 new conditions. Reuse
verified AEF/strong-public caches from v2, raise the envelope when appropriate,
and select only one globally eligible recipe by worst-slot gain then mean. Keep
the original twenty primary tasks and all negative results. Apply the unchanged
full-query/frozen-head gates and separately verify high-resolution contribution.
Full extraction/reader orchestration follows implementation and data-pipeline
tests; registration is not a claim those exports have already run. The goal and
manuscript remain unchanged until the complete scientific acceptance succeeds.
