# One common observation-only backbone across all three regions

An audit found that the inherited public regional encoders used different
historical reconstruction weights, semantic losses, masking settings and map
target encodings. Their shared inference architecture does not establish the
same end-to-end training rule. Keep their scores as historical references and
frozen-feature diagnostics; they cannot alone prove the common-method objective.

Register one candidate with no dependency on these trained regional weights:

| Input branch | High-resolution model | Public comparison |
| --- | ---: | ---: |
| Public Sentinel-2 RGB through the same frozen satellite visual encoder | 16 | 32 |
| Local RGB through that same frozen visual encoder | 16 | Absent |
| Corrected annual public observation statistics | 28 | 28 |
| Public Copernicus surface-height context | 4 | 4 |
| Total output channels | 64 | 64 |

The public comparison receives all 32 visual dimensions from its available
public image, rather than sixteen useful dimensions and sixteen empty channels.
It has the same global pretrained encoder and the same total output width. Both
receive identical annual observations and terrain. A third comparison uses the
exact high-resolution model with only its local images degraded to ten metres;
images, dates and fine validity masks remain fixed. Use all three regions for
every role. Do not choose different source allocations by region.

All visual exports use identical pretrained weights, crop/window rules, 64-channel
regional projection rules, sampling budgets and geographic alignment. Each
region fits its own observation-only projections from all registered tiles using
one seed and rule. Standardize public spectral statistics and terrain before
their PCA. Unit-normalize the visual and spectral projected branches, preserve
projected terrain magnitude, and omit final unit normalization. Public image
sensor/date/compositing differences remain documented.

This is a frozen visual encoder plus unsupervised regional fitting experiment.
There is no new neural loss or regional encoder fine-tuning in this candidate;
do not describe it as a uniform rerun of the historical P10C neural training or
attribute the frozen encoder's global pretraining cost to a small regional fit.
WorldCover and other maps are not targets for these regional projections. Public
terrain is a direct input, whereas AEF used terrain as a training target; this is
a data-enriched product comparison, not an identical-data architecture claim.

Use the fixed CLCD, DW and LCDB contracts, all twenty tasks, both readers, original
support samples and calibration pixels. Rank one shared candidate against AEF
and the strongest public reference in each of the six regional/reader slots.
Keep all failures. Only an eligible common candidate proceeds to full-region
queries, and those remain exploratory map agreement rather than blind tests.
Verify capacity and degraded-resolution contrasts before describing their causes.

CPU preparation may run while visual extraction is active. Reader fits must wait
for the existing extraction and reader queues to complete successfully. Preserve
the original best bundles, stop at the disk reserve or STOP signal, bound retries
and waiting, and leave device zero untouched. No manuscript changes follow merely
from a software check or a prepared embedding.
