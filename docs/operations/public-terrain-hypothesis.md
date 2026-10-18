# Public physiographic context: a separate, controlled data hypothesis

The current shared fusion improves on its matched no-imagery version in all six
region/reader slots but still fails Haidian, chiefly on forest. Normalization
ablations did not fix it. Public physiographic information is a new hypothesis;
it is not a demonstrated performance improvement.

AEF's original Table S2 and S15.8 include Copernicus GLO-30 elevation and terrain
quantities as training targets. They are not listed as raw inference inputs. Our
candidate data branch would expose public terrain directly, so any result must
be described as a data-enriched regional product comparison, not proof of an
architecture-only improvement with identical inference data. Give the exact same
terrain to the matched public/no-imagery model. Keep the original AEF/public
anchors, all classes, supports and reader budgets unchanged. [AEF paper](https://arxiv.org/html/2507.22291v1)

Use only public Copernicus GLO-30 source windows, independently located for all
three regions. GLO-30 is surface height, including buildings and vegetation; do
not present it as fine bare-earth ground truth. LINZ LiDAR DEM/DSM remains reserved
for evaluation and is excluded from this preparation.

The generic producer mosaics only a metric regional window plus a 600 m halo at
30 m. It stores elevation and central-difference slope/aspect as sine/cosine pairs,
with downhill aspect measured clockwise from north. Neighbors with nodata are
excluded. Interpolation onto the original 128x128 tile grids does not add observed
detail. Every source identifier, unsigned asset URL, crop, mask and feature array
is bound to content identity. No primary task labels enter it. Signed access URLs
are transient, not stored in provenance. A synthetic metric plane regression
checks derivatives and aspect; a real Haidian crop provides acquisition evidence,
not a downstream model result. Heights in that crop range approximately 30–1276 m.

No terrain-augmented embedding model has been scored. First materialize and verify
all three windows, then preregister one shared fusion allocation and a matched
public counterpart before observing new scores. A region-specific feature or
threshold chosen from target labels does not satisfy the common-method goal.
