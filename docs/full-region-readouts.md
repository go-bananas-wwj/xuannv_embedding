# Full-region map agreement

`xuannv experiment full-region-readout --spec JOB.json --phase calibration|full_region`
keeps downstream fitting and calibration positions explicit, then scores every
registered regional tile, including support, calibration and buffer locations.
This is an application-region mapping comparison, not an independent held-out
generalization estimate. Keep the historical spatial test results separately.

The data contract records all tile IDs, feature and label hashes, common validity,
original training/calibration positions, task families, resolution and provenance.
Full-region coverage must never be implemented by relabeling a small test split.
The shared region may contain references seen by the encoder. Record each
encoder's image and label exposure separately; geographic exposure does not
establish that a different pretrained model saw the identical semantic targets.

Supported readouts reuse the existing checked implementations: linear
classification, coverage regression, prototype retrieval, MLP, shallow CNN,
random forest and compact U-Net. A matching archived head can be reused only
after exact support and calibration observation fingerprints match. Parameters
and calibration thresholds are frozen before full-region prediction.

Classifier support tiles come from the original support region. A full-region
score includes these fitted examples and the calibration examples, with the
overlap declared in the result identity. Retrieval includes the exemplar region
unless a separately registered exclusion policy is specified. Native land-cover
reference grids and physical boundary distances must remain explicit.

Shortlisting trained encoders is separate from choosing monthly or temporal-mean
readings. Do not call three readings of the same weights three training versions.
Retain all tested readings and unfavorable comparisons. A model selected using
full-region agreement is optimized for this region; it is not thereby established
as a universally better representation.
