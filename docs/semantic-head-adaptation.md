# Joint semantic-head adaptation

`xuannv experiment run --freeze-base --highres-encoding transformer
--train-semantic-head` enables only the existing semantic probe parameters in addition
to the usual new high-resolution modules. It requires a registered public parent and
an active semantic objective. Public encoders, STP, projection and original decoders
remain frozen. The same optimizer learning rate and scheduler apply to the probe and
new modules; no new loss is added.

The opt-in and exact semantic parameter names are recorded in adaptation provenance.
Changing this flag on resume is rejected. Without the flag, historical provenance is
unchanged. Exports reconstruct the same opt-in from the registration. Followup audits
verify the entire public model, all non-probe criterion tensors, finite state and
actual optimizer updates; a registered trainable probe must have changed.

`xuannv experiment diagnose-head-only --config CONFIG --cache CACHE
--checkpoint CHECKPOINT --output RESULT --device cpu` provides a disposable structural
control on two registered training samples. It freezes the public model, updates only
the semantic probe once and checks exact embedding, model and frozen-state equality.
The checkpoint remains unchanged and no weights are saved. This establishes that
readout-only learning cannot improve a frozen embedding; it is not evidence for the
benefit of high-resolution inputs or generalization.
