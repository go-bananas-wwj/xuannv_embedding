# Optional controls for monthly high-resolution fusion

`model.highres_transformer` accepts three independent settings:

| Setting | Default | Behavior |
| --- | --- | --- |
| `allow_base_only` | `false` | Add a zero-increment option to source selection. |
| `coverage_gating` | `false` | Add native input-mask coverage to each source logit. |
| `spatial_readout` | `attention` | Use `mean` for a matched window-pooling control. |

These options do not establish a performance improvement. Evaluate them with the
same parent, samples, training budget and frozen readout protocol.

The base-only option has a query-dependent logit initialized to zero. Its softmax
weight competes with present high-resolution sources. Source weights multiply the
entire affine output, including bias; absent sources and the base-only option
contribute zero. All-missing windows therefore preserve the public feature even
after output biases have been learned. With one available source and zero logits,
the initial source weight is one half, rather than one.

Coverage is the valid native-pixel fraction over patches assigned to each window.
Partial boundary patches use their actual pixel area. Padding and empty packing
slots do not reduce the coverage of a fully valid image. Coverage comes only from
input masks and is not a calibrated quality probability. The additional linear
coverage coefficient starts at zero.

The mean control averages normalized valid tokens within the same window and uses
the shared learned output projection. It retains source gates and the native
encoders, but removes the cross-attention parameters. It is not parameter-matched
to attention. Input values at invalid positions cannot enter its mean.

Optional module initialization preserves the CPU random stream so subsequent
shared modules and paired training masks are not shifted. Shared parameters retain
their historical initialization. With both switches disabled and attention chosen,
the state-dictionary schema and historical operation order remain unchanged.

Checkpoint replay requires the same settings and strict state-dictionary loading.
Do not load a trained attention adapter into the mean control and describe it as
a retrained ablation. Multi-device resume still requires the original world size.
