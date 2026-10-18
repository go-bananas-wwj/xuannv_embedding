# Regional high-resolution tuning progress

```json
{
  "stage": "validation",
  "completed": {
    "public_rotation64": {
      "validation_macro_f1": 0.5523974517528092,
      "minimum_margin": -0.025199912389700208,
      "heads": {
        "mlp": 0.5615555492371871,
        "conv3x3": 0.5432393542684312
      }
    },
    "native_blocks_8": {
      "validation_macro_f1": 0.5782115004805961,
      "minimum_margin": 0.001690490467491923,
      "heads": {
        "mlp": 0.5862932438355689,
        "conv3x3": 0.5701297571256233
      }
    },
    "native_blocks_16": {
      "validation_macro_f1": 0.5863650855325184,
      "minimum_margin": 0.014018425263911172,
      "heads": {
        "mlp": 0.5902724791429944,
        "conv3x3": 0.5824576919220426
      }
    },
    "native_blocks_24": {
      "validation_macro_f1": 0.5870428502864444,
      "minimum_margin": 0.012604854520014697,
      "heads": {
        "mlp": 0.5930415793947428,
        "conv3x3": 0.5810441211781461
      }
    },
    "native_blocks_32": {
      "validation_macro_f1": 0.5884944068113411,
      "minimum_margin": 0.008983749464338997,
      "heads": {
        "mlp": 0.5995657975002119,
        "conv3x3": 0.5774230161224704
      }
    },
    "native_blocks_48": {
      "validation_macro_f1": 0.585046633260808,
      "minimum_margin": 0.01098215515164458,
      "heads": {
        "mlp": 0.59067184471184,
        "conv3x3": 0.579421421809776
      }
    }
  }
}
```
