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
    }
  }
}
```
