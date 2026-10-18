# Regional high-resolution tuning progress

```json
{
  "stage": "validation",
  "completed": {
    "pyramid_24_16_16_8_k5": {
      "validation_macro_f1": 0.6001487610430714,
      "minimum_margin": -0.003267441133260718,
      "heads": {
        "mlp": 0.6087873214236689,
        "conv3x3": 0.5915102006624738
      }
    },
    "pyramid_16_16_16_16_k5": {
      "validation_macro_f1": 0.5968569885684767,
      "minimum_margin": -0.008035126407201165,
      "heads": {
        "mlp": 0.5983090996598546,
        "conv3x3": 0.5954048774770988
      }
    },
    "pyramid_24_16_16_8_k9": {
      "validation_macro_f1": 0.6045231328230718,
      "minimum_margin": 0.0007571932599875764,
      "heads": {
        "mlp": 0.6135114305904215,
        "conv3x3": 0.5955348350557221
      }
    }
  }
}
```
