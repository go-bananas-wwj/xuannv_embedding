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
    },
    "pyramid_16_16_16_16_k9": {
      "validation_macro_f1": 0.5952840576961277,
      "minimum_margin": -0.006351326892538589,
      "heads": {
        "mlp": 0.6021418004890594,
        "conv3x3": 0.588426314903196
      }
    },
    "pyramid_16_24_8_16_k5": {
      "validation_macro_f1": 0.5941794709197036,
      "minimum_margin": -0.009081429475674252,
      "heads": {
        "mlp": 0.5972627965913815,
        "conv3x3": 0.5910961452480257
      }
    },
    "pyramid_16_24_8_16_k9": {
      "validation_macro_f1": 0.5940273005481189,
      "minimum_margin": -0.007082484177930559,
      "heads": {
        "mlp": 0.6003594434784338,
        "conv3x3": 0.587695157617804
      }
    }
  }
}
```
