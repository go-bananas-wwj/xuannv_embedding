# Regional high-resolution tuning progress

```json
{
  "stage": "complete",
  "winner": "context_s16_k5",
  "acceptance_met": true,
  "validation": {
    "context_s16_k3": {
      "validation_macro_f1": 0.597226565123259,
      "minimum_margin": 0.02494368657325652,
      "heads": {
        "mlp": 0.6010701770151301,
        "conv3x3": 0.5933829532313879
      }
    },
    "context_s16_k5": {
      "validation_macro_f1": 0.6005609339313951,
      "minimum_margin": 0.026338375137603176,
      "heads": {
        "mlp": 0.6063442260670557,
        "conv3x3": 0.5947776417957346
      }
    },
    "context_s24_k3": {
      "validation_macro_f1": 0.5936900289031641,
      "minimum_margin": 0.018617701313315305,
      "heads": {
        "mlp": 0.6003230898348816,
        "conv3x3": 0.5870569679714467
      }
    },
    "context_s24_k5": {
      "validation_macro_f1": 0.5981449589203979,
      "minimum_margin": 0.022519271812074915,
      "heads": {
        "mlp": 0.6053313793705895,
        "conv3x3": 0.5909585384702063
      }
    },
    "context_s32_k3": {
      "validation_macro_f1": 0.5940269961122112,
      "minimum_margin": 0.012750433938627737,
      "heads": {
        "mlp": 0.6068642916276633,
        "conv3x3": 0.5811897005967591
      }
    },
    "context_s32_k5": {
      "validation_macro_f1": 0.6015785517607553,
      "minimum_margin": 0.024655189585066628,
      "heads": {
        "mlp": 0.6100626472783124,
        "conv3x3": 0.593094456243198
      }
    }
  },
  "overleaf_modified": false
}
```
