# Regional high-resolution tuning progress

```json
{
  "stage": "training_and_validation",
  "utc": 1792285284.7575538,
  "supervisor_pid": 219835,
  "completed": {
    "frozen_public": {
      "stage": "complete",
      "utc": 1792277062.3609512,
      "validation_macro_f1": 0.5436560323793582,
      "selected_step": 1600,
      "completed_steps": 2400,
      "registered_budget": 2400
    },
    "frozen_anchored": {
      "stage": "complete",
      "utc": 1792277228.2880433,
      "validation_macro_f1": 0.537061551850658,
      "selected_step": 1600,
      "completed_steps": 2400,
      "registered_budget": 2400
    },
    "long_continuation": {
      "stage": "complete",
      "utc": 1792285253.6278892,
      "validation_macro_f1": 0.5481432840126117,
      "selected_step": 2400,
      "completed_steps": 2400,
      "registered_budget": 2400
    }
  },
  "restarts": {},
  "active": {},
  "progress": {
    "long_continuation": [
      {
        "step": 800,
        "validation_macro_f1": 0.5350040796971302
      },
      {
        "step": 1600,
        "validation_macro_f1": 0.5318494043371031
      },
      {
        "step": 2400,
        "validation_macro_f1": 0.5481432840126117
      }
    ],
    "frozen_public": [
      {
        "step": 800,
        "validation_macro_f1": 0.5229426322506436
      },
      {
        "step": 1600,
        "validation_macro_f1": 0.5436560323793582
      },
      {
        "step": 2400,
        "validation_macro_f1": 0.5294129806074674
      }
    ],
    "frozen_anchored": [
      {
        "step": 800,
        "validation_macro_f1": 0.5344566632321414
      },
      {
        "step": 1600,
        "validation_macro_f1": 0.537061551850658
      },
      {
        "step": 2400,
        "validation_macro_f1": 0.5313159946945665
      }
    ]
  }
}
```
