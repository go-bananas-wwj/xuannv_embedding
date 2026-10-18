# Separate local/context features: validation-only outcome

All eight fixed 64D candidates completed validation. The validated parent `wide_k9` was retained. No new query fitting, regression scoring or matched-control training was performed for these candidates. The current 59.1703% MLP / 57.2169% CNN result remains the latest validation-selected regional candidate; the new refinement goal remains active.

The same original archived validation anchor was used throughout. The best parent margin is 0.3689 percentage points; no new candidate exceeded it. Query scores never selected a candidate.

| Candidate | MLP validation | CNN validation | Minimum margin |
|---|---:|---:|---:|
| pyramid_24_16_16_8_k5 | 60.8787 | 59.1510 | -0.3267 |
| pyramid_16_16_16_16_k5 | 59.8309 | 59.5405 | -0.8035 |
| pyramid_24_16_16_8_k9 | 61.3511 | 59.5535 | +0.0757 |
| pyramid_16_16_16_16_k9 | 60.2142 | 58.8426 | -0.6351 |
| pyramid_16_24_8_16_k5 | 59.7263 | 59.1096 | -0.9081 |
| pyramid_16_24_8_16_k9 | 60.0359 | 58.7695 | -0.7082 |
| pyramid_32_16_8_8_k5 | 59.6903 | 58.1364 | -1.3414 |
| pyramid_32_16_8_8_k9 | 60.1630 | 58.2987 | -1.1791 |

This does not establish that the multiscale features harm height regression; height query performance was not measured. All projections, validation heads and negative outcomes are retained. No Overleaf edits.
