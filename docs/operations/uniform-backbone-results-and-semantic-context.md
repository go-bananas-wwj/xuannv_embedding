# Uniform backbone results and the next semantic-context hypothesis

The observation-only common backbone completed all 120 task-reader conditions:
three representations, three regions, both heads and all twenty primary tasks.
Every condition's support and calibration contract matches both fixed original
baselines. All nine exported representations have verified hashes, 64 channels
and one implementation; public images, annual observations and terrain are
identical between each region's roles. Public controls contain no contributing
local source. No historical regional encoder weights enter this experiment.

Calibration macro-F1 in percent:

| Region | Reader | Native high resolution | Public, full 64D budget | Same local images at 10 m | AEF |
| --- | --- | ---: | ---: | ---: | ---: |
| Haidian | MLP | 66.518 | 69.062 | 64.904 | 75.139 |
| Haidian | CNN | 75.377 | 76.128 | 74.425 | 76.892 |
| Harbin | MLP | 54.848 | 52.205 | 51.963 | 56.120 |
| Harbin | CNN | 58.524 | 55.452 | 55.045 | 58.168 |
| West Auckland | MLP | 56.569 | 50.946 | 49.914 | 58.045 |
| West Auckland | CNN | 58.908 | 53.868 | 54.087 | 59.314 |

Native local detail helps relative to degraded versions in all six slots, yet
the common-method acceptance fails. Haidian also trails its public counterpart
in both readers. Do not promote Harbin's one favorable AEF comparison or combine
different readers across regions. No new full-region queries were run.

Haidian's native MLP forest F1 is 39.280%, while its CNN forest F1 is 67.742%.
This motivates testing whether explicit regional semantic information complements
the frozen visual and physical observation features. It does not prove a cause
or justify changing the downstream readers or support samples.

Register one new semantic-context experiment before observing its scores:

1. Train one public-only auxiliary model independently per region. Its inputs
   are the uniform public descriptor (32 public-RGB, 28 observation and 4 terrain
   dimensions) plus the corrected 95-channel annual observations. There are no
   historical neural regional weights, local images or primary reference labels
   in this training.
2. Use the same 128-wide residual model, genuine ESA WorldCover2021v200 targets
   (zero ignored, eleven foreground classes), loss rule and optimization in all
   regions: 1,200 updates, batch16, learning rate0.001, weight decay0.01, seed41.
   Observation reconstruction weight is0.2 and public preservation weight0.02.
   The optional local reconstruction term is zero because no local input exists.
   Region-dependent observation statistics and class frequencies are recorded.
3. Export the eleven predicted foreground probabilities from each fixed model's
   auxiliary head. Give the **same public-only context array** to native, public
   and degraded-local roles within that region. Do not replace the original
   observation descriptor with the weakly supervised model's complete embedding.
4. Fit one fixed 64D combination. Native/degraded use public-RGB16, local-RGB16,
   annual-observation17, terrain4 and predicted-context11. Public controls use
   public-RGB32, annual-observation17, terrain4 and predicted-context11. Normalize
   visual and observation branches as before; retain terrain and probability
   magnitudes after projection, with no final unit normalization.

This retains observation branches but changes their allocation; it may lose
useful spectral information and is not a demonstrated improvement. WorldCover
supervision is allowed and must be disclosed. CLCD, DW and LCDB remain downstream
references only. This is regional neural semantic learning followed by frozen
fusion, not a rerun of the historical monthly-attention architecture.

Three independent neural fits run on devices1–3, leaving device0 untouched.
Bounded local monitoring supports the existing final/checkpoint recovery and a
100GiB reserve. Reader fits wait for those models and their verified context
exports, then use devices1–7. All120 primary task-reader conditions and the
strongest existing public references remain required. No new query is authorized
by a software check alone; apply the existing common-candidate acceptance gate.
