# Calendar encoder with preserved visual and observation features

This is a single conditional follow-up to the calendar encoder experiment,
registered before its feature exports or reader scores. All three regions use
the same recipe and separately fitted regional weights. It is not an accepted
paper result, and the three-region AEF objective is still unmet.

## Evidence motivating the follow-up

The completed native-local and public calendar encoders have the following
original-cohort calibration macro F1 percentages. These are exploratory map
agreement scores, not the full-region diagnostic or an unseen-region test.

| Region | Reader | Native local | Public calendar | AEF |
| --- | --- | ---: | ---: | ---: |
| Haidian | MLP | 71.564 | 65.748 | 75.139 |
| Haidian | CNN | 64.392 | 62.337 | 76.892 |
| Harbin | MLP | 54.198 | 53.366 | 56.120 |
| Harbin | CNN | 57.323 | 58.129 | 58.168 |
| Auckland | MLP | 53.029 | 52.722 | 58.045 |
| Auckland | CNN | 53.788 | 53.675 | 59.314 |

All six native-local slots fail to beat AEF. Completing the remaining degraded
controls is necessary for the complete comparison; none can make this already
measured native-local representation satisfy the fixed goal.

Training reconstruction and WorldCover losses decreased. However, final local
visual reconstruction normalized MSE remains approximately 0.91–0.95. A separate
label-free diagnostic using the exported mean descriptor, saved visual decoder,
seed 41, twenty tiles and at most 256 valid samples per tile explained only about
6–10 percent of local teacher variance. This is evidence of weak visual
reconstruction, not proof that it causes the downstream deficit. Reducing the
training objective alone has not established useful transferable features.

A further label-free audit fitted a new linear decoder on forty tiles per region
and evaluated it on twenty different tiles, with at most 256 valid pixels per
tile, seed 41 and ridge penalty 1. Targets were standardized using fitting samples
only. Neither the encoder nor any primary reader was changed. Held-tile R-squared
against the fitting-mean baseline was:

| Region | Public visual teacher | Local visual teacher |
| --- | ---: | ---: |
| Haidian | 0.0966 | 0.1213 |
| Harbin | 0.0894 | 0.1136 |
| Auckland | 0.0955 | 0.1862 |

Thus, fitting another linear decoder alone does not recover most teacher
variation in this diagnostic. Nonlinear information may remain, and neither this
audit nor reconstruction variance measures downstream usefulness. These results
motivate testing the registered skip; they do not establish its effectiveness.

The existing eighty native/public task-reader fits were also checked for input
scaling. Every reader already standardizes using labeled valid support pixels
only, uses one hundred updates, and clears invalid context after standardization.
All scaler channels are finite and positive; all eighty saved reader identities
and parameter payload checksums match their records. A missing normalization
stage therefore does not explain these scores. In Haidian the main calibration
deficit is forest: native MLP/CNN F1 is 49.55/22.83 percent, against AEF
66.67/71.40 percent. Native forest AP is also lower, so the deficit is not solely
a threshold difference. The other three classes remain reported. The sparse
forest calibration cohort is unchanged; no alternative cohort has been approved.

## One fixed follow-up recipe

Use the existing label-free block exporter, without another encoder training:

| Branch | Output dimensions | Source |
| --- | ---: | --- |
| Calendar representation | 16 | Mean output of the role-matched regional encoder |
| Frozen visual skip | 16 | Frozen SAT493M optical features |
| Physical observation skip | 32 | Same public annual 95-channel summaries |

Fit each projection from at most 65,536 observation samples with seed 41, normalize
each branch and then normalize the concatenated 64D descriptor. All regional
projection fits use this same rule and no primary downstream labels. The skip
preserves an explicit route around the learned bottleneck, but projection can
still lose useful information; improvement must be measured.

Apply the same dimensions and normalization to three roles in every region:

- Native local: native local encoder and native local frozen visual skip.
- Degraded local: its independently fitted encoder and the same local imagery
  degraded to 10 m before frozen feature extraction.
- Public: genuine public-only encoder and full twelve-month public S2 RGB frozen
  features, with no local imagery in either branch.

The existing block engine calls the first branch `public`. Here that is only a
legacy input key: native and degraded encoders contain their corresponding local
input and must never be described as public-only features. Terrain is already an
encoder input in all roles. There is no additional direct terrain branch. Local
SAR is absent from this experiment; public S1 remains included.

## Ordering, controls and stopping rules

The controller must first observe successful completion of all 120 original
calendar calibration conditions and termination of the existing evaluator. It
must not compete for that evaluator's devices. If the original calendar candidate
passes, skip this follow-up and inspect that candidate's full acceptance instead.
Failure of the predecessor stops automatic progression for inspection.

Otherwise fit all eighteen readers: three regions, three roles and two heads.
Keep the original twenty tasks, support tiles, calibration cohort, masks, seeds,
twenty-tile support budget and one hundred reader updates. A proposed balanced
cohort remains unapproved and is not applied. Preserve every task score.

The reference public envelope includes the new matched public skip, plain public
calendar encoder, original public baseline and previously strongest genuine
public calibration results. Require strict gains over AEF and that envelope in
all six regional-reader slots, plus at least 0.5 percentage points mean AEF gain
within each region. Only an eligible common recipe proceeds to full-region
queries. Full-region scores do not select recipes. Passing a calibration gate
does not automatically complete the overall goal or authorize unsupported paper
claims.

Runtime controllers, exact registrations, weights, observations and full result
records stay outside Git. Existing original best bundles and all negative scores
remain preserved. No manuscript text or original introduction is replaced.
