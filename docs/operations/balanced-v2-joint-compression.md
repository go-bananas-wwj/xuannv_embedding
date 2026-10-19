# Balanced v2 joint compression and source diagnosis

The approved balanced cohort is fixed. Original support pools, budgets, targets,
valid masks, seeds and all original v1 results remain retained. Neither changing
calibration coverage nor repeatedly inspecting regional scores creates a blind
geographic test.

The magnitude-preserving experiment finished all eighteen fresh readers and 120
conditions. It fails acceptance and triggers no full-region queries. Auckland
improves over the unit-normalized common skip, but remains below AEF. Haidian's
native imagery also loses to its degraded control for both readers.

| Region | Reader | Native local | Matched public | Degraded local | AEF |
| --- | --- | ---: | ---: | ---: | ---: |
| Haidian | MLP | 82.326 | 81.915 | 82.399 | 83.875 |
| Haidian | CNN | 84.129 | 84.425 | 84.638 | 84.382 |
| Harbin | MLP | 56.943 | 55.666 | 55.941 | 58.396 |
| Harbin | CNN | 59.618 | 59.139 | 59.190 | 58.796 |
| Auckland | MLP | 59.572 | 56.099 | 55.152 | 63.434 |
| Auckland | CNN | 59.958 | 55.946 | 56.748 | 64.683 |

Values are exploratory calibration macro F1 percentages. References are freshly
fitted balanced-v2 references, subsequently reused with identity and sampling
checks. No v1 score is used to set a v2 acceptance threshold. Negative results
are retained; none is a claim of success for the overall goal.

## One registered joint method

The manual allocation of sixteen learned, sixteen visual and thirty-two physical
coordinates can discard useful local visual directions. Test one existing
region-independent alternative: standardize all 223 input coordinates together,
then fit a single 64-component PCA using seed 41 and at most 65,536 label-free
samples per region. Preserve score magnitudes. Use the original orderless
physical descriptor, not the unsuccessful calendar-statistics replacement.

Apply exactly this algorithm to all three regions and native, genuine public and
degraded roles. Their source arrays and encoders stay fixed. Each region fits
its own scaler and projection. No primary task label enters projection fitting;
all regional tiles may contribute unlabeled observations. No encoder is newly
trained. The recipe is registered before its new exports or task scores exist.

Export nine representations and fit eighteen fresh MLP/CNN readers, covering all
120 primary task-reader conditions. Existing verified v2 AEF/original/strongest
public references remain cached. The public envelope can only increase when the
new genuine public control is evaluated. Acceptance still requires strict wins
in all six regional-reader slots against AEF and the public envelope, with each
region's two-reader mean AEF gain at least 0.5 percentage points. Only an eligible
common candidate receives full-region queries with its frozen heads/thresholds.

## Separate source diagnostic

Auckland's largest deficits include Urban Parkland/Open Space (LCDB 2) and Exotic
Forest (LCDB 71). The native visual branch is present for approximately 99.9% of
class 71 calibration positives and 99.2% of support positives. Missing imagery
alone cannot explain the deficit. Presence is not evidence of semantic quality.
The LCDB target is the Class_2023 field, and the frozen satellite teacher's input
normalization matches its official model configuration.

For these two classes only, evaluate the existing full 64D visual descriptor
with native, public and degraded sources and the same MLP/CNN rules: six fresh
readers and twelve diagnostic conditions. Keep the original masks and negatives
from all other classes. Verify supports, observations and target identities
against the corresponding v2 AEF conditions. This tests source information before
manual branch compression; it does not change the main tasks or public envelope,
and cannot enter common-candidate ranking or demonstrate three-region success.

The existing package supports both studies without production code changes.
The preceding 996-test and package gates apply to the unchanged frozen source;
runtime receipts additionally verify actual source, input, cohort and output
identities. Use only free authorized devices 1–7, a bounded queue and disk reserve.
No automatic success flag completes the paper goal or modifies the manuscript.

Class names follow the [official LCDB v6 classification attachment](https://lris.scinfo.org.nz/layer/123148-lcdb-v60-land-cover-database-version-60-mainland-new-zealand/attachments/22474/).
