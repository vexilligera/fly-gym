# Claw cell-body validation

## Analysis fixed before fitting

Source: Mamiya et al. (2023), physiology dataset
<https://doi.org/10.5061/dryad.dbrv15f6q>, article
<https://doi.org/10.1016/j.neuron.2023.07.009>. Use the six summary files for
Figure 5 / S6. They contain GCaMP7f/tdTomato ratio changes (delta R/R), source
normalized signals, cell positions, tibia angles, fly IDs and timestamps.
These are processed data from female right front legs. Some segmented regions
contain multiple overlapping cell bodies; do not count them as guaranteed
individual neurons. The source preprocessing is already applied.

- Keep the source arrays exactly, including negative times and signals. Convert
  the restricted, numeric-only pickle files to NPZ; never execute the notebook.
  The source-normalized traces are preserved but **not used for fitting or
  evaluation**. No new normalization, test-trace gain or baseline fit is allowed.
- Primary cohorts: JR209 (50 regions, 7 flies) and JR688 (28 regions, 6 flies).
  The all-claw 73D10 cohort (53 regions, 7 flies) is reference only. Namespace
  fly IDs by driver. Equal column indices across movement orders are not treated
  as proof of exact cell correspondence.
- JR209: train flies 1–4, select on fly 5, independent animal test flies 6–7.
  JR688: train flies 1–3, select on fly 4, independent animal test flies 5–6.
  Training and selection use trial type 2 (start extended). Test both orders on
  unseen flies. Type 3 (start flexed) on training/selection flies is a separate
  protocol transfer test, not independent-animal evidence.
- Fit one population observation curve per driver: offset + nonnegative gain
  times a sigmoid of angle. JR209 decreases with angle; JR688 increases with
  angle. This uses driver labels, never response-derived labels on test regions.
  Retain any mislabeled/mixed responses, including the occasional flexion cells
  labeled by JR688 described in the paper.
- Candidate causal relaxation times are 0, 0.3, 1 and 3 seconds, selected only on
  the validation fly in that cohort. They describe effective calcium dynamics,
  not measured GCaMP kinetics, membrane constants or spike rates. Initialize at
  the first angle's steady response. Fit offset, gain, half-response angle and
  width by bounded least squares with three fixed starts (45, 90, 135 degrees).
  Bounds: offset [-2,5], gain [0,20], midpoint [0,180], width [1,90]. Weight flies
  equally, regions equally within each fly, and time samples equally per region.
- Baselines: a cohort-specific training constant and the old extension-only
  input curve, with offset and nonnegative fluorescence gain learned from the
  same training data. Also report the other cohort's frozen curve as a wrong-
  driver control. It is a diagnostic, not a calibrated amplitude comparison.
- Freeze every model and selection decision before scoring tests. Report raw
  delta R/R MSE, MSE relative to the training constant, correlation, and failures
  per region and per fly. Aggregate equally by fly; do not infer independent
  sample sizes from regions or frames. No significance or population CI claim
  is made from two test flies per driver.
- Descriptive low-angle (<=60 degrees) versus high-angle (>=120 degrees)
  contrasts quantify tuning separately for each region and order. This is a
  check of the published phenomenon, not a label used by the prediction model.
- Audit all available annotation fields for our four MaleCNS IDs and preserve
  MANC cross-references. A cross-reference alone does not link a recording to an
  EM neuron. No flexion/extension assignment is made without such evidence.
- No change to the neural encoder, reflex mechanics, or maze follows
  automatically. This tests sensory calcium predictions, not whole-reflex
  calibration or natural walking.

## Reproduction

```sh
.venv/bin/python scripts/prepare_claw_cell_recordings.py
.venv/bin/python scripts/validate_claw_cells.py
.venv/bin/python scripts/check_claw_cell_validation.py
```

The source ZIP is expected under `outputs/leg-calibration-source/mamiya2023/`.
The imported NPZ and JSON manifest are checked into `brain/calibration_data/`;
fitting uses those portable files and does not require the source ZIP. Results
are written to `outputs/claw-cell-validation/` for publication under
`wasm/calibration/` after validation.

## Validation results

All 262 region traces were imported: 156 from the two primary driver cohorts,
106 from the unused mixed all-claw cohort. Source data are preserved exactly.
The source ZIP SHA256 is
`866e8487b51b8e2cbc1c9ec7d875a33f918249d14c76c115e37601b05cf59862`.

| Test group | Flies | Region traces | MSE / training-constant MSE | Flies beating constant |
| --- | ---: | ---: | ---: | ---: |
| JR209 unseen flies, both orders | 2 | 30 | 0.896956 | 2/2 |
| JR688 unseen flies, both orders | 2 | 16 | 0.840612 | 1/2 |
| JR209 reverse order, previously seen flies | 5 | 35 | 0.234553 | 5/5 |
| JR688 reverse order, previously seen flies | 4 | 20 | 0.728058 | 4/4 |

These are ratios of mean per-fly MSEs, not pooled-frame statistics. JR688 fly 6
fails (1.87145 times baseline error); JR209 fly 7 improves only slightly
(0.988554). Across both orders, 20/30 JR209 and 9/16 JR688 held-out region traces
beat the baseline. Failures are retained in the report and viewer.

The direct low-angle/high-angle contrast, independent of the constrained fit,
is flexion-preferring in 30/30 held-out JR209 traces and extension-preferring in
12/16 held-out JR688 traces. The remaining JR688 responses are not discarded or
reassigned for prediction. These trace counts include both movement orders and
must not be interpreted as that many independent cells or flies.

JR209 selects effective relaxation 3 s, midpoint 71.11 degrees and width 17.54
degrees; JR688 selects 1 s, midpoint 136.50 degrees and width 10.08 degrees. The
3 s value reaches the largest candidate considered. The parameters describe
population calcium responses and may absorb adaptation, hysteresis and
measurement effects. They are not validated single-neuron spike thresholds or
GCaMP kinetics. We do not extend the search after looking at test performance.
High correlations alongside weak raw-amplitude prediction show why waveform
agreement alone is insufficient for physiological calibration.

All four MaleCNS records were audited across every released annotation field.
Types SNpp50 and SNpp51 have FeCO-claw synonyms on other neurons in the release
(12 and 1, respectively), but the exact four have no tuning or peripheral soma
position. Cross-references are 815843→MANC 13310, 817680→21776, 912317→42121;
935383 has no MANC body ID. All four have VFB identifiers. None is a verified
match to a recorded cell region. This audit does not claim an exhaustive
morphology-registration search, which remains future work.

The raw-array round trip, equal-fly weights, complete fly separation, held-out
poisoning, causal filtering, recomputed scores, and unchanged reflex source
checks pass. The deployed neural encoder and maze are not modified by this
experiment. The next calibration step needs per-cell tuning/variability and an
anatomical match; this population fit is not accepted as physiological circuit
calibration.
