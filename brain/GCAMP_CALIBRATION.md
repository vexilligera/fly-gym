# GCaMP calibration pilot

This offline experiment compares the full released FlyWire spiking model with
published, individual-fly calcium recordings. It fits a few candidate neural
parameters and an explicit fluorescence observation model. It does not change
the live maze, establish innate navigation, or identify every synaptic strength.

## Experimental data and neuron matching

The source is Shiu, Sterne et al., *Taste quality and hunger interactions in a
feeding sensorimotor circuit*, eLife 2022,
[Figure 3 source data 1](https://cdn.elifesciences.org/articles/79887/elife-79887-fig3-data1-v3.xlsx),
[DOI: 10.7554/eLife.79887](https://doi.org/10.7554/eLife.79887), CC BY 4.0.
The original spreadsheet remains unmodified. `scripts/prepare_gcamp.py` verifies
its SHA-256, preserves its numerical observations and source-column addresses,
and writes `brain/calibration_data/shiu2022_taste.json`.

The pilot contains three GCaMP6s cell types imaged during proboscis presentation
of 1 M sucrose in food-deprived females. Water and bitter traces from the same
flies are retained but are not fitted by this sugar-only experiment.

| Published name | v783 annotation | Sugar training flies | Sugar test flies |
| --- | --- | ---: | ---: |
| Clavicle | AN_GNG_30 | 6 | 3 |
| G2N-1 | CB0616 | 4 | 2 |
| Zorro | CB0192 | 4 | 2 |

Named neuron/root-ID matches are cross-checked against Shiu et al., Nature 2024
Supplementary Table 1A and the pinned v783 annotations. The Clavicle right
homologue is additionally identified in the released `sez_neurons.pickle`.
These are cell-type correspondences across animals, not recordings of the
individual used for electron microscopy. Recording hemisphere is not specified
in the workbook; averaging the listed bilateral model homologues is an explicit
observation assumption.

The last third of chronological fly identifiers within each type, with at least
two flies, is held out. All tastants for a fly stay in the same split. Two G2N-1
sugar recordings have missing tail samples, which stay null and are masked.
The source contains nine Clavicle sugar traces; all are retained even though
the paper's general figure caption gives a smaller sample-size range.

## Timing and preprocessing

The one-photon workbook gives times at 1.2-second intervals, with stimulus flags
at frames 20 through 25. The pilot follows that explicit time axis and holds
each flag until the next sample: onset 24 s, offset 31.2 s. Methods describe a
nominal 0.8 Hz acquisition rate (1.25 s/frame), so precise inferred kinetics
remain uncertain. Two-photon sheets have a larger discrepancy between their
time columns and the stated acquisition rate; they are excluded here. Resolving
the original acquisition timing is necessary before treating this as a precise
physiological calibration.

Published values are delta-F/F. Each trace is recentered using only its own
pre-stimulus frames 9–18, including for held-out flies. The fitting/scoring
window is 18–43.2 s: six seconds before sugar onset through twelve seconds after
offset. Missing observations are excluded rather than treated as zero.

## Neural candidates versus fluorescence parameters

All 138,639 neurons and 15,091,983 connection rows run on the B300. The candidate
grid varies:

- Global effective synaptic gain: 0.75, 1, or 1.25 times the released 0.275 mV
  scale. Anatomical synapse counts, signs, and connectivity remain intact.
- Sugar sensory drive: 50, 100, or 200 Hz imposed on the released 21-cell
  unilateral sugar cohort.

That input cohort is a proxy for physical taste stimulation, not a verified
bilateral recruitment pattern or a calibrated 1 M concentration response. A fit
can compensate for this mismatch; it must not be interpreted as identifying
the true biological synaptic gain. The 200 Hz, gain-1 model is the baseline.

Each candidate is simulated with the same input random streams. The simulated
protocol has 1.2 s baseline, 7.2 s stimulus, and 12 s washout; earlier prehistory
is analytically silent under the released model's resting conditions. Mean
bilateral firing rates are recorded in 20 ms bins. No MuJoCo rendering or body
movement is needed for this neural experiment.

To compare spikes with fluorescence, a causal first-order filter predicts an
effective calcium signal:

`calcium_next = exp(-dt/tau) * calcium + (1-exp(-dt/tau)) * firing_rate`.

For each cell type, training data fit one nonnegative fluorescence gain and one
effective decay time constrained to 0.2–10 s. These are measurement parameters,
not membrane time constants or neural weights. This approximation omits GCaMP
saturation, a separate rise time, compartment-specific calcium, bleaching, and
motion artifacts. A fitted parameter at a bound is reported explicitly.

Both the baseline and each neural candidate receive independently fitted
observation parameters with identical flexibility. This prevents crediting a
neural change merely because its fluorescence scale was fitted and the baseline
was not. Nevertheless, gain/indicator/input degeneracies remain, especially
with only one stimulus concentration.

## Selection and checks

Training error is normalized by each cell type's training RMS response, floored
at 0.05 delta-F/F, and averaged with equal weight across cell types. The lowest
training error selects the candidate before test responses are scored. No
parameters are refitted on held-out flies.

The report includes per-fly test errors and a paired bootstrap that resamples
test flies within cell types. Its interval reflects the small biological sample,
not uncertainty in stimulus timing, neuron matching, or model structure. A
second simulation seed checks the baseline and selected candidate with the
previously fitted fluorescence parameters fixed.

These recordings were already part of the literature used to evaluate the
original model. Our split tests generalization of this particular fit, not
entirely new external validation. A convincing calibrated model also needs
additional stimulus strengths, independently recorded cells, matched inputs,
and intervention tests. This sugar-circuit result cannot validate visual
navigation, nerve-cord control, learning, or whole-brain dynamics.

## First measured result

The nine-candidate grid was run on B300 job 5807547 using implementation
`5faebd0`. Training selected synaptic gain **1.0** and sugar drive **50 Hz**.
Thus the selected candidate did not change the released synaptic scale.

| Evaluation | Baseline: gain 1, 200 Hz | Selected: gain 1, 50 Hz |
| --- | ---: | ---: |
| Training, 14 flies | 0.59389 | 0.58707 |
| Test, 7 flies, simulation seed 43 | 0.32846 | 0.33006 |
| Same test flies, simulation seed 144 | 0.32920 | 0.32616 |

Values are normalized mean squared error, lower is better. Both models have
their fluorescence parameters fitted on training data only. The selected model
is about 0.49% worse on the first test simulation and 0.92% better with the
second seed. The direction reverses and the size is small. **The pilot does not
support replacing the live baseline.** The bootstrap interval from the first
seed alone must not be mistaken for robustness to simulation or protocol
uncertainty.

The plotted measurements show transient responses that decline while the
workbook's sugar-stimulus flag remains on. The constant-input model largely
holds its response until stimulus offset. Some measured activity also rises
before the marked onset. Global coupling/input scaling cannot resolve these
waveform differences. Stimulus/frame alignment and sensory or circuit
adaptation are concrete priorities for the next model revision; tuning coupling
alone would risk compensating for the wrong mechanism. A new revision would
need additional untouched validation data because these test traces have now
been inspected.

Software checks passed: CUDA/Brian2 counts matched under identical explicit
inputs, voltage and synaptic-state differences were below 1e-9 mV, gain changes
produced the expected causal response, and returning to gain 1 restored the
baseline exactly. Numerical recovery and train/test-isolation tests also passed.

## Reproduction

The checked-in JSON contains the inspected source observations and metadata.
To regenerate it, run `scripts/prepare_gcamp.py` using Python with `openpyxl`
and the pinned local FlyWire annotation file.

Inside the allocated B300 environment:

```sh
.venv/bin/python scripts/validate_cuda.py
.venv/bin/python scripts/validate_gcamp.py --cuda
.venv/bin/python scripts/calibrate_gcamp.py --simulate --fit --seed-check
```

Generate the plot with Python containing NumPy, SciPy and Matplotlib:

```sh
python scripts/calibrate_gcamp.py --plot
```

Results are written under `outputs/gcamp-calibration/`: raw model-rate grid,
fit report, different-seed check, and PNG/PDF comparison. No fitted profile is
automatically loaded by the server. The gain setter is opt-in for offline CUDA
experiments, defaults to 1, and preserves the released numerical baseline.
