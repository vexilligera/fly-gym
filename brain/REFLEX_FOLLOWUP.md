# Reflex follow-up: analysis design

The new sensory benchmark uses the public `claw_magnet_Mamiya2018.parquet`
release from Dallmann et al. (2025), DOI 10.5061/dryad.gqnk98t16. The original
recordings are from Mamiya et al. (2018), DOI 10.1016/j.neuron.2018.09.009.
The source is processed calcium from R73D10 claw axon regions, not identified
single neurons and not spike rates. Preserve source values and timestamps.

This design is fixed before fitting or inspecting prediction scores:

- Use only the X-branch ROI (`L1_x`) for fitting and primary evaluation. Preserve
  Y/Z recordings as unused reference data; cross-ROI animal identity is not verified.
- Train on flexion-first ramp trials from animal IDs 1–6. Select between two
  candidate observation models using flexion-first trials from IDs 7–8.
- Final animal test: IDs 9–10, both stimulus orders. Protocol transfer test:
  extension-first trials from IDs 1–8. These latter tests share animals with
  training/validation and are not independent-animal validation.
- Normalize only by the training calcium maximum; use equal weight per trial.
  Do not estimate gain, time shift, baseline, or other parameters on test data.
- Candidate 1: fourth-degree polynomial in `(angle - 90)/90` followed by a
  causal double-exponential calcium kernel. Candidate 2: the same polynomial
  plus quadratic terms multiplied by the most recent movement direction.
  Direction changes only when backward-difference velocity exceeds 5 deg/s;
  it is zero before the first such movement. This represents a limited history
  hypothesis motivated by the published hysteresis, not an identified mechanism.
- Fix kernel rise/decay at 0.03/0.30 s, following Dallmann's published prediction
  code. These are assumptions, not inferred indicator or membrane constants.
  Initialize filtering at the initial angle response and filter each trial
  independently with its actual source time step. Fit linear coefficients by
  least squares, with no cross-trial convolution or regularization search.
- Report MSE relative to a constant learned from training and Pearson correlation
  per trace. Retain failures. Also report a 1 s circularly shifted-input negative
  control, without treating its autocorrelated samples as independent replicates.
- Compare the frozen models with the original provisional extension-only
  encoder as an observation reference, with only its fluorescence gain and
  intercept fitted on training. No conversion of calcium to Hz is claimed.

The physical follow-up varies imposed flexion/extension (10, 20, 30 degrees),
ramp duration (0.05, 0.1, 0.2 seconds), and distal mass/inertia (1x or 2x), with
matched circuit-disconnected controls and fixed seed 1. These are simulation
sensitivity experiments, not measured fly reflexes. Retain the previous default
trial and verify its numerical outcome is unchanged. Do not transfer a pooled
claw calcium curve to particular MaleCNS sensory IDs or alter the maze.

## Recordings found and their use

| Release | Available measurements | Use in this experiment |
| --- | --- | --- |
| [Mamiya 2018, re-released by Dallmann 2025](https://doi.org/10.5061/dryad.gqnk98t16) | `claw_magnet_Mamiya2018.parquet`, 849,677 bytes; 60 processed GCaMP6f traces, 3 claw axon ROIs, 2 ramp orders; primary X-branch cohort has 10 animals; paired angles and timestamps | Downloaded, hashed, imported all 60; fit and score 20 X-branch traces. This is the most directly useful new sensory benchmark. |
| [Dallmann 2025](https://doi.org/10.1038/s41586-025-09554-2) | Claw treadmill data (144 MB), hook passive-replay and active-movement data, 9A and descending-neuron calcium; behavior annotations | File manifests and analysis code verified; larger datasets not imported here. Suitable future tests of behavioral context and presynaptic inhibition, with different normalization and indicator metadata to inspect before pooling. |
| [Agrawal 2020](https://doi.org/10.5061/dryad.k3j9kd55t) | 13Bα, 9Aα and 10Bα calcium, stimulus protocols, optogenetic movements, separate electrophysiology archive | Existing 12-trace 13Bα benchmark retained. Physiological cell matching remains unresolved; calcium alone does not supply a spike-to-force calibration. |
| [Pratt 2026](https://doi.org/10.1038/s41467-026-69333-z), [data](https://doi.org/10.5061/dryad.fxpnvx153) | CxHP8 hair-plate GCaMP7f with coxa kinematics; optogenetic activation/silencing and walking measurements | Public metadata, file list and paper verified. Useful for a future coxa limit-reflex fixture; not a tibia-claw calibration and not imported here. |
| [Mamiya 2023](https://doi.org/10.1016/j.neuron.2023.07.009) | Proprioceptor feature selectivity and topographic maps | Subsequently downloaded [public physiology data](https://doi.org/10.5061/dryad.dbrv15f6q). The separate [cell-body validation](CLAW_CELL_VALIDATION.md) imports 262 region traces and evaluates driver-specific GCaMP7f response curves. |

The [Dallmann codebase](https://github.com/chrisjdallmann/feco-inhibition) also
contains a Brian2 connectome simulation of MANC circuits, adapted from Shiu's
model. This is another useful published neural simulator, distinct from the
MaleCNS data-release repositories. Its existence does not supply a calibrated
complete brain–VNC–muscle model. Our new Python observation benchmark implements
the published kernel concept with independently fitted coefficients, explicit
animal splits and causal per-trial filtering; it does not execute their MATLAB
analysis or claim exact reproduction of their figures.

## Results

The validation animals selected the history model over the angle-only model
(normalized validation MSE 0.00648 versus 0.01379). Parameters were then frozen.

- Final test: two unseen animals, four trials. MSE is **41.13%** of the
  training-constant baseline error (58.87% reduction); median Pearson r **0.844**.
  All four beat the constant baseline, but individual relative MSE ranges from
  0.113 to 0.788. The model still misses response amplitude and some transitions.
- Opposite-order protocol test: eight trials from animals used in training or
  selection. MSE is **24.83%** of baseline; median r **0.902**; all eight beat the
  baseline. These are not eight additional independently held-out animals.
- The history model is not better than the static model on every trial. Model
  selection used validation data only; failures and alternative scores remain
  in the machine-readable report.
- The 1 s shifted-input control increases aggregate error in both test groups,
  but autocorrelation makes it a weak temporal check, not proof of mechanism.

The physical sweep uses the original provisional encoder, not the new fit:

| Perturbation from 100° | Motor spikes | Maximum connected/disconnected difference |
| --- | ---: | ---: |
| Flex 10°, 20° or 30° | 0 in each | 0° in each |
| Extend 10° | 2 | 0.463° |
| Extend 20° | 17 | 2.415° |
| Extend 30° | 18 | 2.428° |
| Extend 20°, 0.05 s ramp | 17 | 2.206° |
| Extend 20°, 0.20 s ramp | 24 | 2.029° |
| Extend 20°, doubled distal mass/inertia | 17 | 2.376° |

Every disconnected control has zero motor spikes. The flexion failure is an
input-model limitation, not evidence that real flies lack a flexion reflex.
Real claw axon pools have mixed angle tuning and history dependence; those
population curves cannot establish which of our four selected MaleCNS sensory
cells should activate or inhibit the particular motor pools. A successful
fluorescence prediction therefore does not justify inserting it as a firing
rate or driving all four cells identically.

## Reproduce

Download the small public Parquet file and its README from Dryad into
`outputs/leg-calibration-source/dallmann2025/`, then run:

```sh
python scripts/prepare_claw_recordings.py
python scripts/continue_leg_reflex.py
python scripts/validate_reflex_followup.py
```

The last two commands only need the compact versioned data. Output includes a
frozen fit, every test score, original/predicted view traces, all nine paired
physical trials, a PNG/PDF figure, runtime/source hashes and validation JSON.
The viewer adds interactive recording and perturbation selectors at
`/calibration/`. The original MuJoCo video remains the 20° extension trial.

The full follow-up ran locally and on B300 node `slurm-b300-128-021` in Slurm
allocation `5807547` from experiment commit `528079a`. Every spike count agreed;
physical angle differences agreed to within 1e-10 degrees. Calcium coefficients
agreed to numerical precision across the two SciPy/NumPy runtimes. Published
artifacts are the B300-generated results. Validation checks animal separation,
held-out-data poisoning, causal filtering, all nine disconnected controls,
finite mechanics, and reproduction of the original 20° reflex.

Next: resolve flexion/extension subtypes within the actual sensory cell IDs,
obtain matched physiological or optogenetic motor recordings, and validate
bidirectional muscle recruitment. Do not use this pooled observation benchmark
as a substitute for those measurements.
