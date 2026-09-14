# Four-sensor assignment sensitivity experiment

## Design fixed before running

This experiment tests all 16 binary assignments of the four selected MaleCNS
sensory neurons to flexion (F) or extension (E) response families. Letter order
is ascending body ID: 815843, 817680, 912317, 935383. The labels are hypotheses;
no winning biological mapping can be selected without matched downstream data.

Use midpoint and width from the previously frozen Mamiya 2023 observation fit
in `wasm/calibration/claw-cells-fit.json`. Convert the sigmoid's **shape** to a
dimensionless value between 0 and 1, then multiply by an assumed maximum rate.
Discard fitted fluorescence offsets, amplitudes and relaxation times. In
particular, the 1–3 second observation relaxation is not a neural time constant.
An instantaneous sensory rate is an explicit unvalidated hypothesis here.
Rates are absolute angle-dependent and can be tonic at the 100-degree rest
angle. This is not a measured conversion from calcium to spikes.

- Primary setting: 150 Hz maximum, zero midpoint shift.
- Primary perturbations: flex/extend 10, 30 and 50 degrees, 0.1 s ramp, plus
  flex/extend 30 degrees with 0.2 s ramp. Rest 100 degrees, onset 0.3 s, hold
  0.2 s, total duration 1.6 s. All imposed angles are within the body model's
  physical range. These speeds exceed the slow calcium recordings and are
  model stress tests, not validated sensory dynamics.
- Sensitivity settings: maximum rates 75 and 300 Hz; or both response-family
  midpoints shifted by -10 and +10 degrees, with nominal 150 Hz. Run flex/extend
  30 degrees at the 0.1 s ramp for these four one-factor changes. Do not fit any
  parameters to the desired movement.
- Seeds 1, 2 and 3 for every assignment/setting/protocol. Common sensory random
  draws across assignments at a given seed reduce Monte Carlo comparison noise;
  seeds are simulation replicates, not biological samples.
- For every ramp duration, add a zero-displacement sham with the same clamp and
  release schedule. For each perturbation/sham, compute one disconnected passive
  trace; it is independent of mapping, sensory gain and seed. Preserve this
  rationale when reusing controls. In total: 1,056 connected trials plus ten
  disconnected controls. Additional regression/silencing controls are separate.
- Evoked circuit angle effect = (connected perturbation - disconnected
  perturbation) - (connected sham - disconnected sham), comparing matching times
  after release. This removes the corresponding tonic/sham effect, although
  nonlinear muscle mechanics still make it an operational contrast.
- Report angle traces, signed restoring contrast, total and evoked motor spikes,
  and body-ID spike counts. No desired-reflex score or best biological mapping.
- Compare every pair of assignments using RMS differences of their mean evoked
  angle traces and pooled flexor/extensor activity. Report resolution-dependent
  indistinguishability at 0.5 degrees and 5 Hz (assumed measurement resolutions,
  not measured experimental noise). Also report seed-to-seed variation.
- Suggest a perturbation/readout by candidate separation under this model;
  this is a proposed experiment, not biological evidence. Preserve gain and
  threshold dependence rather than selecting a favorable setting.

The original `brain/leg_reflex.py`, sensory encoder, neural parameters, muscle
coupling and live maze are unchanged. A separate vector-input circuit subclass
uses the original LIF dynamics and is checked against the original integrator
and physical default trial under identical legacy input. Anatomical weights,
unknown-transmitter handling and omitted inputs are unchanged.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_assignment_sweep.py --workers 6
.venv/bin/python scripts/check_assignment_sweep.py
```

Use the existing B300 Slurm allocation with six CPU workers, leaving capacity
for the live server. Full-precision trial outputs and frozen source hashes are
saved under `outputs/assignment-sweep/`. Browser traces are display reductions;
all analyses use the full saved 5 ms traces.
