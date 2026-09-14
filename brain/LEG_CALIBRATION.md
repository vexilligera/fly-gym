# Left-front-leg calibration experiment

This experiment implements a reduced MaleCNS nerve-cord circuit coupled to the
FlyMimic muscle model, measurement import, a calcium-response calibration
reference, and causal controls. It is available at `/calibration/`. The videos
are recorded MuJoCo trials played five times slower than simulation time.
The maze controller and its female FlyWire brain are unchanged.

**Status: anatomy connected; physiological calibration not accepted.** The
calcium fit does not identify the parameters of the simulated circuit. The
recorded 13Bα cells have no verified MaleCNS body-ID match in this implementation.
No parameters from this fit are copied into the neural engine or the maze.

## Anatomical circuit

Official MaleCNS v1.0 download source:
https://janelia-flyem.github.io/male-cns/download/ (CC BY 4.0).

`scripts/prepare_leg_connectome.py` reads the full segment connection table,
curated body annotations and consensus transmitter predictions. SHA-256 hashes
and source URLs are retained in `calibration_data/leg_connectome.json`.
The 151,856,684 source rows include segmentation fragments, not that many
connections between curated neurons. Selection only uses annotated neurons:

- Four left ProLN sensory neurons typed SNpp50/SNpp51: body IDs 815843,
  817680, 912317 and 935383. The release calls these types FeCO claw in annotated
  homologues. Their individual position tuning is not resolved.
- Five left front-leg `Ti flexor MN` cells: 807165, 809912, 818057, 819384,
  909831; two `Ti extensor MN` cells: 800636, 815344.
- Every annotated VNC intrinsic neuron on a two-edge path between those sensory
  and motor populations. Then retain all induced connections, without applying
  a synapse-count threshold or adding synthetic edges.

Result: **194 neurons, 8,133 edges, 80,165 anatomical synapses**. Only 19.95% of
the incoming anatomical synapses onto these selected neurons remain inside the
subset. This is a deliberately reduced circuit, not a full VNC/CNS simulation.
Intermediate neurons can be on either side, according to the actual graph.
An IN13B lineage label does not establish a match to recorded 13Bα neurons.

## Neural and physical assumptions

`leg_reflex.py` uses a separate NumPy/SciPy LIF engine, with provisional
rest/reset -52 mV, threshold -45 mV, membrane tau 20 ms, synaptic tau 5 ms,
delay 1.8 ms, refractory 2.2 ms, dt 0.1 ms and 0.275 mV per anatomical synapse.
These are starting assumptions from the earlier brain model, not measured
parameters of these VNC cell types. This standalone discretization is not
claimed to be numerically identical to the main Brian2/CUDA brain engine.
Acetylcholine is positive, GABA negative, and glutamate provisionally negative
inside the CNS; an alternative positive-glutamate run checks sensitivity.
Three unclear-transmitter cells have zero outgoing dynamical weights; their
anatomical edges remain in the data file. Receptor-specific signs, electrical
synapses, tonic drive and neuromodulation are absent.

Actual landmark-derived femur–tibia angle feeds a provisional encoder:
`rate = 150 * clip((angle - 100) / 20, 0, 1)` Hz. All four inputs receive
independent Bernoulli spike streams with this rate. This makes every selected
cell extension-tuned, an explicit hypothesis rather than an inferred receptive
field. No interneuron or motor neuron is directly stimulated during this test.

Motor population mean rates are smoothed over 50 ms; muscle excitation is
`clip(0.2 * rate / 100 Hz, 0, 1)`. Flexor and extensor pools drive FlyMimic's
`LFTibia_flex_93434` and `LFTibia_extensor_93932` Hill-type actuators.
Pooling loses slow/intermediate/fast recruitment and individual motor-unit
properties. The other 13 muscle actuators receive zero excitation.

The original 15-muscle body is restrained by joint equality constraints except
for the tibia. A constant external torque balances the resting tibia; the same
torque applies in all trials. The fixture holds the leg at 100°, extends it to
120° from 0.3–0.4 s, holds until 0.6 s and releases. MuJoCo runs at 0.1 ms;
neural/body exchange occurs every 1 ms. Anatomical angles are computed from
body landmarks, since increasing MJCF tibia qpos actually *decreases* this angle.
The fixture logs muscle forces in model units; these are not measured or
validated leg-tip forces. The body retains its original muscle parameters.

No CPG, imitation-learning policy or trajectory replay supplies muscle commands.
The mechanically imposed perturbation is the stimulus, not an output command.

## Experimental observations and splits

Agrawal et al., *Central processing of leg proprioception in Drosophila*,
eLife 2020, https://doi.org/10.7554/eLife.60299.
Raw data: https://doi.org/10.5061/dryad.k3j9kd55t (CC0).

The public website's `Imaging_data.zip` and `Behavior_data.zip` download links
work in a browser. During this run, API downloads returned 401 and direct
command-line website downloads returned 403; the public Chrome downloads
succeeded without login. Raw ZIPs are kept under `outputs/leg-calibration-source/`.

`prepare_leg_recordings.py` preserves 12 GCaMP6f traces from 13Bα ramp-and-hold,
extension-first and flexion-first swing protocols; source MATLAB file and cell
indices are recorded. Missing samples remain null. It also preserves per-fly
mean stimulation/control movement traces from four unloaded and seven loaded
headless flies receiving 720 ms 13Bα optogenetic activation.

Calcium training uses ramp records 1–2; ramp record 3 is held out. All nine swing
records are protocol tests. The small integer fly labels are not confirmed to
refer to the same animals across files, so cross-protocol results are not
claimed to be validation on independent animals. The source README gives
7.57 Hz while article methods give 8.01 Hz. Both rates are evaluated.

The effective observation model is a logistic angle-response curve followed by
a causal first-order filter, fitted jointly with a fluorescence gain. It is
baseline-centered using the pre-movement part of each trace. Interpolation
provides continuous angle input across missing samples; only originally finite
angle/fluorescence pairs contribute to scores. Four parameters are fitted to
training records only. Its gain is not a conversion to Hz and its time constant
combines neural and indicator dynamics. It omits adaptation and fluorescence
saturation. Multiple optimizer starts are selected using training residuals.

Behavior times are provisionally assigned 300 Hz with the first sample -0.2 s
relative to stimulation, based on the methods; the exported MAT file lacks a
timestamp/laser channel. Summaries use the last 200 ms of stimulation minus the
preceding 200 ms. They are retained as reference observations, **not fitted**:
the 13Bα optogenetic experiment is not equivalent to the fixture's mechanical
extension. Behavior splits reserve unloaded fly 4 and all loaded flies for
future evaluation; no fitting has been done on those movement records.

## Measured outcome

The seed-1 connected circuit produced 17 motor spikes and up to **2.415°** of
additional flexion relative to its disconnected counterpart. Disconnection
left sensory input intact but produced zero motor spikes. Sensory-off produced
zero neural spikes. Motor silencing retained 419 other neural spikes and gave
the same passive movement as the disconnected trial. Passive spring and muscle
forces return the leg even without neural activity, so return-to-rest alone is
not evidence of a reflex.

Seeds 2 and 3 produced 1.757° and 1.427° additional flexion. Treating CNS
glutamate as positive gave 1.034° in seed 1. These are sensitivity results of
the provisional model, not biological confidence intervals.

The held-out ramp calcium record had MSE 0.04444, 15.9% of the baseline-centered
zero-response error. **All five extension-first swing records were worse than
zero response**, with relative MSE 1.89–31.08. A fit to one stimulus therefore
does not justify transferring parameters to the circuit. Initial conditions,
fluorescence normalization, stimulus history and adaptation require work before
interpreting the mismatch as a neural mechanism.

`validate_leg_reflex.py` verifies anatomical laterality, motor direction in
physics, disconnected/silenced controls, finite trajectories, matched clocks,
and exact independence of fitted parameters from corrupted held-out data.
The promotion gate rejects unverified cell matching and missing spike-to-force
calibration regardless of the numerical observation fit.

The complete experiment and validation also passed on B300 node
`slurm-b300-128-021`, Slurm allocation `5807547`, using commit `7211d29`.
The cluster reproduced all spike counts and the 2.414848° causal effect
(local/cluster difference below 3e-13 degrees). The published replay, figure,
report and validation are the cluster-generated artifacts. The report records
input and implementation hashes plus Python, NumPy, SciPy and MuJoCo versions.

## Reproduction

Use an environment with current FlyGym 2.1 and its MuJoCo 3.9 dependency.
The maze's older legacy FlyGym environment should remain separate if present.

```sh
python scripts/prepare_leg_recordings.py
python scripts/prepare_leg_connectome.py --source brain/data/male-cns-v1.0
python scripts/calibrate_leg.py --render
python scripts/validate_leg_reflex.py
```

The compact, versioned data files permit the last two commands without
downloading the full connectome or raw recordings again. Rendering on Linux
requires EGL (the cluster uses its existing `deploy/egl` loader). Results,
traces, replay videos, figure and validation JSON go to `outputs/leg-calibration/`.
The reviewed artifacts are copied into `wasm/calibration/` for the static
Tailscale viewer. A new fit does not automatically publish or change the maze.

Next physiological work: verify a 13Bα connectome match, obtain identified
motor-unit spike/force traces, resolve sensory tuning and source timing, then
fit neuron and muscle parameters against matched stimulation protocols with
untouched validation observations.
