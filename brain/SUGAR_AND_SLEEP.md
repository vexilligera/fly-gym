# Sugar interaction and sleep: wiring and limits

The implemented first step is a **sugar-taste circuit assay after maze arrival**.
It lets us watch modeled sensory and downstream feeding-initiation activity,
plus an articulated proboscis driven by an engineered MN9-to-servo mapping.
It does not transport liquid, simulate pumping or digestion, or measure pleasure. Sleep is a feasible extension, but is not implemented or demonstrated
by this assay's silent baseline.

## What is wired now

```text
Maze food-zone arrival (thorax within 2.5 mm)
  → user starts a separate taste presentation with torso and legs held
  → brain reset; 2 s baseline / 4 s sugar input / 2 s washout
  → 21 released sugar GRNs receive imposed 0–200 Hz input
  → all 138,639 neurons and 15,091,983 connection rows remain active in the model
  → actual modeled spikes in the full brain and the two MN9 motor readouts
  → engineered joint targets → MuJoCo rostrum/haustellum movement
```

The 21 input IDs are copied exactly from the pinned Eon release's
[`EXPERIMENTS['sugar']`](https://github.com/eonsystemspbc/fly-brain/blob/a3db62f9436074e485c0278290c2164ed6150808/code/benchmark.py).
They are stored in [`sugar_input.py`](sugar_input.py). Current v783 annotations
label this unilateral cohort `LB3`, `gustatory`, `sugar/water`, on the left.
The curated experimental sugar assignment is more specific than the annotation
label alone; we do not stimulate every cell labelled sugar/water.

The released example identifies the two MN9 readouts. Current v783 annotations
call them `CB0701`, ingestion motor neurons:

| Readout | FlyWire root ID | How it is used |
|---|---|---|
| MN9 left | 720575940618238523 | Observed only |
| MN9 right | 720575940660219265 | Observed only; opposite the stimulated cohort |

Older notebooks contain a missing sugar ID, a missing MN9 ID, and conflicting
side comments. The implementation uses the current benchmark cohort and only
readouts present in v783, with sides taken from the current annotations. It
fails explicitly on mismatches instead of silently substituting neurons.

[Shiu et al., Nature 2024](https://doi.org/10.1038/s41586-024-07763-9)
studied this sensory-to-feeding-initiation transformation. MN9 is associated
with rostrum lifting during proboscis extension. Its spikes are not a direct
measurement of swallowing, consumption, satiety, reward, or happiness. That
paper's physiological validation also does not validate our whole maze or
provide a calibrated conversion from sucrose concentration to imposed Hz.

The assay's source input is a Poisson-like event drive at the chosen rate. It
uses the existing LIF model: 0.1 ms steps, 20 ms membrane time constant, 5 ms
synaptic decay, 1.8 ms transmission delay, and released signed synapse counts
scaled by 0.275 mV. Source events add the existing 68.75 mV input kick. No MN9,
dopamine, or reward neurons are directly driven.

During the assay, vision and odor input are off to isolate the taste response.
The torso and legs remain held at their actual arrival posture and time. A
separate posed copy of the MuJoCo scene articulates the mouth. The brain and
proboscis share a separate 0–8 s assay clock. The display runs at 0.2× neural time so the
response is watchable. The 3D view shows measured model spikes per 20 ms bin;
the chart aggregates actual counts into 100 ms bins. A 0 Hz control repeats the
same reset and timing. This is a controlled stimulus protocol, not natural
feeding emerging from the navigation circuit.

New sugar inputs use an independent CUDA random stream (seed 43), preserving
the existing navigation input stream (seed 42). A recurrent-network check
verified identical old-channel spike counts, membrane voltages, and synaptic
states with the inactive sugar channels added. The CUDA/Brian2 numerical
comparison also passed. Full-network assay measurements on the B300:

| Imposed sugar input | Mean GRN firing | Mean MN9 left | Mean MN9 right |
|---|---:|---:|---:|
| 0 Hz | 0 Hz | 0 Hz | 0 Hz |
| 50 Hz | 48.55 Hz | 14.75 Hz | 18.50 Hz |
| 100 Hz | 98.01 Hz | 50.75 Hz | 68.50 Hz |
| 200 Hz | 195.96 Hz | 62.75 Hz | 94.50 Hz |

These are averages over the four-second stimulus in one reset simulation per
rate, not animal measurements or confidence intervals. They establish that
the implemented input produces a dose-dependent downstream response in this
model. `scripts/validate_sugar.py` reproduces the checks; detailed traces are
in `outputs/sugar-validation.json`. HTTP controls, pause/resume, ownership,
invalid-input handling, held torso/leg timing, and the 0 Hz control are checked by
`scripts/validate_sugar_api.py` against a staging service.

The deployed maze was rerun after this change. It reached the food zone at
81.76 simulated seconds, with all 4,089 recorded path points identical to the
previous run. Starting the assay through the browser reproduced the 200 Hz
results above while preserving the body's position, path, and arrival time.
The live 3D view displayed taste-driven network spikes during stimulation;
the completed chart showed the baseline, response, and washout without browser
errors. The assay can be replayed from the arrival state with **Watch proboscis +
brain**, or compared with **Run no-taste control**.

## Visible proboscis action

The native NeuroMechFly rostrum and haustellum meshes and masses are retained.
`brain/proboscis.py` freezes every existing body at its measured arrival pose,
then adds three dynamic hinges and position servos in a separate MuJoCo model.
The original maze model, path, and walking dynamics are untouched. The main
MJPEG camera switches to a three-quarter close-up; **Enlarge fly** opens it
full-screen. Maze walls are hidden in this observer close-up to prevent occlusion;
the navigation scene is unchanged. The neural and mouth clocks advance together in 20 ms bins, with
200 MuJoCo steps per bin. Playback is paced to at most 0.2× real time.

Only measured left/right MN9 firing rates enter the mouth decoder. After a
120 ms low-pass filter, the mean rate divided by 100 Hz sets a bounded extension
fraction. This sets rostrum pitch from 0 to −100° and coupled haustellum pitch
from 0 to +70°. Yaw is 25° × extension × (right − left)/(right + left + 20 Hz).
The right MN9 is contralateral to the released left sugar-GRN cohort. This
chosen sign illustrates turning toward the stimulated side; the angle gain,
haustellum coupling, and frequency-to-angle conversion are not established by
the connectome or calibrated to an animal. No oscillation or sucking cycle is
invented. The mouth retracts when the MN9 response subsides.

These are gravity-compensated, damped joint dynamics, with no mouth collision,
contact-triggered taste, fluid intake, or feedback to the brain. The food-zone
arrival threshold does not establish that the mouth touches the sugar. This
visualizes a feeding-initiation command and an approximate movement response;
it is not a complete physical feeding loop.

`scripts/validate_proboscis.py` checks bilateral symmetry, bounded targets,
stationary zero input, extension and retraction, dynamic joint limits, preservation
of all fixed body poses, unchanged navigation state, rendering, and reset.
`scripts/validate_sugar_api.py` additionally checks that full-connectome MN9
activity causes movement, the no-taste control stays still, and brain/mouth
clocks and MJPEG assay timestamps agree.

Both validations passed locally/on the B300 as applicable. In the full-network
200 Hz API trial, sampled rostrum extension reached 94.70° and yaw reached
5.65° left; the angles returned to within 0.001° of rest after washout. The
no-taste control remained within 10⁻⁶° of rest. These are outputs of the chosen
servo mapping, not measured animal kinematics.

## How to extend this into actual sugar interaction

1. Add contact sensors on the tarsi/labellum and a sugar-solution surface.
   The existing 2.5 mm arrival radius is an evaluation zone, not proof of
   mouth contact. Use physical contact to gate taste input, distinct from the
   airborne food-odor field.
2. Calibrate concentration-to-GRN activity from experiments, including onset,
   adaptation, mixture effects, and nutritional-state dependence. The current
   50/100/200 Hz controls are stimulation settings, not mM or ppm.
3. Replace the approximate three-hinge servos with validated joint limits,
   collision geometry, muscle dynamics, and additional motor-neuron mappings.
   The current extension animation has dynamic joints but no sipping mechanics.
   MN9 alone is insufficient for the entire proboscis/pumping cycle.
4. Gate a fluid-consumption model on labellum contact and pumping. Track liquid
   remaining, ingested volume, and an explicitly modeled internal nutrient
   state. Motor spikes alone must not increment a consumption counter.
5. Add validated hunger/satiety and neuromodulatory dynamics if internal state
   is needed. Reward learning further requires plasticity and appropriate
   reward-circuit models. The present fixed-weight LIF network lacks those
   mechanisms. Label outputs as feeding, satiety, or modeled reward—not happiness.

## Can this fly sleep?

We can implement a **sleep-like state model** and test its behavior. The present
model cannot establish sleep: its neurons have uniform fixed parameters, no
circadian drive, no homeostatic sleep pressure, no sleep-specific modulation,
and zero basal input. Removing sensory input after a reset therefore produces
silence by construction. The walking decoder also supplies a constant drive;
it has no sleep/wake gate. Pausing the viewer or freezing the body is unrelated
to physiological sleep.

A useful first sleep extension needs:

- A slow homeostatic variable that accumulates during waking and dissipates
  during sleep, plus a separately specified circadian/light drive.
- Verified neuron mappings and state-dependent excitability or synaptic
  modulation for a chosen sleep/arousal circuit. Anatomical names alone do
  not provide those dynamics; stimulating all `FB6*` cells would not be justified.
- A defined route from that state to locomotor drive and responsiveness,
  with light/odor or another modeled stimulus used to test awakening.
- Tests distinguishing quiet wakefulness from sleep: sustained quiescence,
  reduced responsiveness to weak stimuli, reversibility with strong stimuli,
  and compensatory sleep after deprivation.
- Longer experiments and explicit time units. Traditional fly scoring often
  uses at least five minutes of inactivity; our 120 s maze limit and 8 s neural
  assay do not even reach that duration. Inactivity alone is insufficient.

Sleep circuitry is not wholly contained in our brain dataset. The dFB is
heterogeneous, and the widely used 23E10 driver also labels sleep-promoting
VNC-SP neurons outside the brain. Our model has no reconstructed VNC. A recent
[dFB-specific study](https://doi.org/10.1371/journal.pbio.3003014) supports roles
in sleep promotion and homeostasis while demonstrating why the exact targeted
population matters. We can study a selected brain circuit with stated limits;
we cannot infer a complete sleep mechanism from this connectome alone.

[Recent work on distinct fly sleep states](https://doi.org/10.1016/j.cub.2026.01.015)
also finds that pooling all inactivity of five minutes or more can obscure
homeostatic effects. Validation should therefore examine bout duration and
arousal/rebound behavior, not just a binary inactivity threshold. Neither
this literature nor a successful engineered sleep-like state demonstrates
subjective experience or dreaming in the simulated fly.
