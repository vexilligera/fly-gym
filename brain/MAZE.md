# Vision and olfaction in the sugar maze

Open `/maze/` and press **Run maze**. The default **Branching maze** has 25
connected grid cells, 20 wall pieces, and five terminal cells (including the
start and sugar cells). The unique cell-center route to sugar passes through
12 openings and makes eight turns. Corridors have 7 mm of clear width. The
fly starts at (−16, −16) mm in the 40 × 40 mm arena. The original **Simple
maze** with two offset baffles and a (−13, −12) mm start remains selectable.
The sugar patch stays at the center of both layouts. Geometry stays fixed
between trials; the trial seed changes leg phases, not the maze.
Arrival means the thorax enters a 2.5 mm radius; no eating is simulated.
Both eyes, two antenna readings, the body, path, and brain activity update live.
After arrival, **Watch proboscis + brain** opens a separate neural/mouth assay:
2 s baseline, 4 s input to the released sugar GRNs, and 2 s washout. A no-taste
control is available. The brain resets and shares an assay clock with a
three-hinge MuJoCo proboscis. The torso and legs are held in a declared feeding pose facing sugar solution;
the recorded navigation arrival remains unchanged. Labellum contact gates taste. The
close-up streams extension and turning driven by an approximate MN9 decoder. See [SUGAR_AND_SLEEP.md](SUGAR_AND_SLEEP.md) for
the exact wiring, measured responses, ingestion limits, and sleep feasibility.
Choose vision + smell, either sense alone, or both disconnected. **Food odor
on** independently controls the source for the next trial.

**Neuron readouts** is now the default movement source. The earlier
**Sensory policy** remains selectable as a comparison. **Silence descending
neurons** suppresses all annotated bilateral DNp09, DNa02, and MDN cells from
the start of a neuron-readout trial, while keeping the selected sensory inputs.
Changing these controls applies to the next Run/Reset. Pause/Resume preserves
the active trial's controller. The live table shows actual population firing
rates smoothed over 100 ms, with raw 20 ms rates on hover.

The server advances the entire loop every 20 ms, independently of browser
polling. Pause stops at a completed bin; Resume continues the same trial.
Run maze and Reset start a fresh brain and body. The default time limit is
120 simulated seconds; 5, 15, 30, and 60 seconds remain selectable. The path retains
the entire trial. A trial stops on arrival,
loss of balance, or its time limit. The manual and stripe experiments share
this brain; a running experiment must be paused before switching modes.

The body camera uses a persistent MJPEG stream. Frames include their own
simulation timestamp and can arrive more frequently than brain/eye readouts.
The existing odor overlay is unchanged and uses the active layout's field.
Layout changes rebuild the body world and field on the simulation worker;
the browser replaces the wall map and groups trial results by layout.

## Sensory and neural model

1. The actual MuJoCo scene is rendered by two body-attached eye cameras.
   NeuroMechFly's fisheye optics and 721 samples per eye are retained. Dark
   contrast in a fixed retinal mask drives the 7,932 annotated R1–6 cells
   through the synthetic registration described in [VISION.md](VISION.md).
2. Two virtual antenna sensors, fixed in the thorax frame at
   (0.65, ±0.38, 0.08) mm, sample a normalized odor field. There is no active
   antennal movement. The odor source is co-located with the sugar; sugar
   itself is not treated as a volatile chemical.
3. `OdorField` solves the steady equation `D ∇²c − λc = 0` on a 0.5 mm grid,
   with D = 12 mm²/s, λ = 0.12/s, c = 1 in the central source, and no flux
   through walls. These are chosen simulation parameters, not a calibrated
   food odor. Bilinear sampling uses only unblocked neighbors. Odor spreads
   around openings; a sealed compartment receives no source odor. There is
   no wind, turbulence, vertical flow, or evolving plume. Turning the source
   off starts with zero odor everywhere, without a modeled washout period.
4. A chosen encoder converts the two local samples into Poisson input rates
   for 35 left and 33 right `ORN_DM1` cells. The common rate is
   `110 mean(c)/(mean(c)+0.04)` Hz. Bilateral contrast is amplified by 16,
   clipped to ±0.85, and applied as a left/right rate multiplier. This
   comparison occurs **outside the brain** and exaggerates small antenna
   differences. It is not a measured olfactory transduction model or a
   claim that DM1 encodes sucrose. Only one food-odor channel is represented.
5. Eye and odor events enter the same 20 ms simulation interval of all
   138,639 neurons and 15,091,983 released signed connection rows. No
   connections are removed. L2 and DM1 projection-neuron activity is produced
   by the recurrent network; those readout cells are not directly stimulated.
6. In default neuron-readout mode, only the smoothed descending-neuron rates
   determine the motor gains. No DN is directly stimulated during navigation.
   The exact shared adapter in `brain/motor_readout.py` uses:

   ```text
   F = mean(DNp09 left, right) / 100 Hz
   R = mean(MDN left, right) / 100 Hz
   T = (DNa02 left − right) / 100 Hz
   leg gains = clip([F − R − 0.6 T, F − R + 0.6 T], −1.2, 1.2)
   ```

   Rates use a causal exponential filter with a 100 ms time constant, reset
   to zero. Zero readout gives zero gain; negative gain reverses CPG phase.
   This adapter is the same one used on the manual brain-stimulation page.
   There is no constant walking drive, wall avoidance, odor-following rule,
   escape timer, direct L2/ORN steering, or successful-policy fallback in this
   mode. L2 and ORN/PN activity remain visible for inspection.
7. The optional comparison policy reads L2 counts and **ORN** spike rates.
   L2 sector responses are smoothed over 60 ms and produce wall avoidance;
   left/right ORN activity is smoothed over 120 ms and produces odor steering.
   A strong front response triggers a short held turn. These signals modulate
   a constant walking drive through the existing six-leg recorded-step CPG.
   DM1 projection neurons are displayed as downstream responses, but do not
   steer this controller. Their released inputs are substantially bilateral.

`MazePolicy` receives neural activity, fixed synthetic retinal directions,
modality switches, and its history. It receives no body pose, target position,
maze geometry, odor map, distance-to-goal, or waypoint plan. World coordinates
are used only to construct the scene/field, sample sensors, and score arrival.
The UI's odor map is an observer visualization.

Removing the sensory policy does **not** remove all engineered assumptions.
Sensory encoders still impose contrast/rate transforms and a bilateral odor
contrast gain. The descending-neuron roles, gain scales, and recorded-step CPG
are an uncalibrated brain-to-body interface, not a reconstructed VNC. The model
has no spontaneous drive, internal state, plasticity, or leg/proprioceptive
feedback into the connectome. DN activity can be silent, weak, or inappropriate;
neuron-controlled motion does not establish accurate innate navigation.
The GCaMP pilot did not justify changing the live neural parameters.

`scripts/validate_neuron_navigation.py` runs a 120 s default trial, 5 s matched
vision-only/odor-only/no-senses/DN-silenced controls, a simple-maze comparison
policy trial, and a post-reset direct-stimulation positive control. It checks
that DN cells receive no external input during navigation, every delivered
gain matches the common adapter, the sensory policy is never called, and
silencing removes DN spikes while preserving sensory activity. It saves every
20 ms rate/gain/position sample under `outputs/neuron-navigation/`. These are
single-start exploratory tests, not a population navigation benchmark.

To test just the easier layout with the same neuron interface, run
`scripts/validate_neuron_navigation.py --layout simple --trial-only --duration 120`
on the allocated GPU. This records every neural/body bin separately under
`outputs/neuron-navigation-simple/`, without replacing the branching-maze
records or changing the live browser trial. The page displays the measured
neuron-readout result for the active maze layout.

## Brain chemistry: what “interaction” means here

The connectome engine uses uniform point-neuron LIF dynamics: 20 ms membrane
time constant, 5 ms synaptic decay, 1.8 ms transmission delay, and released
signed synapse counts scaled by 0.275 mV. External Poisson events add voltage
to selected sensory neurons. A cell's spikes affect its connected neighbors
through those weights. All cells and released wiring remain in the model.

This approximates electrical consequences of transmission; it does not
simulate neurotransmitter molecules. There are no ligand concentrations at
receptors, binding kinetics, receptor subtypes, release vesicles, reuptake,
drug diffusion, detailed ion channels, or molecule-specific dose responses.
Real photoreceptors use graded potentials and histamine; the released LIF
model's signs and the artificial event encoder do not reproduce that pathway.
The antenna field is an environmental scent proxy, not a biochemical model
inside the brain. A separate imposed sugar-GRN assay now probes taste-related
circuit activity and drives an engineered proboscis articulation. Hunger, reward
learning, contact-controlled feeding, pumping, digestion, and
metabolism remain absent. Adding a chemical name would not establish its effects.

## Measured checks and interpretation

Neuron-readout checks on B300, revision `d5b9769`, used the branching maze,
heading 75°, leg seed 1, and the fixed backend neural stream:

| Neuron-readout trial | Duration | DNp09 / MDN spikes | DNa02 spikes, left / right | Peak absolute gain | Result |
|---|---:|---:|---:|---:|---|
| Vision + smell | 120 s | 0 / 0 | 6,231 / 86 | 0.522 | No arrival; 25.83 mm from sugar |
| Vision only | 5 s | 0 / 0 | 0 / 0 | 0 | No locomotor command |
| Smell only | 5 s | 0 / 0 | 232 / 2 | 0.427 | Turning; no arrival |
| Both disconnected | 5 s | 0 / 0 | 0 / 0 | 0 | Entire brain silent |
| DNs silenced, senses on | 5 s | 0 / 0 | 0 / 0 | 0 | Sensory/network activity persists; no motor command |

All bilateral DNp09 and MDN cells were silent during the 120 s trial, so the
adapter produced only differential leg commands. The body moved within the
starting corner, ending 3.30 mm from its initial position and farther from
sugar. Zero-gain controls had only 0.0031 mm net passive drift over 5 s.
The silenced trial still produced 2,852,015 network spikes, separating the
absence of motor commands from the absence of sensory activity. DN cells
were never direct input targets in these trials; the sensory policy was
guarded against being called. Its separately selected simple-maze regression
still reached sugar in 1.60 s. These different-duration/layout runs are
mechanism and regression checks, not matched estimates of navigation success.
The compact results are versioned in `wasm/maze/readout-validation.json`.

This shows a working neural-readout interface and a failure of useful maze
navigation in the tested configuration. It does not identify a unique cause:
sensory mapping, neural dynamics, choice of descending cells, and motor
decoding all remain incompletely validated. No forward bias or fallback was
added to make the experiment succeed.

A separate simple-maze neuron-readout trial on B300, revision `dbf4426`, used
the same 75° heading, leg seed 1, both sensory inputs, and unchanged neural and
motor parameters. It ran for the full **120 s without arrival**. Distance from
sugar started at 17.60 mm, reached a minimum of 16.19 mm, and ended at 19.27 mm;
the arrival threshold is 2.5 mm. DNa02 emitted 6,354 left and 61 right spikes.
All DNp09 and MDN cells remained silent, so forward and reverse drive were zero
throughout. The body moved within the starting area; making the maze simpler
did not resolve the missing forward readout in this tested configuration.
Every bin passed the direct-input exclusion, shared-adapter, and brain/body
clock checks, and the sensory policy was never called. The earlier comparison
policy reached the simple maze's sugar zone in 1.60 s. Complete rate, command,
and body records are in `outputs/neuron-navigation-simple/`; the compact result
is also stored under `simple` in `wasm/maze/readout-validation.json`.

The historical navigation results below used **Sensory policy**. They are not
evidence for success with the default neuron readouts.

For the **simple maze**, `scripts/validate_maze.py` runs matched headings 45°, 75°, 105° with leg-phase
seeds 1, 2, 3 and a 5 s limit for each of five conditions. Neural randomness
resets to the fixed backend stream. B300 results on revision `12dbefa`:

| Condition | Arrivals | Arrival time, simulated seconds |
|---|---:|---|
| Vision + smell | 3/3 | 1.58–1.66 |
| Smell only | 3/3 | 1.56–1.62 |
| Vision only | 0/3 | — |
| Both disconnected | 0/3 | — |
| Combined, source off | 0/3 | — |

Combined trials generated both L2 and downstream DM1 projection-neuron
spikes. Both-disconnected trials remained neurally silent with constant
walking gains. Source-off and vision-only trials gave identical trajectories.
A separate straight-drive test verified actual wall contacts and blockage
below the left baffle's opening; contact can mechanically deflect the body
along a wall. Finite state and synchronized neural/body clocks are checked
on every step. A sealed-compartment field check verifies wall impermeability.

This small, smooth-field maze is easy to navigate using odor alone from these
starts. These data **do not demonstrate a benefit from adding vision**. Visual
input changes the controller's turns, but wall avoidance remains crude and
can get stuck. The decoder has no learning, memory map, planning, reconstructed
descending decision circuit, or VNC. Success is an engineering demonstration,
not evidence that natural navigation emerged from the full connectome or that
the rest of the brain is necessary for the behavior.

`scripts/validate_complex_maze.py` checks the harder layout independently.
All 25 cell centers connect with 1.5 mm clearance from the solid walls. The
default combined trial (75°, seed 1) timed out at 30 simulated seconds, 18.4 mm
from sugar; a second (90°, seed 2) timed out at 10 s, 21.6 mm away. Body centers
stayed outside walls in every recorded frame. The simple reference still
reached sugar in 1.60 s. **The existing reactive controller does not solve
these harder trials.** Geometry/connectivity checks do not imply navigation
success. No route planner, memory, or hidden waypoint input was added.

A subsequent live trial on revision `81f7eb4` ran the branching maze with
vision + smell, heading 75°, seed 1, and food odor for the full 60 simulated
seconds. It completed 3,000 synchronized brain/body steps and retained all
3,001 path points. The fly stayed upright, finished 19.9 mm from sugar, and
came no closer than 9.1 mm. Throughput was 0.353× real time, about 170 seconds
of simulation-worker time. This longer run also did not solve the maze.
The final state and checks are saved as `outputs/maze-60s-state.json` and
`outputs/maze-60s-summary.json` on the cluster and development checkout.

With the time limit raised to 120 s on revision `c573e3e`, the same condition,
heading, and seed **reached the food zone at 81.76 simulated seconds**, 2.424 mm
from its center. Arrival stopped the trial automatically after 4,088 steps,
about 237 wall seconds. The first 60 s of the body trajectory matched the
previous run exactly. Every recorded body center remained outside solid walls;
the fly stayed upright and the brain/body clocks remained synchronized.
This is one successful longer trial, not a reliability estimate or evidence
that odor alone caused arrival.

The earlier close pass was 9.067 mm from sugar at 1.20 s, behind a separating
wall, with eight cell openings still between that cell and the source. In a
nearby diagnostic sample at 1.14 s, antenna concentrations were only 0.001627
and 0.001616 (source concentration is 1). Smoothed left/right ORN rates were
4.46 and 3.08 Hz. Odor requested a +0.093 turn, while visual avoidance requested
−0.088, leaving a +0.006 net turn. Thus the sensory signal was present but weak,
and the two steering terms nearly canceled.

Across 1,135 snapshots in the longer run, 94.5% of commands used the fixed
wall-escape turn. In that mode, the decoder uses a held ±0.65 turn instead of
adding the continuous odor and visual terms; odor can still influence the
direction selected when an escape begins. Persistent front-wall activity can
retrigger this mode. The controller has no temporal scent-trend comparison,
dead-end memory, or route planning. These observations explain slow progress
despite odor input; a matched sensory-ablation test would be needed to measure
how much olfaction contributed to this eventual arrival.

Diagnostics were polled every 0.2 wall seconds plus request time, so the 94.5%
is a fraction of sampled commands, not an exact per-step fraction. The escape
indicator now describes the command actually applied, including the last bin
of each held turn; a 1,000-bin comparison verified unchanged motor outputs.
The records are `outputs/maze-120s-state.json`, `outputs/maze-120s-summary.json`,
and `outputs/maze-120s-diagnostics.jsonl`. The local analysis also records the
trajectory comparison in `outputs/maze-120s-analysis.json`.

## Camera rate and simulation speed

The original viewer bundled JPEGs with brain JSON and waited 150 ms between
requests. One tailnet snapshot request measured 0.92 s for 79 kB, which can
reduce displayed updates to roughly one per second. That is separate from
compute throughput: the harder maze generated about 18 frames/s on the B300.
Mean time per 20 ms simulation step was 29.4 ms for body physics, 12.4 ms for
brain computation, 6.2 ms for eyes/retina, and 7.5 ms for body render/JPEG.
Thus the simulation advanced at about 0.36× real time, even though many more
camera frames existed than the polling viewer could display.

`/api/maze/camera.mjpg` now sends those cached JPEG frames on one persistent
connection, capped at 20 frames/s. It skips stale frames under backpressure;
it never runs additional simulation or renders on an HTTP thread. Held frames
repeat periodically when paused. `/api/maze/status?body=0` omits the body JPEG
from neural polling while streaming; the browser falls back to snapshots if
the stream fails. Local API validation received about 19 frames/s and verified
that one stream advances and survives switching layouts. Network/device
conditions still limit displayed frame rate. A subsequent private Tailscale
check received 31 distinct frames at 19.1 frames/s; the in-app browser rendered
the stream successfully. Streaming improves delivery,
not simulation throughput or biological fidelity.

Browser and network buffers can still delay an already-open MJPEG stream.
Reset now disconnects that stream immediately, shows a reset indicator, and
displays the time-zero JPEG returned by the reset command. Paused and completed
trials use snapshots; resuming navigation opens a fresh stream. Late status
responses from before a command are discarded. Feeding uses snapshots paired
with the brain response. `node scripts/validate_maze_ui.mjs` checks these camera
transitions and delayed-response races without needing a running simulation.

Detailed trajectories: `outputs/maze-validation.json`. Compact browser report:
`wasm/maze/validation.json`. `scripts/validate_maze_api.py` checks live images,
continuous stepping, modality/source controls, arrival, pause/resume, invalid
requests, origins, and ownership across the three interfaces. The existing
stripe API checks are also run as regression coverage.

Displayed brain dots are anatomical annotation anchors with actual modeled
spike counts in the latest completed bin. They are not neuron morphology,
voltage, calcium, or chemical measurements. Browser updates can skip bins;
live playback need not keep biological real time. The odor colors are a
square-root contrast display of normalized concentration, not ppm.
