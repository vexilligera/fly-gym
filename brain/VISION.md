# Visual navigation experiment

Open `/vision/` and select **Run new trial**. The fly walks toward a dark
vertical stripe. Both compound-eye images, the latest brain spike bin, body
render, decoded direction, and trajectory are live outputs of the simulation.
The existing `/connectome/` manual neural/odor experiments remain available.

The full feedback loop runs on one compute worker: MuJoCo body physics on the
CPU, EGL eye/body rendering on the allocated GPU, and the recurrent brain on
CUDA. The browser observes snapshots; it does not clock the simulation.
Pause stops at a completed 20 ms bin. A trial ends on entering a 3 mm target
zone or reaching its time limit. One shared brain is used by both pages;
manual stimulation is rejected while a visual trial is running.

## Signal path

1. Two cameras attached to the fly's eyes render the actual arena. Their
   position, orientation, 157° vertical field of view, fisheye correction, and
   721 ommatidia per eye follow NeuroMechFly's vision configuration. Camera
   orientations are baked as quaternions to preserve the source model's
   extrinsic XYZ convention in the flattened MJCF. Camera geometry supplies
   pixel viewing directions. The fly's own geometry is hidden from these eyes.
2. Each ommatidium's active color channel supplies luminance. The isolated
   dark-stripe task uses `clip((0.65 - luminance)/0.65, 0, 1)` in a fixed
   above-ground retinal mask. Contrast becomes 0–180 Hz external Poisson input
   to the release's 7,932 annotated R1–6 cells. This is a chosen contrast
   encoder, not phototransduction or a natural photoreceptor response model.
3. All 138,639 neurons and 15,091,983 signed connection rows are simulated.
   The inputs propagate through these connections; the 1,697 L2 readout cells
   are **not** directly stimulated. Their actual spike counts are smoothed
   with a 100 ms time constant. An artificial population vector estimates
   target bearing from their assigned retinal directions.
4. A hand-designed decoder converts that bearing into left/right CPG gains.
   A constant 0.9 walking drive remains in all control conditions; visual
   activity changes steering. The existing six coupled oscillators and
   recorded joint trajectories control 42 position actuators and six adhesion
   actuators. These are not reconstructed descending, VNC, muscle, or motor
   neuron dynamics.
5. Brain and body advance 20 ms, then the eyes render the new body pose. Target
   world coordinates are used for rendering and evaluation/termination only.
   They never enter the bearing estimator or motor gain calculation.

## Synthetic retinal registration

The bundled annotation table has cell types and brain-space anchor positions,
but no retinal column IDs or receptive-field directions. We do **not** treat
those anchors as retinal coordinates. Instead, L2 cells are assigned to pixels
within the annotated side using a reproducible artificial registration (seed
783). R1–6 cells inherit the pixel of their strongest released L2 connection
(6,291 of 7,932 have such a link). The others receive a deterministic cyclic
pixel assignment. Connection signs, counts, and identities are unchanged.

This allows a causal, testable visual controller. It does not recover the
animal's actual retinotopy. Replacing it with a verified column/receptive-field
registration and calibrated cell-type dynamics is a separate scientific task.

## Controls and validation

- **Eyes disconnected:** cameras still render, but visual input rates are zero.
  From reset, the unforced brain remains silent. Baseline walking remains.
- **Mapping shuffled:** a fixed permutation (seed 2026) scrambles all 1,442
  retinal channels before neural input. The decoder retains its original map.
  This preserves retinal values, not necessarily each neuron's input rate.
- **Neural steering disconnected:** eyes still drive the brain, but decoded
  direction no longer changes the left/right walking gains.

`scripts/validate_vision.py` runs six matched settings per condition: stripe /
initial heading / leg-phase seed = (−30°, 0°, 1), (30°, 0°, 1), (−35°, 15°, 3),
(35°, −15°, 3), (−20°, −10°, 5), (20°, 10°, 5). The stripe starts 18 mm from the
origin; each trial lasts at most two simulated seconds; success is distance
below 3 mm. Both models reset before each trial. Neural randomness resets to
the fixed backend stream; the configurable seed changes initial leg phases.

On the B300, vision succeeded in **6/6** settings (0.62–0.98 simulated seconds),
versus **2/6** for each of the three controls. The nearly aligned starting
headings also succeed with straight walking. Deterministic replay, finite body
state, synchronized clocks, silent disconnected inputs, and constant gains in
the appropriate controls passed. The sample is small and deliberately simple;
these are engineering checks, not estimates of natural fly behavior.

Detailed trajectories and readouts: `outputs/vision-validation.json`.
Browser summary: `wasm/vision/validation.json`.
`scripts/validate_vision_api.py` additionally checks continuous background
stepping, pause/resume, reset, image decoding, input validation, origin checks,
and ownership between the visual and manual interfaces. Existing full-release
index/coordinate/activity and manual-intervention checks also pass.

## Scientific limits

- Real photoreceptors use graded voltage and histamine release. The model
  uses uniform LIF dynamics and preserves the released signs, including many
  excitatory R1–6 outputs. This is **not a validated photoreceptor/lamina model**.
- The motor readout is an engineered L2 decoder. The experiment does not
  establish that natural descending visual navigation emerges from the full
  connectome, or that the rest of the brain is necessary for this behavior.
- No receptor binding, chemical concentrations, drug effects, detailed ion
  channels, adaptation, plasticity, learning, memory, goals, or planning are
  simulated. Synaptic sign/weight and a common delay/decay approximate chemical
  transmission; the fly does not contain a molecular chemistry simulation.
- There is one high-contrast stripe on a light floor, with no competing targets
  or obstacle-avoidance task. Some viewpoints produce weak or absent visual
  responses. Broader visual behavior has not been validated.
- Displayed dots are anatomical annotation anchors; colors show modeled spike
  counts in the latest 20 ms bin. They are not morphology, voltage, calcium,
  or chemical measurements. Browser snapshots can skip bins. Eye images are
  stamped at the beginning of the step; brain/body state is stamped at its end.
- Playback speed includes compute-node physics and rendering, but excludes
  network/display delay. Live updates do not guarantee biological real time.

Implementation: `brain/visual_world.py`, `brain/vision_input.py`,
`brain/navigation.py`, `brain/service.py`, and `wasm/vision/`.
