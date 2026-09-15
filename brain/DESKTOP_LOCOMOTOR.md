# DesktopFly-style motor circuit

Open /locomotor/ to run an isolated six-leg experiment in the browser. Select
forward, backward, left/right stimulation, or a causal control, then Run.
The worker advances neurons and mechanics independently of rendering. Reset
recreates the neural and mechanical state and clears the trail and plots.
The page pauses when hidden and stops at 10 simulated seconds.

## Source and implementation

The reference uses Denis Shiryaev's DesktopFly revision
32b00011e83c3dc85fa3ea0b3934155b04f1635d. Its JavaScript LocomotorSim,
LegDynamics and SixLegDynamics code is copied without modification under
wasm/locomotor/vendor, with the MIT license. The extracted MaleCNS circuit
and report retain their CC BY 4.0 license, full provenance and raw contacts.
wasm/locomotor/provenance.json pins every vendored file's SHA-256 digest.

The circuit has 1,045 neurons: 16 descending, 622 intermediate, 220 motor,
153 sensory and 34 ascending. It retains 17,224 directed connections and
708,689 anatomical contacts. It is not the whole CNS. Original annotation,
transmitter and omitted-input information is preserved in the vendored data.

Artificial drive enters DNp09, MDN or DNa01/DNa02. A “40 Hz” setting means
the upstream current adapter min(0.35, rate * 0.004), not a guaranteed spike
frequency. The 1 ms LIF model has input normalization, adaptation, refractory
periods, separate excitatory/inhibitory decay and a fixed subthreshold baseline.
Mean motor rates become activation rate / (rate + 50).

Antagonistic motor pools drive hip, elevation and knee forces. Joint angle,
speed and contact/load return through pooled leg sensory encoders. There is
no desired gait phase, recorded step trajectory, commanded body speed or
odor-seeking policy. The foot-support solver nevertheless makes substantial
mechanical assumptions: rigid ground projection and inferred no-slip body
motion, with no complete body dynamics or balance model. Distances, forces
and segment lengths are model units. The new 3-D body surface is illustrative;
its rendered leg endpoints match the sensed mechanical endpoints.

The neural/body feedback loop is 120 Hz, with 600 Hz mechanical substeps and
an exact 8/8/9 ms neural schedule. Neural time lags body time by less than 1 ms
and agrees every 25 ms. Display schedules at 30, 60 and 120 Hz produce
identical recorded state. The UI reports computed filtered neuron rates.

## Reference results

Run node scripts/validate_desktop_locomotor.mjs. It checks pinned source
hashes, graph structure, controls, physical endpoints, reset and clocks.
The original eight-second DNp09 reference reproduces:

- 12,868 motor spikes and 34,593 sensory spikes.
- Forward travel 10.6436566602 model units.
- Foot-contact onsets after 3 s: RF/LF/RM/LM/RH/LH = 20/22/19/11/16/14.
- Late path length 60.5536576632 model units.

The nine ten-second UI trials are in wasm/locomotor/report.json.
DNp09 produces repeated stepping on every leg; MDN gives late forward
travel −12.6561. Left/right stimulation after an identical 3 s prefix changes
late yaw by +0.5865 / −0.2111 rad relative to its baseline. The raw baseline
itself turns; no symmetry correction is added. Cutting synaptic transmission
or silencing all motor neurons gives zero propulsion and zero motor spikes.
Removing feedback changes activity and stepping, but does not abolish walking.
These are model/control checks, not validation against measured locomotion.

## MuJoCo transfer

The detailed FlyMimic asset in this checkout has muscles for the left front
leg only. A separate fixture therefore tests the existing left-front tibia
muscles, leaving the six-leg reference unchanged:

    node scripts/validate_desktop_locomotor.mjs
    .venv/bin/python scripts/run_desktop_transfer.py --videos

brain/desktop_locomotor.py ports the upstream neural model to Python/Numba.
A fixed two-second input compares voltage, adaptation, rates, currents and
refractory states with upstream JavaScript: all errors were zero on the Mac,
and total, motor and sensory spike counts matched.

The four-second fixture uses bilateral DNp09 drive at 40 Hz-equivalent.
Mean left-front tibia motor-pool rates drive the existing flexor/extensor
Hill-type actuators with fixed gain 0.2. The current fixture's physical knee
angle and velocity feed back through the upstream pooled proprioceptor
encoder, converted to flexion-positive coordinates around the 100° rest pose.
No contact or other-leg sensory signal is invented. Hip/elevation are fixed.

Mac results: 290 left-front flexor and 83 extensor spikes, maximum additional
extension 30.511° and flexion 5.181° compared with the silenced control.
Silencing motor neurons and disabling synaptic transmission yield identical
passive mechanics. Feedback-off gives a different neural response and joint
trajectory. Recorded videos play at fivefold slow motion.

An independent rerun on slurm-b300-128-021 passed the same port and physical
controls. All 1,045 per-neuron spike counts matched the Mac in each of the four
conditions. Maximum joint-angle difference was below 1.4e-12 degrees. The
comparison is published in wasm/locomotor/runtime-comparison.json.

The report, videos and raw traces are published beside the six-leg viewer.
The tests establish a causal one-joint transfer. They do not establish
six-leg MuJoCo walking, accurate motor physiology or an innate sensory policy.
Several differences from the older reflex model change together: circuit
coverage, normalized synapses, excitability, adaptation, input drive, sensory
encoding and pooled muscle coupling. The extensor response cannot be
attributed to any one of these changes from this comparison alone.

## Scope

The four claw sensors 815843, 817680, 912317 and 935383 are in this circuit,
with sensory joint and direction still null. The prior 16-assignment
experiment is unchanged. Pooled motion sensing does not establish their
biological tuning. The experiment does not connect the full female FlyWire
brain to this male subset, change the maze controller, infer sugar attraction,
or implement feeding, sleep or learning.
