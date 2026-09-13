# NeuroMechFly on this Mac

The default page now includes the **full released FlyWire v783 brain model**,
wired through an explicit descending-neuron adapter to the walking controller.
See [brain/README.md](brain/README.md) for exact coverage, chemicals, limitations,
source versions, and reproduction instructions. The original CPG-only lab
described below remains available separately at `/lab/`.

The `/vision/` page now closes the visual feedback loop on the compute server.
See [brain/VISION.md](brain/VISION.md) for compound eyes, synthetic retinal
registration, the engineered steering decoder, control trials, and limitations.
Run a new trial there to watch the eye images, neural spikes, body, and path.

The official FlyGym repository is installed in `.venv` (FlyGym 2.1.0,
MuJoCo 3.9.0, Python 3.14). Both official browser apps have locally generated
model assets and local copies of MuJoCo WebAssembly and Three.js.

## Run

```sh
cd /Users/nyz/Desktop/flygym
./start-local.sh
```

Open **http://127.0.0.1:8000/**. Keep the terminal running; Ctrl+C stops the
server. Use `./start-local.sh --port 8001` if port 8000 is already occupied.
The server listens on loopback only and serves the `wasm` directory, not your
source repository or Python environment. Once installed, the simulation does
not need an Internet connection. Documentation links still point online.

- `/connectome/`: full 138,639-neuron LIF network with synchronized brain/body stepping.
- `/lab/`: original CPG-only inspector, with a real-time fly simulation.
- `/game/game.html`: official slalom game, including the original keyboard controls.
- `/viewer/viewer.html`: official joint/force viewer; drag to orbit, scroll to zoom.

The native desktop viewer is also available:

```sh
.venv/bin/python scripts/launch_interactive_viewer.py
```

## What to try

1. Click **Show me the sequence** for walk → left turn → right turn → stop.
   It takes about 22 seconds at 0.1× simulation speed and pauses at the end.
2. Watch the left/right drive, six leg-cycle dials, and measured knee angle.
   Resume, then use Walk / Turn / Stop or W/A/S/D/Q to experiment.
3. Switch to **Two tripods** and alternate the two step buttons.
4. Switch to **Six legs** and coordinate the individual steps yourself.

The lab shows actual controller state and MuJoCo joint positions. Its six dials
show phase, with shading from oscillator amplitude. Grip/release describes the
adhesion actuator command; it is not a foot-contact sensor reading. Contacts
counts all collision contacts, not the number of feet touching the ground.
The chart uses radians and simulated seconds, retaining 250 ms of history.

## What “brain” means in the original `/lab/` page

Real flies use sensory processing in the brain and ventral nerve cord (VNC),
descending/ascending pathways, local motor circuits, and sensory feedback.
This demo supplies only an abstract locomotion controller:

`your command → left/right drive → six coupled oscillators → recorded step trajectories → position actuators → MuJoCo body motion`

It does **not** simulate a whole-brain connectome, individual spiking neurons,
visual decisions, odor navigation, or sensory reflex corrections. The browser
game controller does not feed contact forces back into its CPG. MuJoCo handles
contact mechanics and position-actuator tracking error. FlyGym's separate
Python `HybridController` implements stumbling and retraction corrections;
see the official tutorials for neural-controller research beyond this demo.

The oscillator model is a functional abstraction of local coordination, not
evidence that the fly brain consists of these six oscillators.

## Rebuild assets

```sh
MPLCONFIGDIR=/tmp/flygym-mpl .venv/bin/python scripts/dev/properdocs_hooks.py
```

The hook downloads pinned browser dependencies if missing and generates missing
assets. To explicitly regenerate a model, run the corresponding
`scripts/dev/build_wasm_game_assets.py` or `build_wasm_viewer_assets.py`.

The optional lab bridge is enabled only by `game.html?inspect`. The original
game remains available without that parameter. The bridge sends telemetry only
to its same-origin parent and accepts commands only from that parent.

Upstream: https://github.com/NeLy-EPFL/flygym (commit `38c8ec6`).
Documentation: https://neuromechfly.org/ . License: Apache-2.0; see LICENSE.
