# Full-connectome brain connected to NeuroMechFly

The local `/connectome/` page runs **all 138,639 neurons and all 15,091,983
weighted connection rows** in Eon's released FlyWire v783 model. The connection
rows sum to **54,492,922 anatomical synapse counts**. Connections are not
thresholded or reduced to a subgraph. All 14 neurons without a matching public
annotation are still simulated.

Brian2 runs the recurrent brain on the Mac CPU. The browser runs the existing
MuJoCo fly. Brain and body advance in matching 20 ms batches. This is a real
simulation of the released network; it does not replay recorded spike trains.

The optional CUDA backend runs the same LIF equations on a B300 Slurm compute
node. See [`../deploy/README.md`](../deploy/README.md) for the private Tailscale
URL, allocation lifetime, validation, and numerical differences. The UI reports
the active backend, GPU, hostname, and job ID. The new `/vision/` experiment
runs compound-eye rendering, full-brain dynamics, and body physics on the
compute node. See [VISION.md](VISION.md) for its synthetic retinal registration,
engineered L2 steering decoder, matched controls, and scientific limitations.

## Run

From the repository root, run `./start-local.sh` and open
http://127.0.0.1:8000/connectome/ . Initialization takes around 10–20 seconds
on this Mac after dependencies are installed. The first compilation can take
longer. The brain advances only while **Run brain & body** is active.
Pause before closing the page. The server retains one shared brain state;
use one active connectome page at a time.

Choose a stimulation preset, then Run:

- **DNp09:** stimulate both annotated walking-related descending neurons.
- **DNp09 + DNa02:** add left or right steering-related neuron stimulation.
- **MDN:** stimulate the four moonwalker descending neurons.
- **ORN_DM1:** stimulate 35 left and 33 right annotated olfactory neurons.
- **None:** no external drive; the model is silent when initialized at rest.
- **Silence DNp09:** suppress these neurons' output spikes, an artificial
  intervention useful for checking the connection to the movement decoder.

**Reset both** between independent experiments. Turning off an input does not
instantly erase recurrent activity or the decoder's 100 ms smoothing history.

## Live spatial activity

The brain panel plots all **138,625 matched annotation anchors** in 3D. Drag to
rotate, scroll to zoom, and click a point or a responding-neuron button to
inspect its FlyWire ID, cell type, transmitter annotation, and current spike
count. **All anchors** toggles the faint anatomical context; it does not remove
neurons from the simulation. **Reset view** restores the initial camera.

- Teal points are firing neurons without direct external input in that bin.
- Amber points are firing neurons receiving direct external input. They can
  also receive recurrent input, so the color does not attribute each spike's
  cause exclusively to stimulation.
- Larger points mean more spikes in the completed **20 ms simulation bin**.
  Every firing neuron with a coordinate is shown, with no activity downsampling
  or invented transitions. Pause holds the last completed bin; reset clears it.
- Choose **Odor channel · ORN_DM1** to see a distributed response from the
  sensory input, or DNp09 to see the neural drive coupled to walking.

These are live outputs of the running LIF model, not measurements from a living
fly. They show spikes, not membrane voltage, calcium imaging, transmitter
concentration, or individual spike timestamps within the bin. The simulation
speed is displayed relative to wall time and includes the neural request and
body advancement; **live updating does not imply biological real-time speed**.
Brian and MuJoCo clocks must match before a frame is displayed.

Coordinates come from the pinned FlyWire annotation file's `pos_x/y/z` fields:
annotation anchors, typically on a neuron's backbone, not complete morphologies
or soma positions. The source voxel dimensions are **4 × 4 × 40 nm**, converted
to micrometers with correct anisotropic scaling. The default view uses X to the
right, −Y up, and −Z depth. All 14 unlocated neurons still simulate and contribute
to totals; their active count is reported separately when nonzero. The renderer
does not draw the 15 million simulated connection edges.

`GET /api/brain/geometry` returns immutable parallel `indices` and flattened
`positions_um` arrays. Indices always refer to the released completeness table
and the same Brian2 neuron ordering. Each `/api/brain/step` response contains
lossless sparse `activity.indices` / `spike_counts` arrays for the completed bin
plus `input_indices`. `GET /api/brain/neuron/<index>` resolves an index to the
exact string FlyWire root ID and annotations; IDs never pass through JavaScript
floating-point numbers. The stream stores no unbounded per-spike history.

## Where data ends and modeling begins

1. **Measured/reconstructed:** the released neuron identities, connection
   counts, and public cell-type annotations. The release's signed connection
   weights incorporate neurotransmitter-based excitation/inhibition assumptions.
2. **Neural dynamics:** the Shiu/Eon leaky integrate-and-fire equations in
   Brian2, with uniform neuron parameters: rest/reset −52 mV, threshold −45 mV,
   membrane time constant 20 ms, synaptic decay 5 ms, refractory period 2.2 ms,
   synaptic delay 1.8 ms, and 0.275 mV per signed synapse count. Timestep 0.1 ms.
   Artificially stimulated neurons have zero refractory time, following the
   released activation protocol. Poisson input events add 68.75 mV.
3. **Local adapter:** 100 ms filtered descending-neuron firing rates become
   two CPG gains. This is hand-designed, not recovered from a VNC connectome:

   ```text
   forward = mean(DNp09 left, right) / 100 Hz
   reverse = mean(MDN left, right) / 100 Hz
   turn = (DNa02 left − right) / 100 Hz
   gain_left  = clip(forward − reverse − 0.6 × turn, −1.2, 1.2)
   gain_right = clip(forward − reverse + 0.6 × turn, −1.2, 1.2)
   ```

4. **Body controller:** the upstream six-oscillator CPG and recorded leg-step
   trajectories drive 42 position actuators plus six adhesion commands.
   MuJoCo calculates joint motion, gravity, inertia, and contacts.

Stimulating DNp09 directly demonstrates a causal neural-to-body interface; it
does not demonstrate an autonomous brain decision to walk. The full recurrent
network also responds, and its measured readout changes the adapter's output.

## Chemicals

An odor molecule in a real fly activates sensory receptors; sensory neurons
signal into antennal-lobe and downstream circuits. Here an **illustrative
encoding** maps normalized odor exposure 0–1 to 0–150 input events/s in the
annotated ORN_DM1 cells. The resulting electrical activity travels through the
actual network. The control does not specify a chemical identity, molar
concentration, receptor affinity, or a spatial odor plume. It does not guarantee
attraction, aversion, or locomotion.

Neurotransmitter effects are reduced to the release's signed synaptic weights.
There are no neurotransmitter molecules, vesicles, receptor-binding kinetics,
reuptake, drug doses, pharmacokinetics, receptor subtypes, or slow modulation.
Changing a weight or stimulation rate is not a validated chemical treatment.

## Limitations

- This is the **released brain model**, not an entire central nervous system:
  the VNC, peripheral circuits, and detailed muscles are not reconstructed.
- Anatomical connectivity is not a complete description of neuronal function.
  All neurons use the same simple dynamics; morphology and detailed ion channels
  are absent. Cell types and transmitter labels include predictions and errors.
- No learning, synaptic plasticity, memory training, flight, or molecular biology.
- The manual `/connectome/` experiment has externally specified stimuli. Vision,
  proprioception, and body contacts are not fed back into the brain.
- The neural/body interface is sampled every 20 ms and uses chosen scaling and
  smoothing. Motion is not validated against biological walking data for these
  interventions. It is not a claim of consciousness or a complete living fly.

## Sources and provenance

- Eon: https://github.com/eonsystemspbc/fly-brain
  commit `a3db62f9436074e485c0278290c2164ed6150808`.
- Shiu et al., *A Drosophila computational brain model reveals sensorimotor
  processing*, Nature (2024): https://doi.org/10.1038/s41586-024-07763-9 .
- FlyWire annotations: https://github.com/flyconnectome/flywire_annotations
  commit `8587524c1748ce5ef2080822a2fc890fc03bf597`.
- Brain equations follow the upstream Brian2 materials. Their MIT notice is
  retained in `SHIU_LICENSE`. The Eon repository retains its own GPL license and
  notices; FlyWire data and annotations retain their source terms.
- `provenance.json` and `validation.json` record file checksums, exact counts,
  neuron mappings, and the intervention test results.

## Reproduce this installation

Use Python 3.14 in an isolated virtual environment, then install the exact
working dependencies using `requirements-lock.txt` and install FlyGym editable:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r brain/requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
git clone https://github.com/eonsystemspbc/fly-brain.git external/fly-brain
git -C external/fly-brain checkout a3db62f9436074e485c0278290c2164ed6150808
mkdir -p brain/data
curl -L https://raw.githubusercontent.com/flyconnectome/flywire_annotations/8587524c1748ce5ef2080822a2fc890fc03bf597/supplemental_files/Supplemental_file1_neuron_annotations.tsv -o brain/data/neuron_annotations.tsv
.venv/bin/python scripts/dev/properdocs_hooks.py
./start-local.sh
```

With the browser experiment paused, run:

```sh
.venv/bin/python scripts/validate_connectome.py
```

This checks every connection's pre/post FlyWire ID against its index, exact
network counts, silence at rest, downstream recruitment from DNp09 input,
suppression with DNp09 silenced, propagation from ORN input, and reverse drive
from MDN stimulation. It also checks all 138,625 anchor mappings and coordinate
conversions, lossless sparse activity counts, and their agreement with reported
neuron identities and descending rates. These are software/integration checks, not biological
validation of the motor adapter.
