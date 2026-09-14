# Assignment sweep results

The [fixed design](ASSIGNMENT_SWEEP.md) ran to completion on B300 node
`slurm-b300-128-021`, allocation 5807547: 1,056 connected 1.6-second trials and
ten reusable passive controls. Six CPU workers completed the simulation batch
in 874.18 seconds. Seeds 1–3 were tested for every case. All 16 mappings remain
hypotheses; no matched biological downstream target was fitted.

## Main observations

- At nominal gain, the largest pairwise difference in mean post-release angle
  traces was 0.387234 degrees RMS. The largest pooled flexor/extensor rate
  difference was 1.942469 Hz RMS. None of the 120 pairs exceeded the preselected
  0.5-degree / 5-Hz comparison resolutions on a primary test.
- The same pairwise criterion separated zero pairs in the matched +/-30-degree
  tests at 75, 150 or 300 Hz maximum sensory rate, or with either +/-10-degree
  midpoint shift. This is resolution-dependent closeness, not proof that the
  mappings are identical. RMS averages the full post-release interval; brief
  peaks can be larger and finer, time-resolved measurements could distinguish
  some predictions.
- Primary perturbations produced 772 flexor motor spikes across 384 trials,
  but zero extensor spikes. Across all 1,056 connected perturbation/sham trials,
  there were 2,315 flexor spikes and still zero extensor spikes. Changing these
  sensory labels and the tested gains/thresholds did not recruit the extensor
  pathway in this provisional reduced circuit.
- At higher gain some shams produce motor activity. The double contrast removes
  the matching sham response from the circuit contribution to movement. The
  largest peak absolute mean angle contrast across all cases was 1.94823 degrees;
  that is a peak circuit effect, not a pairwise RMS difference or accuracy score.

Swapping only neuron 815843's label had the largest influence on movement:
the mean of the maximum angle RMS differences across its eight single-label
pairs was 0.280 degrees, versus 0.069–0.082 degrees for the other three cells.
This identifies a sensitive model input, not its true flexion/extension label.

## More informative downstream measurements

The strongest candidate-dependent intrinsic readout in this search was
**904707, IN13A006**, during 50-degree extension. Its mean post-onset response
minus sham spans 22.05 Hz across mappings; typical within-mapping seed SD is
2.04 Hz. These are model spike-count rates over 1.3 seconds, not recorded
physiology or a verified match to an experimental cell.

Other suggested readouts include 804551 (IN01A025), 905496 (IN21A006),
800802 (IN21A004), and 930012 (IN21A003). The viewer shows their predictions
for all 16 assignments. This suggests measuring particular downstream neurons
could constrain the mapping better than using leg motion or pooled motor
activity alone. The suggestion is conditional on this circuit and its assumed
parameters, and needs anatomical matching and real recordings.

The absent extensor recruitment is a model limitation to investigate before
using a desired balanced reflex to select a mapping. It does not establish
that the biological pathway is absent: omitted inputs, receptor/sign choices,
neural gains and thresholds remain uncertain. No mapping was promoted to the
original reflex or the sugar maze.

## Verification and artifacts

The vector-input LIF adapter reproduced every tested original voltage, current,
refractory state, delayed-event queue, spike count and activation under connected,
disconnected, sensory-off and motor-silenced conditions. The original physical
default reproduced all body-ID counts and the published joint trajectory.
Silencing controls had matched passive mechanics. All 1,066 trial hashes,
configuration identities, clocks and finite traces passed the sweep checks.

Analysis checks cover identical signals, a known one-degree difference, seed
consistency, all raw-file hashes, seven directly recomputed physical contrasts,
and all 256 display traces. Imposed sensory inputs are excluded from proposed
downstream readouts. A separate mixed high-gain trial (FEFE, extension 30 degrees,
seed 2) reproduced all body-ID counts locally and on B300; maximum angle
difference was 9.1e-13 degrees (tolerance 1e-10).

Full-precision trials are retained on the cluster in
`/mnt/home/zny/flygym/outputs/assignment-sweep/` and locally in
`outputs/assignment-sweep-cluster/`. Published summaries, provenance, checks,
display traces and figures are in `wasm/calibration/assignment-*`.

After running the design's commands, generate and check summaries with:

```sh
.venv/bin/python scripts/analyze_assignment_sweep.py
.venv/bin/python scripts/check_assignment_analysis.py
```
