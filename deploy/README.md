# B300 Slurm deployment

Private URL: **https://cw-login-zny.alpaca-elnath.ts.net:8443/connectome/**

Visual navigation: **https://cw-login-zny.alpaca-elnath.ts.net:8443/vision/**

Vision + olfaction maze: **https://cw-login-zny.alpaca-elnath.ts.net:8443/maze/**

The current deployment is Slurm job **5807547**, on `slurm-b300-128-021`, with
one NVIDIA B300, 8 CPUs, and 24 GiB host memory. It expires at
**2026-09-14 15:26:05 UTC / 2026-09-15 00:26:05 JST**, or earlier if canceled. The `low`
QoS is preemptible. This is a Slurm allocation, not a permanent hosted service.

All files are under `/mnt/home/zny/flygym` on `crwv` and the shared compute
filesystem. The CUDA brain and HTTP server execute on the allocated compute
node. In `/connectome/`, MuJoCo body physics executes in the browser. In
`/vision/` and `/maze/`, MuJoCo physics, eye rendering, neural stepping, and steering all run
on the compute node; the browser observes snapshots and draws the brain anchors.

```text
Tailnet browser (private HTTPS :8443)
  → existing cw-login-zny Tailscale identity on slurm-login-0
  → loopback SSH tunnel :18080, managed by deploy/gateway.py
  → loopback HTTP :8000 on the allocated B300
  → all 138,639 neurons / 15,091,983 connection rows on CUDA
```

Tailscale Serve uses the existing `/mnt/home/zny/.tailscale` identity and
`/mnt/home/zny/tailscale` binaries. No public Funnel is enabled. The backend
binds only to `127.0.0.1`; its browser-origin allowlist explicitly includes the
Tailscale HTTPS origin. The gateway follows this job's compute-node changes
after a Slurm requeue, and exits when the allocation ends. A requeue resets the
brain state; use **Reset both** after reconnecting. The gateway tolerates the
empty node field while a requeued job is pending. The 120-second maze update
restarted this job on 2026-09-13 at 15:26:05 UTC; the expiry above reflects that restart.

## Operations (on the login node)

Code is synchronized through `https://github.com/vexilligera/fly-gym`, branch
`codex/vision-olfaction-maze`. On the Mac this remote is named `workspace`
(official FlyGym remains `origin`); on the cluster it is `origin`. Changes are
committed and pushed on the Mac, then pulled on the shared cluster checkout:

```sh
# Mac
git -c http.version=HTTP/1.1 -c http.postBuffer=52428800 push workspace codex/vision-olfaction-maze
# Login node: fetch code; existing environment, datasets, and assets stay local.
cd /mnt/home/zny/flygym
git pull --ff-only
```

Restart the running server after backend changes with `scontrol requeue 5807547`
only after validation. This resets the live experiment and temporarily takes
the endpoint offline while the job is pending. The gateway follows the new
node. Runtime state, downloaded connectome data, generated meshes, environments,
logs, and the private EGL loader are excluded from Git; a fresh clone still
needs the installation/data/asset steps described below.

```sh
cd /mnt/home/zny/flygym
squeue -j 5807547
tail -f outputs/slurm-5807547.log
curl http://127.0.0.1:18080/api/brain/status
```

The current job ID, compute node, and gateway PID are recorded in
`outputs/job-id`, `outputs/compute-node`, and `outputs/gateway.pid`.

To start a new allocation after this one has ended:

```sh
cd /mnt/home/zny/flygym
sbatch --parsable deploy/slurm.sbatch > outputs/job-id
nohup .venv/bin/python deploy/gateway.py > outputs/gateway.log 2>&1 &
echo $! > outputs/gateway.pid
```

The checked deployment marker `deploy/ready` must exist before the batch job
starts the server. The script waits at most 30 minutes for preparation.
Do not start a second job/gateway while the current one is active.

To stop this allocation:

```sh
scancel 5807547
/mnt/home/zny/tailscale/tailscale --socket=/mnt/home/zny/.tailscale/tailscaled.sock serve --https=8443 off
```

This removes only this service's Tailscale listener, not other Tailscale state.
The gateway exits when it sees the job end.

## Installation and verification

The isolated server environment uses Python 3.12.13. It needs the dependencies
in `brain/requirements-vision-b300-lock.txt`; the Mac's broader FlyGym lock is separate.
The browser assets were generated locally and copied along with the exact
connectome and annotation files recorded in `brain/provenance.json`.

```sh
/mnt/home/zny/.local/bin/uv pip sync --python .venv/bin/python brain/requirements-vision-b300-lock.txt
srun --jobid=5807547 --overlap --ntasks=1 --cpus-per-task=8 .venv/bin/python scripts/validate_cuda.py
srun --jobid=5807547 --overlap --ntasks=1 --cpus-per-task=8 .venv/bin/python scripts/validate_connectome.py
```

Pause the browser before running the full-connectome checks: they reset and
advance the one shared brain. Check outputs are `outputs/cuda-validation.json`
and `brain/validation.json` on the cluster.

The CUDA implementation uses float64 exact linear LIF updates and Brian2's
threshold/refractory/delay ordering. It retains all signed connection rows;
synapse counts accumulate as integers before conversion to postsynaptic input.
The input RNG differs from Brian2, so equal seed values do not promise identical
whole-brain spike trains. Identical explicit inputs on the recurrent test
network matched spike counts exactly, with voltage errors below 10⁻⁹ mV.
Full-release index/coordinate/activity checks and intervention checks passed.

Initial full-network checks measured approximately **3.5× real time for odor
input and 6.4× for direct walking input** in brain computation. This excludes
network transit and browser physics. The current browser exchanges one 20 ms
window per request, so playback over a long-distance tailnet can be limited by
round-trip latency even when the brain itself runs faster than real time.

## Visual runtime

See `brain/VISION.md` for the signal path and limits. The visual experiment adds
the official NeuroMechFly eye geometry and 721 ommatidia per eye, an explicitly
synthetic retinal registration, R1–6 input to the full network, and an engineered
L2-spike-to-CPG steering decoder. All released synapses remain intact. It does
not model molecular phototransduction, natural retinotopy, or a reconstructed VNC.

The compute image contains the NVIDIA EGL driver and GL dispatch library but
lacks the GLVND EGL loader. We extracted the official Ubuntu package privately,
without changing system packages:

```sh
mkdir -p deploy/egl
curl -fL https://archive.ubuntu.com/ubuntu/pool/main/libg/libglvnd/libegl1_1.4.0-1_amd64.deb -o deploy/egl/libegl1_1.4.0-1_amd64.deb
cd deploy/egl
dpkg-deb -x libegl1_1.4.0-1_amd64.deb .
```

Package SHA-256:
`4a35c0e925e15a076e7ce11d7c76f8ecb16615fc347c2c09afa4c205a7ca8ec4`.
The Linux CUDA server automatically adds this project-local directory to the
dynamic linker path (one interpreter re-exec), selects `MUJOCO_GL=egl`, and
defaults to four Numba threads. MuJoCo opens the first EGL device it can
initialize inside the allocation. Do not run it on the login node.

To reproduce visual checks in this allocation:

```sh
cd /mnt/home/zny/flygym
srun --jobid=5807547 --overlap --ntasks=1 --cpus-per-task=4 env MUJOCO_GL=egl NUMBA_NUM_THREADS=4 LD_LIBRARY_PATH="$PWD/deploy/egl/usr/lib/x86_64-linux-gnu" .venv/bin/python scripts/validate_vision.py
```

The matched 24-trial check passed: 6/6 arrivals with vision, 2/6 for each of
disconnected eyes, shuffled registration, and disconnected neural steering.
Two almost-aligned starts also succeed with straight walking. These small
controlled experiments validate this engineered loop, not natural fly vision.
`outputs/vision-validation.json` contains trajectories and per-bin readouts;
`wasm/vision/validation.json` contains the public summary. API pause/reset,
continuous stepping, origin guards, image decoding, and manual-mode regression
checks passed as well. Visual trials stop automatically; closing the browser
does not pause the compute loop before the trial's time limit.

## Multisensory maze

See `brain/MAZE.md` for the exact signal path, chemical-model limits, and
measured sensory controls. `/maze/` offers a branching 25-cell maze and the
original simple baffles, with central sugar,
a pre-equilibrated 2-D food-odor field, and two local antenna samples. Both
visual and olfactory input enter the same full-brain time interval. The
movement decoder is engineered and receives neural activity, without a maze
map or target coordinates. The brain, stripe, and maze pages share one worker.

```sh
srun --jobid=5807547 --overlap --ntasks=1 --cpus-per-task=4 env MUJOCO_GL=egl NUMBA_NUM_THREADS=4 LD_LIBRARY_PATH="$PWD/deploy/egl/usr/lib/x86_64-linux-gnu" .venv/bin/python scripts/validate_maze.py
srun --jobid=5807547 --overlap --ntasks=1 --cpus-per-task=1 .venv/bin/python scripts/validate_maze_api.py --port 8000
```

The standalone behavioral check constructs its own brain/world. The API check
controls the specified server and resets its shared simulation. Use a staging
server when preserving a running experiment matters. Results are recorded in
`outputs/maze-validation.json` and `outputs/maze-api-validation.json`; the small
browser report is versioned in `wasm/maze/validation.json`.

## Sugar-taste assay

After a maze arrival, `/maze/` offers **Watch sugar response** and a no-taste
control. The 21 released sugar GRNs drive the full network; the two MN9 cells
are readouts only. This is a reset, held-body neural assay with a separate
0–8 s clock, played at 0.2× speed. It does not simulate mouth mechanics or
ingestion. See `brain/SUGAR_AND_SLEEP.md` for wiring and limitations.

`scripts/validate_sugar.py` checks the pinned input cohort, unchanged legacy
CUDA inputs, stimulus timing, a silent control, and downstream responses at
50/100/200 Hz. It needs the pinned `external/fly-brain/code/benchmark.py` as
well as the existing data. `scripts/validate_sugar_api.py --port 8001` checks
HTTP controls against a staging server, including arrival gating, pause/resume,
body/clock separation, ownership, and reset. Reports are saved in `outputs/`.
