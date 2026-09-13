"""Matched closed-loop visual trials on the full connectome and MuJoCo body.

Run on the allocated compute node with MUJOCO_GL=egl. Writes the complete
trajectory/decoder record plus a small browser-readable validation summary.
This validates this engineered controller, not the biological retinal mapping.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.model import ConnectomeBrain
from brain.navigation import VisualNavigation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='cuda', choices=['cuda', 'brian2'])
    args = parser.parse_args()
    started = time.perf_counter()
    brain = ConnectomeBrain(backend=args.backend)
    nav = VisualNavigation(brain)
    assert not set(brain.visual.readout).intersection(brain.visual.receptors)
    assert nav.world.bearings[0].mean() > 0 > nav.world.bearings[1].mean(), 'Eye cameras must face opposite sides'
    records = []
    settings = [(-30, 0, 1), (30, 0, 1), (-35, 15, 3), (35, -15, 3), (-20, -10, 5), (20, 10, 5)]
    for condition in ['vision', 'blind', 'shuffled', 'readout_off']:
        for target, heading, seed in settings:
            nav.reset(condition=condition, target_deg=target, heading_deg=heading, seed=seed, duration=2)
            frames = []
            while not nav.done:
                state = nav.step(images=False)
                assert abs(state['time'] - state['brain']['time']) < 1e-7
                assert np.isfinite(state['position']).all()
                if condition == 'blind':
                    assert state['brain']['spikes'] == 0, 'Disconnected eyes must leave the unforced brain at rest'
                if condition in ('blind', 'readout_off'):
                    assert state['decoder']['gains'] == [.9, .9]
                frames.append({key: state[key] for key in ['time', 'position', 'heading_deg', 'score', 'decoder']})
            record = dict(config=state['config'], reached=state['status'] == 'reached',
                          time=state['time'], final_distance_mm=state['score']['distance_mm'],
                          minimum_distance_mm=min(f['score']['distance_mm'] for f in frames),
                          L2_spikes=sum(f['decoder']['L2_spikes'] for f in frames), frames=frames)
            records.append(record)
            print(json.dumps({k:v for k,v in record.items() if k != 'frames'}), flush=True)
    # Replay a matched trial to verify deterministic reset of both systems.
    reference = records[0]
    nav.reset(**reference['config'])
    while not nav.done:
        replay = nav.step(images=False)
    np.testing.assert_allclose(replay['position'], reference['frames'][-1]['position'], atol=1e-9)
    results = {condition: sum(r['reached'] for r in records if r['config']['condition'] == condition)
               for condition in ['vision', 'blind', 'shuffled', 'readout_off']}
    passed = results['vision'] >= 5 and results['vision'] > max(results['blind'], results['shuffled'], results['readout_off'])
    report = {'schema': 'vision-validation-v1', 'hostname': platform.node(), 'metadata': brain.metadata,
              'passed': passed, 'trials_per_condition': len(settings), 'successes': results,
              'wall_seconds': time.perf_counter() - started, 'records': records,
              'scope': '2 simulated seconds/trial; stripe 18 mm from origin; success within 3 mm. Six matched starts. Synthetic registration and engineered L2 decoder only.'}
    (ROOT / 'outputs/vision-validation.json').write_text(json.dumps(report, indent=2))
    public = {k: report[k] for k in ['schema', 'passed', 'trials_per_condition', 'successes', 'scope']}
    public['summary'] = (f"Deployment check: stripe reached in {results['vision']}/6 visual trials, "
                         f"{results['blind']}/6 with eyes disconnected, {results['shuffled']}/6 with shuffled mapping, "
                         f"and {results['readout_off']}/6 with neural steering disconnected. "
                         'Six matched starts, 2 simulated seconds each. This tests the engineered loop, not biological fidelity.')
    (ROOT / 'wasm/vision/validation.json').write_text(json.dumps(public, indent=2))
    print(public['summary'], flush=True)
    nav.world.close()
    if not passed:
        raise SystemExit('Visual controller did not outperform the controls across the matched trials')


if __name__ == '__main__':
    main()
