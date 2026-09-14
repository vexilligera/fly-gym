"""Measure actual DN-controlled behavior and interventions on the allocated GPU.

No success criterion is imposed: a stalled fly is a result, not a failed test.
Tests establish signal provenance, zero-drive controls, and policy isolation.
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.motor_readout import DESCENDING_GROUPS, descending_motor


def unit_checks():
    zero = dict.fromkeys(DESCENDING_GROUPS, 0.)
    assert descending_motor(zero)['gains'] == [0., 0.]
    forward = {**zero, 'DNp09_left':100., 'DNp09_right':100.}
    backward = {**zero, 'MDN_left':100., 'MDN_right':100.}
    np.testing.assert_allclose(descending_motor(forward)['gains'], [1, 1])
    np.testing.assert_allclose(descending_motor(backward)['gains'], [-1, -1])
    left = descending_motor({**zero, 'DNa02_left':100.})['gains']
    right = descending_motor({**zero, 'DNa02_right':100.})['gains']
    np.testing.assert_allclose(left, right[::-1])
    assert left[0] < left[1]
    np.testing.assert_allclose(descending_motor({**forward, **{k:v for k,v in backward.items() if k.startswith('MDN')}})['gains'], [0, 0])
    assert max(descending_motor({**zero,'DNp09_left':10000})['gains']) <= 1.2
    print('PASS: zero drive, opposing drive, bilateral steering and bounds', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit-only', action='store_true')
    parser.add_argument('--duration', type=float, default=120)
    parser.add_argument('--layout', choices=('complex','simple'), default='complex')
    parser.add_argument('--trial-only', action='store_true',
                        help='Run the selected layout with both senses; skip previously verified controls')
    args = parser.parse_args()
    unit_checks()
    if args.unit_only:
        return
    from brain.model import ConnectomeBrain
    from brain.maze_navigation import MazeNavigation, MazePolicy
    brain = ConnectomeBrain(backend='cuda')
    nav = MazeNavigation(brain)
    output = ROOT / ('outputs/neuron-navigation' if args.layout=='complex' else 'outputs/neuron-navigation-simple')
    output.mkdir(parents=True, exist_ok=True)
    dn_indices = np.concatenate([brain.groups[key] for key in DESCENDING_GROUPS])
    report = {'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'metadata':brain.metadata,'trials':[],
              'limits':'One heading/leg seed and fixed neural seed. No trained parameters or claim of natural navigation.'}
    cases = [('combined', {'duration':args.duration}),
             ('vision_only', {'condition':'vision_only','duration':5}),
             ('odor_only', {'condition':'odor_only','duration':5}),
             ('neither', {'condition':'neither','duration':5}),
             ('silenced', {'silence_descending':True,'duration':5})]
    if args.trial_only:
        cases=cases[:1]
    for name, arguments in cases:
        nav.reset(layout=args.layout,**arguments)
        initial_distance=nav.state['score']['distance_mm']
        minimum_distance=initial_distance
        assert nav.policy is None
        trace, total_spikes = [], 0
        dn_counts = dict.fromkeys(DESCENDING_GROUPS, 0)
        started = time.perf_counter()
        # Fail immediately if neuron mode ever invokes the sensory policy.
        with patch.object(MazePolicy,'step',side_effect=AssertionError('Policy entered in DN mode')):
            while not nav.done:
                state = nav.step(images=False)
                neural, decoder = state['brain'], state['decoder']
                minimum_distance=min(minimum_distance,state['score']['distance_mm'])
                assert decoder['gains'] == neural['gains'] == descending_motor(brain.filtered)['gains']
                assert not set(dn_indices) & set(neural['activity']['input_indices'])
                if name in ('neither','silenced'):
                    assert decoder['gains'] == [0.,0.]
                    assert not brain.last_delta[dn_indices].any()
                if name == 'neither':
                    assert neural['spikes'] == 0
                if name == 'silenced':
                    assert set(neural['silenced_indices']) == set(dn_indices)
                total_spikes += neural['spikes']
                for key in DESCENDING_GROUPS:
                    dn_counts[key] += int(brain.last_delta[brain.groups[key]].sum())
                trace.append({key:state[key] for key in ('time','position','heading_deg','decoder')}
                              | {'rates_hz':neural['rates_hz'], 'filtered_rates_hz':neural['filtered_rates_hz']})
                if nav.frames % 500 == 0:
                    print(json.dumps({'case':name,'time':state['time'],'distance_mm':state['score']['distance_mm'],
                                      'dn_spikes':dn_counts}), flush=True)
        if name in ('combined','silenced'):
            assert total_spikes > 0
        path = np.asarray(state['path'])
        result = {'name':name, 'config':state['config'], 'status':state['status'], 'time':state['time'],
                  'distance_mm':state['score']['distance_mm'],
                  'initial_distance_mm':initial_distance, 'minimum_distance_mm':minimum_distance,
                  'net_displacement_mm':float(np.linalg.norm(path[-1]-path[0])),
                  'path_length_mm':float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
                  'peak_abs_gain':float(np.max(np.abs([row['decoder']['gains'] for row in trace]))),
                  'dn_spikes':dn_counts, 'brain_spikes':total_spikes,
                  'wall_seconds':time.perf_counter()-started}
        report['trials'].append(result)
        (output / f'{name}-trace.json').write_text(json.dumps(trace)+'\n')
        (output / f'{name}-final.jpg').write_bytes(base64.b64decode(nav.world.images()['body']))
        (output / 'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(result), flush=True)

    if args.trial_only:
        report['checks']=['DN input exclusion','exact shared readout gains','policy never called','brain/body clock agreement']
        report['passed']=True
        (output / 'report.json').write_text(json.dumps(report,indent=2)+'\n')
        nav.world.close()
        print('PASS: selected-layout DN trial and signal provenance',flush=True)
        return

    # The previous policy remains an explicitly selected comparison.
    nav.reset(controller='sensory_policy',layout='simple',duration=5)
    while not nav.done:
        state=nav.step(images=False)
    assert state['status']=='reached', state['status']
    report['comparison']={'controller':'sensory_policy','layout':'simple','status':state['status'],'time':state['time']}
    # Reset after a silencing trial must restore the full brain's output.
    brain.reset()
    positive=[brain.step(stimulus='walk',rate_hz=100) for _ in range(15)]
    assert positive[-1]['gains'][0]>0 and positive[-1]['gains'][1]>0
    assert positive[-1]['silenced_indices']==[]
    report['checks']=['DN input exclusion','exact shared readout gains','policy never called',
                       'all-senses-off silence','DN silencing with sensory activity',
                       'comparison-policy simple-maze arrival','silencing clears after reset']
    report['passed']=True
    (output / 'report.json').write_text(json.dumps(report,indent=2)+'\n')
    nav.world.close()
    print('PASS: DN maze loop, sensory controls, silencing, reset and comparison',flush=True)


if __name__=='__main__':
    main()
