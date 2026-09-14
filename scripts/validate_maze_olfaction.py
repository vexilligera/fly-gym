"""Isolate the simple-maze odor pathway with source and input disconnections.

Run on the allocated GPU. All cases reset brain/body, disable vision, and use
descending readouts. Report actual ORN and downstream PN spikes in every bin.
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.model import ConnectomeBrain
from brain.maze_navigation import MazeNavigation
from brain.motor_readout import DESCENDING_GROUPS


def main():
    brain = ConnectomeBrain(backend='cuda')
    nav = MazeNavigation(brain)
    output = ROOT / 'outputs/simple-maze-olfaction'
    output.mkdir(parents=True, exist_ok=True)
    groups = {key:brain.groups[key] for key in DESCENDING_GROUPS}
    for side in ('left','right'):
        groups['ORN_DM1_'+side] = brain.groups['ORN_DM1_'+side]
        groups['DM1_lPN_'+side] = brain.odor_projection[side]
    orn_indices = set(np.concatenate([groups['ORN_DM1_'+side] for side in ('left','right')]))
    report = {
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'group_sizes':{key:len(indices) for key,indices in groups.items()},
        'group_ids':{key:[brain.ids[i] for i in indices] for key,indices in groups.items()},
        'trials':[],
        'limits':'Food-associated volatile proxy, not sucrose vapor or a calibrated receptor model. One start/seed. Zero spontaneous input; PN firing is simulated, not a biological recording.',
    }
    for name, condition, source in [('source_off','odor_only',False),
                                    ('input_disconnected','neither',True),
                                    ('odor_on','odor_only',True)]:
        nav.reset(layout='simple',condition=condition,food_odor=source,
                  controller='descending',heading_deg=75,seed=1,duration=5)
        trace = []
        totals = dict.fromkeys(groups,0)
        total_spikes = 0
        while not nav.done:
            state = nav.step(images=False)
            neural = state['brain']
            assert set(neural['activity']['input_indices']) <= orn_indices
            assert abs(state['time']-neural['time']) < 1e-7
            counts = {key:int(brain.last_delta[indices].sum()) for key,indices in groups.items()}
            for key,count in counts.items(): totals[key] += count
            total_spikes += neural['spikes']
            trace.append({'time':neural['time'],'odor':state['odor'],
                          'input_rates_hz':neural['olfaction']['input_rates_hz'],
                          'rates_hz':{key:count/len(groups[key])/.02 for key,count in counts.items()},
                          'group_spikes':counts, 'brain_spikes':neural['spikes'],
                          'position':state['position'],'gains':state['decoder']['gains']})
            if name != 'odor_on':
                assert neural['olfaction']['input_rates_hz']==[0.,0.]
                assert neural['spikes']==0 and state['decoder']['gains']==[0.,0.]
            if not source:
                assert state['odor']==[0.,0.]
        if source:
            assert min(trace[0]['odor'])>0
        if name=='odor_on':
            assert all(totals[key]>0 for key in groups if key.startswith(('ORN','DM1')))
        result = {
            'name':name,'config':state['config'],'duration_s':neural['time'],
            'initial_concentration':trace[0]['odor'],
            'mean_concentration':np.mean([frame['odor'] for frame in trace],axis=0).tolist(),
            'mean_input_rates_hz':np.mean([frame['input_rates_hz'] for frame in trace],axis=0).tolist(),
            'mean_rates_hz':{key:count/len(groups[key])/neural['time'] for key,count in totals.items()},
            'group_spikes':totals,'brain_spikes':total_spikes,
        }
        report['trials'].append(result)
        (output / f'{name}-trace.json').write_text(json.dumps(trace)+'\n')
        print(json.dumps(result),flush=True)
    report['passed'] = True
    report['checks'] = ['vision disconnected in all cases','source off removes local odor and all spikes',
                        'odor present with neural input disconnected gives no spikes',
                        'only ORNs receive direct input','ORN and downstream PN response with odor on',
                        'brain/body clocks agree']
    (output / 'report.json').write_text(json.dumps(report,indent=2)+'\n')
    nav.world.close()
    print('PASS: isolated simple-maze olfactory response and both negative controls',flush=True)


if __name__=='__main__':
    main()
