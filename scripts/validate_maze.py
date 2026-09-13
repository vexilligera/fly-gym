"""Measured maze trials and input ablations; run on the allocated GPU."""
from pathlib import Path
import base64
import json
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.model import ConnectomeBrain
from brain.maze_navigation import MazeNavigation

started = time.time()
brain = ConnectomeBrain(backend='cuda')
nav = MazeNavigation(brain)
# A body driven straight into the left baffle must not pass through it.
nav.world.reset(heading_deg=0)
max_x = -np.inf
for _ in range(100):
    nav.world.step([.78, .78])
    max_x = max(max_x, nav.world.position[0])
assert max_x < -6.5, max_x
records = []
for condition in ('combined', 'vision_only', 'odor_only', 'neither', 'source_off'):
    for seed, heading in enumerate((45, 75, 105), 1):
        nav.reset(condition='combined' if condition=='source_off' else condition,
                  heading_deg=heading, seed=seed, duration=5, food_odor=condition!='source_off')
        sums = dict(spikes=0, L2_spikes=0, PN_spikes=0, ORN_spikes=0)
        frames = []
        while not nav.done:
            s = nav.step(images=False)
            b = s['brain']
            sums['spikes'] += b['spikes']
            sums['L2_spikes'] += s['decoder']['L2_spikes']
            sums['PN_spikes'] += sum(b['olfaction']['PN_spikes'])
            sums['ORN_spikes'] += int(sum(brain.last_delta[brain.groups['ORN_DM1_'+side]].sum() for side in ('left','right')))
            frames.append({key:s[key] for key in ('time','position','heading_deg','odor','decoder','score')})
            if condition=='combined' and heading==75 and nav.frames==30:
                for key, data in [('body',nav.world.images()['body']), *zip(('left','right'),nav.world.images()['eyes'])]:
                    Path(f'outputs/maze-{key}.jpg').write_bytes(base64.b64decode(data))
            if condition=='neither':
                assert b['spikes']==0 and s['decoder']['gains']==[.78,.78]
            if condition in ('vision_only','neither','source_off'):
                assert b['olfaction']['input_rates_hz']==[0.0,0.0]
            if condition=='source_off':assert s['odor']==[0.0,0.0]
        record={'condition':condition,'config':s['config'],'status':s['status'],
                'time':s['time'],'distance_mm':s['score']['distance_mm'],
                'contact_bins':s['contact_bins'],'activity':sums,'frames':frames}
        records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!='frames'}), flush=True)
        Path('outputs/maze-validation.json').write_text(json.dumps({'trials':records,'wall_block_max_x':float(max_x),'wall_seconds':time.time()-started},indent=2))
nav.world.close()
