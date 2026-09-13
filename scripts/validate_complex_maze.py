"""Check maze connectivity, body clearance, and measured controller behavior."""
from collections import deque
from pathlib import Path
import argparse
import base64
import json
import sys
import time
import numpy as np
from scipy.ndimage import binary_erosion, label
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from brain.maze_layouts import LAYOUTS, PASSAGES
from brain.odor_field import OdorField

def geometry_checks():
    adjacency={i:[] for i in range(25)}
    for a,b in PASSAGES:
        adjacency[a].append(b);adjacency[b].append(a)
    previous={0:None};queue=deque([0])
    while queue:
        cell=queue.popleft()
        for neighbor in adjacency[cell]:
            if neighbor not in previous:
                previous[neighbor]=cell;queue.append(neighbor)
    assert len(previous)==25 and len(PASSAGES)==24
    assert sum(len(a)==1 for a in adjacency.values())==5
    route=[];cell=12
    while cell is not None:route.append(cell);cell=previous[cell]
    directions=[(b%5-a%5,b//5-a//5) for a,b in zip(route,route[1:])]
    assert len(directions)==12
    assert sum(a!=b for a,b in zip(directions,directions[1:]))==8
    layout=LAYOUTS['complex'];field=OdorField(layout.walls)
    # Leave 1.5 mm clearance from walls for the body and feet.
    clearance=binary_erosion(~field.blocked,iterations=3)
    components,_=label(clearance)
    centers=[(x,y) for y in range(-16,17,8) for x in range(-16,17,8)]
    def index(x):return int(round((x-field.axis[0])/field.spacing))
    regions={int(components[index(y),index(x)]) for x,y in centers}
    assert len(regions)==1 and 0 not in regions,regions
    assert all(field.sample(centers)>0)
    print(json.dumps({'geometry':'passed','cells':25,'dead_ends':5,'walls':len(layout.walls),
                      'body_clearance_mm':1.5,'start_odor':float(field.sample([layout.start])[0])}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--geometry-only',action='store_true');args=parser.parse_args()
    geometry_checks()
    if args.geometry_only:return
    from brain.model import ConnectomeBrain
    from brain.maze_navigation import MazeNavigation
    brain=ConnectomeBrain(backend='cuda');nav=MazeNavigation(brain)
    records=[]
    for layout,heading,seed,duration in [('complex',75,1,30),('complex',90,2,10),('simple',75,1,5)]:
        nav.reset(layout=layout,heading_deg=heading,seed=seed,duration=duration)
        totals={}
        originals=[]
        def profile(obj,name,label):
            original=getattr(obj,name);originals.append((obj,name,original))
            def timed(*args,**kwargs):
                start=time.perf_counter();result=original(*args,**kwargs)
                totals[label]=totals.get(label,0)+time.perf_counter()-start
                return result
            setattr(obj,name,timed)
        profile(nav.world,'vision','eye_render_and_retina')
        profile(brain,'step_multisensory','brain')
        profile(nav.world,'step','body_physics')
        profile(nav.world,'images','body_render_and_jpeg')
        frames=[];started=time.perf_counter()
        if layout=='complex' and seed==1:
            Path('outputs/complex-maze-preview.jpg').write_bytes(base64.b64decode(nav.state['images']['body']))
        while not nav.done:
            s=nav.step(images=True)
            frames.append({k:s[k] for k in ('time','position','score','odor','decoder')})
        elapsed=time.perf_counter()-started
        for obj,name,original in originals:setattr(obj,name,original)
        report={'config':s['config'],'status':s['status'],'time':s['time'],
                'distance_mm':s['score']['distance_mm'],'contact_bins':s['contact_bins'],
                'frames':nav.frames,'wall_seconds':elapsed,'worker_fps':nav.frames/elapsed,
                'phase_ms':{k:1000*v/nav.frames for k,v in totals.items()}}
        print(json.dumps(report),flush=True)
        records.append({**report,'trajectory':frames})
        Path('outputs/complex-maze-validation.json').write_text(json.dumps(records,indent=2))
        if layout=='simple':assert s['status']=='reached'
    nav.world.close()

if __name__=='__main__':main()
