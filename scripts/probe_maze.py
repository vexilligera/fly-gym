"""Development probe for the multisensory maze (prints a measured trajectory)."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from brain.model import ConnectomeBrain
from brain.maze_navigation import MazeNavigation
b=ConnectomeBrain(backend='cuda')
n=MazeNavigation(b)
records=[]
for condition,heading in [('combined',75),('odor_only',75),('vision_only',75)]:
 n.reset(condition=condition,heading_deg=heading,duration=10)
 frames=[]
 while not n.done:
  s=n.step(images=False)
  if n.frames%25==0 or n.done:
   record={key:s[key] for key in ['time','position','heading_deg','score','decoder','odor','status']}
   frames.append(record);print(condition,json.dumps(record),flush=True)
 records.append({'config':s['config'],'frames':frames,'final':{key:s[key] for key in ['status','time','score','contact_bins']}})
Path('outputs/maze-probe.json').write_text(json.dumps(records,indent=2))
n.world.close()
