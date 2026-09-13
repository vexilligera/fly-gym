"""Check maze HTTP behavior, sensory disconnections and shared-brain ownership."""
import argparse
import base64
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8001);args=p.parse_args()
def api(path,body=None,origin=None):
    req=urllib.request.Request(f'http://127.0.0.1:{args.port}/api/'+path)
    if body is not None:
        req.data=json.dumps(body).encode();req.add_header('Content-Type','application/json')
    if origin:req.add_header('Origin',origin)
    with urllib.request.urlopen(req,timeout=120) as r:return json.load(r)
def reject(path,body,code=400,origin=None):
    try:api(path,body,origin)
    except urllib.error.HTTPError as e:assert e.code==code,(path,e.code)
    else:raise AssertionError('Invalid command accepted: '+path)
def finish():
    for _ in range(200):
        s=api('maze/status')
        if s.get('done'):return s
        assert s['status']!='error',s
        time.sleep(.1)
    raise AssertionError('Trial did not finish')

assert api('brain/status')['status']=='ready'
s=api('maze/reset',{'condition':'combined','duration':5,'heading_deg':75,'seed':1})
assert s['time']==0 and s['brain'] is None
for key,data in [('body',s['images']['body']),*zip(('left','right'),s['images']['eyes'])]:
    image=Image.open(io.BytesIO(base64.b64decode(data)))
    assert image.size==((640,480) if key=='body' else (225,256))
geometry=api('maze/world')
assert len(geometry['walls'])==6 and geometry['sugar']==[0,0]
assert max(map(max,geometry['odor_field']['values']))==1
api('maze/start',{})
time.sleep(.3)
s=api('maze/status');assert s['time']>0 and s['brain']['time']==round(s['time'],2)
assert len(s['contrast'][0])==721 and len(s['odor'])==2
assert s['brain']['olfaction']['input_rates_hz'][0]>0
reject('brain/step',{'stimulus':'walk'})
reject('vision/start',{'condition':'vision'})
assert api('vision/status')['status']=='idle'
paused=api('maze/pause',{})
time.sleep(.15);assert api('maze/status')['time']==paused['time']
for invalid in ({'food_odor':'yes'},{'duration':0},{'heading_deg':float('nan')},{'seed':True},{'condition':'bad'},{'target_deg':30}):
    reject('maze/reset',invalid)
    current=api('maze/status');assert current['time']==paused['time'] and current['trial_id']==paused['trial_id']
reject('maze/start',{},403,'https://unapproved.invalid')
api('maze/start',{})
arrived=finish();assert arrived['status']=='reached' and arrived['score']['distance_mm']<2.5
for condition in ('combined','vision_only','odor_only','neither'):
    api('maze/start',{'condition':condition,'duration':.2})
    s=finish();b=s['brain'];assert abs(s['time']-.2)<1e-7
    if condition=='neither':assert b['spikes']==0 and s['decoder']['gains']==[.78,.78]
    if condition in ('vision_only','neither'):assert b['olfaction']['input_rates_hz']==[0,0]
api('maze/start',{'food_odor':False,'duration':.2})
s=finish();assert s['odor']==[0,0] and s['brain']['olfaction']['input_rates_hz']==[0,0]
api('vision/reset',{'duration':.1})
reject('maze/pause',{})
assert api('vision/status')['config']['condition']=='vision'
api('maze/reset',{'duration':.1})
api('brain/reset',{});assert api('maze/status')['status']=='idle'
result={'passed':True,'checks':['decodable live images','maze and field geometry','autonomous dual sensory stepping','shared brain ownership','pause/resume/reset','invalid parameters preserve trial','origin guard','default reaches food zone','all four sensory conditions','source off zeros samples','cross-mode pause preserves owner']}
Path('outputs/maze-api-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
