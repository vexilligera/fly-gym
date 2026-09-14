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
s=api('maze/reset',{'controller':'sensory_policy','layout':'simple','condition':'combined','duration':5,'heading_deg':75,'seed':1})
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
for invalid in ({'food_odor':'yes'},{'duration':0},{'heading_deg':float('nan')},{'seed':True},{'condition':'bad'},{'target_deg':30},{'layout':'unknown'},{'layout':[]},{'controller':'unknown'},{'silence_descending':'true'},
                {'controller':'sensory_policy','silence_descending':True}):
    reject('maze/reset',invalid)
    current=api('maze/status');assert current['time']==paused['time'] and current['trial_id']==paused['trial_id']
reject('maze/start',{},403,'https://unapproved.invalid')
api('maze/start',{})
arrived=finish();assert arrived['status']=='reached' and arrived['score']['distance_mm']<2.5
for condition in ('combined','vision_only','odor_only','neither'):
    api('maze/start',{'controller':'sensory_policy','condition':condition,'duration':.2})
    s=finish();b=s['brain'];assert abs(s['time']-.2)<1e-7
    if condition=='neither':assert b['spikes']==0 and s['decoder']['gains']==[.78,.78]
    if condition in ('vision_only','neither'):assert b['olfaction']['input_rates_hz']==[0,0]
api('maze/start',{'food_odor':False,'duration':.2})
s=finish();assert s['odor']==[0,0] and s['brain']['olfaction']['input_rates_hz']==[0,0]
# The new default consumes actual DN gains; sensory input never directly drives DNs.
for arguments in ({'condition':'combined'}, {'condition':'neither'}, {'silence_descending':True}):
    api('maze/start', {**arguments,'duration':.4})
    s=finish(); b=s['brain']
    assert s['config']['controller']=='descending'
    assert s['decoder']['gains']==b['gains']==b['motor_readout']['gains']
    if arguments.get('condition')=='neither' or arguments.get('silence_descending'):
        assert s['decoder']['gains']==[0,0]
    if arguments.get('silence_descending'):
        assert b['spikes']>0 and len(b['silenced_indices'])>0
        assert not set(b['silenced_indices']) & set(b['activity']['indices'])
        assert all(b['rates_hz'][key]==0 for key in b['rates_hz'] if not key.startswith('ORN'))
# One persistent camera connection spans live frames, pauses, and layout reset.
first=api('maze/reset',{'layout':'complex','duration':2})
assert api('maze/world')['layout']=='complex' and len(api('maze/world')['walls'])==20
stream=urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/maze/camera.mjpg',timeout=10)
assert stream.headers['Content-Type'].startswith('multipart/x-mixed-replace')
def camera_frame():
    while True:
        line=stream.readline()
        assert line,'Camera stream ended unexpectedly'
        if line.strip()==b'--flyframe':break
    headers={}
    while True:
        line=stream.readline().strip()
        if not line:break
        key,value=line.decode().split(':',1);headers[key]=value.strip()
    frame=stream.read(int(headers['Content-Length']))
    assert Image.open(io.BytesIO(frame)).size==(640,480)
    return headers
assert camera_frame()['X-Trial-Id']==first['trial_id']
api('maze/start',{})
times=[];start=time.monotonic()
for _ in range(20):times.append(float(camera_frame()['X-Simulation-Time']))
delivery_fps=20/(time.monotonic()-start)
assert all(b>a for a,b in zip(times,times[1:])),times
assert delivery_fps>5,delivery_fps
light=api('maze/status?body=0');assert 'body' not in light['images'] and len(light['images']['eyes'])==2
api('maze/pause',{})
replacement=api('maze/reset',{'layout':'simple','duration':.1})
for _ in range(30):
    frame=camera_frame()
    if frame['X-Trial-Id']==replacement['trial_id']:break
else:raise AssertionError('Camera stream did not switch layouts')
assert float(frame['X-Simulation-Time'])==0
assert len(api('maze/world')['walls'])==6
stream.close()
api('vision/reset',{'duration':.1})
reject('maze/pause',{})
assert api('vision/status')['config']['condition']=='vision'
api('maze/reset',{'duration':.1})
api('brain/reset',{});assert api('maze/status')['status']=='idle'
result={'passed':True,'camera_delivery_fps':delivery_fps,'checks':['decodable live images','both maze layouts and field geometry','autonomous dual sensory stepping','shared brain ownership','pause/resume/reset','invalid parameters preserve trial','origin guard','simple maze reaches food zone','all four sensory conditions','source off zeros samples','cross-mode pause preserves owner','persistent camera stream advances','camera stream survives layout reset','camera-free polling payload']}
Path('outputs/maze-api-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
