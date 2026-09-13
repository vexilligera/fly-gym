"""Exercise continuous stepping, pause/reset, ownership and HTTP validation."""
import argparse
import base64
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8000);args=p.parse_args()
def api(path, body=None, origin=None):
    req=urllib.request.Request(f'http://127.0.0.1:{args.port}/api/'+path)
    if body is not None:
        req.data=json.dumps(body).encode();req.add_header('Content-Type','application/json')
    if origin:req.add_header('Origin',origin)
    with urllib.request.urlopen(req,timeout=120) as response:return json.load(response)

def rejected(path,body,code,origin=None):
    try:api(path,body,origin)
    except urllib.error.HTTPError as error:assert error.code==code
    else:raise AssertionError('Invalid command was accepted')

assert api('brain/status')['status']=='ready'
s=api('vision/reset',{'target_deg':30,'condition':'vision','duration':3})
assert s['time']==0 and s['brain'] is None
for key,img in [('body',s['images']['body']),('left',s['images']['eyes'][0]),('right',s['images']['eyes'][1])]:
    decoded=base64.b64decode(img)
    image=Image.open(io.BytesIO(decoded))
    assert image.size==((640,480) if key=='body' else (225,256))
    Path(f'outputs/vision-api-{key}.jpg').write_bytes(decoded)
first=s['trial_id']
api('vision/start',{})
# Advance without a stream of browser step requests.
time.sleep(.4)
s=api('vision/status');assert s['time']>0
rejected('brain/step',{'stimulus':'walk'},400)
paused=api('vision/pause',{})
time.sleep(.15)
assert api('vision/status')['time']==paused['time']
api('vision/start',{})
time.sleep(.3)
s=api('vision/pause',{});assert s['time']>paused['time']
rejected('vision/reset',{'heading_deg':float('nan')},400)
assert api('vision/status')['time']==s['time']
rejected('vision/reset',{'condition':'invalid'},400)
rejected('vision/start',{},403,'https://unapproved.invalid')
reset=api('vision/reset',{'condition':'blind','duration':.1})
assert reset['time']==0 and reset['trial_id']!=first
api('vision/start',{})
for _ in range(100):
    s=api('vision/status')
    if s['done']:break
    time.sleep(.1)
assert s['done'] and s['time']>.09 and s['brain']['spikes']==0
assert s['decoder']['gains']==[.9,.9]
api('brain/reset',{})
assert api('vision/status')['status']=='idle'
result={'passed':True,'checks':['images decode','autonomous compute loop','pause is stable','resume continues clocks','reset identity','invalid input rejected without state loss','same-origin guard','manual/navigation ownership','blind brain remains silent']}
Path('outputs/vision-api-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
