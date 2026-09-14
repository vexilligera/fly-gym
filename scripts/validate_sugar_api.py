"""End-to-end sugar assay controls against a staging HTTP service."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import urllib.error

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8001)
args = parser.parse_args()
base = f'http://127.0.0.1:{args.port}/api/'

def api(path, data=None):
    request = urllib.request.Request(base+path, data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)

def reject(path, data):
    try:
        api(path, data)
    except urllib.error.HTTPError as error:
        assert error.code == 400
    else:
        raise AssertionError('Invalid action accepted')

def finished(check=lambda s: None):
    deadline = time.monotonic()+120
    while time.monotonic() < deadline:
        state = api('maze/status?body=0')
        assert state['status'] != 'error', state
        check(state)
        if state['done']:
            return state
        time.sleep(.15)
    raise AssertionError('Trial did not finish')

assert api('brain/status')['status'] == 'ready'
reject('maze/taste', {'rate_hz':200})
api('maze/start', {'layout':'simple','duration':5})
arrival = finished()
assert arrival['status'] == 'reached'
for bad in [201, -1, True, '200', float('nan')]:
    reject('maze/taste', {'rate_hz':bad})
    current = api('maze/status?body=0')
    assert current['trial_id'] == arrival['trial_id'] and current['brain']['time'] == arrival['brain']['time']

begin = api('maze/taste', {'rate_hz':200})
assert begin['taste']['time'] == 0 and begin['brain'] is None
reject('maze/taste', {'rate_hz':200})
reject('brain/step', {'stimulus':'walk'})
time.sleep(.5)
paused = api('maze/pause', {})
time.sleep(.3)
assert api('maze/status?body=0')['taste']['time'] == paused['taste']['time']
api('maze/start', {})

motion = []
def held(state):
    assert state['position'] == arrival['position'] and state['path'] == arrival['path']
    assert state['time'] == arrival['time']
    assert state['taste']['torso_legs_held'] and not state['taste']['body_held']
    assert not state['taste']['vision_and_odor_input']
    mouth = state['taste']['proboscis']
    assert abs(mouth['time'] - state['taste']['time']) < 1e-8
    motion.append(mouth)
    if state['brain']:
        assert abs(state['brain']['time'] - state['taste']['time']) < 1e-8

sugar = finished(held)
assert sugar['status'] == 'reached' and sugar['taste']['time'] == 8
assert len(sugar['taste']['trace']) == 80
assert sum(p['MN9_right_hz'] for p in sugar['taste']['trace'][20:60]) > 0
assert min(p['angles_deg']['rostrum_pitch'] for p in motion) < -40
assert max(p['angles_deg']['mouth_yaw'] for p in motion) > 1
assert max(abs(x) for x in motion[-1]['angles_deg'].values()) < .02
assert all(not p['ingestion'] for p in motion)
sugar_motion = motion.copy()
with urllib.request.urlopen(base+'maze/camera.mjpg', timeout=10) as response:
    headers = response.read(240)
    assert b'X-View: proboscis' in headers and b'X-Assay-Time: 8.000' in headers
print('Sugar presentation, moving mouth, held torso/legs, pause/resume and ownership passed', flush=True)
api('maze/taste', {'rate_hz':0})
motion.clear()
control = finished(held)
assert all(p['GRN_hz'] == p['MN9_left_hz'] == p['MN9_right_hz'] == 0 for p in control['taste']['trace'])
assert max(abs(x) for p in motion for x in p['angles_deg'].values()) < 1e-6
reset = api('maze/reset', {'layout':'simple','duration':.1})
assert 'taste' not in reset and reset['time'] == 0 and reset['brain'] is None
result = {'passed':True,'arrival_time':arrival['time'],
          'checks':['arrival required','invalid rate preserves trial','shared ownership',
                    'pause and resume','held torso/legs and synchronized brain/mouth clock',
                    'downstream MN9 response','extension and turning','washout retraction',
                    'MJPEG assay timestamp','silent and stationary no-taste control','reset clears assay'],
          'sugar_trace':sugar['taste']['trace'],'control_trace':control['taste']['trace'],
          'sugar_motion':sugar_motion,'control_motion':motion}
Path('outputs/sugar-api-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('Sugar API validation passed', flush=True)
