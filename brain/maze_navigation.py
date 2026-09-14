"""Engineered fusion of visual L2 and olfactory ORN activity in a simple maze.

The policy receives neural activity and its own history only. World pose,
wall geometry, odor maps and sugar coordinates do not enter the motor decoder.
"""
import time
import uuid
import numpy as np
from brain.maze_world import MazeWorld
from brain.maze_layouts import get_layout


class MazePolicy:
    def __init__(self, bearings, mask, seed):
        self.bearings = np.asarray(bearings)
        self.mask = np.asarray(mask)
        self.visual = np.zeros(len(bearings))
        self.olfactory = np.zeros(2)
        self.turn_bias = 1 if seed%2 else -1
        self.escape = 0
        self.steps = 0

    def step(self, visual_counts, orn_rates, vision, olfaction):
        self.visual += (1-np.exp(-.02/.06))*(visual_counts-self.visual)
        self.olfactory += (1-np.exp(-.02/.12))*(np.asarray(orn_rates)-self.olfactory)
        a = self.bearings
        # Population means, not a geometric raycast or a ground-truth depth map.
        def level(low,high):
            selected = self.mask & (a>=low) & (a<high)
            return float(self.visual[selected].mean()) if selected.any() and vision else 0.0
        left, front, right = level(.25,1.5), level(-.25,.25), level(-1.5,-.25)
        ol = self.olfactory if olfaction else np.zeros(2)
        odor_direction = float(np.clip((ol[0]-ol[1])/(ol.sum()+8),-1,1))
        odor_turn = 1.05*odor_direction
        # Repel the higher wall response. At a head-on wall choose and hold a
        # turn until it clears rather than oscillating on shot noise each bin.
        avoidance = float(np.clip(1.2*(right-left),-.65,.65))
        if vision and front > .18 and self.escape <= 0:
            self.turn_bias = 1 if (right-left+.2*odor_direction)>=0 else -1
            self.escape = 12
        escaping = self.escape > 0
        if escaping:
            turn = .65*self.turn_bias
            forward = .52
            self.escape -= 1
        else:
            turn = np.clip(odor_turn+avoidance,-.65,.65)
            forward = .78 * (1-.45*np.clip(front,0,1))
        gains = np.clip([forward-turn,forward+turn],.12,1.2)
        self.steps += 1
        return gains, {'L2_spikes':int(np.sum(visual_counts)),
                       'wall_left':left,'wall_front':front,'wall_right':right,
                       'odor_left_hz':float(ol[0]),'odor_right_hz':float(ol[1]),
                       'odor_turn':float(odor_turn),'visual_turn':avoidance,
                       'turn':float(turn),'escape':escaping, 'gains':gains.tolist()}


class MazeNavigation:
    def __init__(self,brain):
        self.brain = brain
        self.world = None
        self.reset()

    def reset(self,condition='combined',heading_deg=75,seed=1,duration=120,food_odor=True,layout='complex'):
        get_layout(layout)
        if condition not in ('combined','vision_only','odor_only','neither'):
            raise ValueError('Unknown maze sensory condition')
        for value,low,high in [(heading_deg,-180,180),(duration,.1,120)]:
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not low<=value<=high:
                raise ValueError('Heading must be −180…180°; duration .1…120 seconds')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<=100000:
            raise ValueError('Seed must be an integer in 0…100000')
        if not isinstance(food_odor,bool):raise ValueError('food_odor must be a boolean')
        if self.world is None or self.world.layout.key != layout:
            replacement = MazeWorld(layout)
            if self.world is not None:self.world.close()
            self.world = replacement
        self.config = dict(condition=condition,heading_deg=heading_deg,seed=seed,duration=duration,food_odor=food_odor,layout=layout)
        self.trial_id = uuid.uuid4().hex
        self.brain.reset()
        self.world.food_odor = food_odor
        self.world.reset(heading_deg=heading_deg,seed=seed)
        self.world.vision();self.world.smell()
        indices = self.brain.visual.readout_pixels
        self.policy = MazePolicy(self.world.bearings.ravel()[indices],self.world.visual_mask.ravel()[indices],seed)
        self.frames = 0;self.wall_seconds = 0;self.done = False;self.status = 'paused'
        self.taste = None
        self.contact_bins = 0
        self.state = self.snapshot(None,{},True)
        return self.state

    def step(self,images=True):
        if self.done:return self.state
        if self.taste is not None:
            started = time.perf_counter()
            neural = self.taste.step()
            self.world.feeding.step(neural['sugar']['MN9_hz'])
            if abs(self.world.feeding.data.time-neural['time']) > 1e-7:
                raise RuntimeError('Brain/proboscis clock mismatch')
            self.frames += 1
            self.done = self.taste.done
            self.status = 'reached' if self.done else 'tasting'
            self.state = {**self.state, 'status': self.status, 'frame': self.frames,
                          'done': self.done, 'brain': neural, 'taste': self.taste_snapshot()}
            if images:
                self.state['images'] = {**self.state['images'], 'body': self.world.feeding.image()}
            time.sleep(max(0, .1-(time.perf_counter()-started)))
            return self.state
        started = time.perf_counter()
        vision = self.config['condition'] in ('combined','vision_only')
        olfaction = self.config['condition'] in ('combined','odor_only')
        contrast, odor = self.world.vision(), self.world.smell()
        neural = self.brain.step_multisensory(contrast,odor,vision,olfaction)
        gains, decoder = self.policy.step(self.brain.last_delta[self.brain.visual.readout],
                                          neural['olfaction']['ORN_rates_hz'],vision,olfaction)
        self.world.step(gains)
        if abs(self.world.data.time-neural['time'])>1e-7:raise RuntimeError('Brain/body clock mismatch')
        self.frames += 1
        self.world.path.append(self.world.position[:2].tolist())
        score = self.world.score()
        self.contact_bins += int(score['wall_contacts']>0)
        reached = score['distance_mm']<2.5
        self.done = reached or not score['upright'] or self.world.data.time>=self.config['duration']-1e-8
        self.status = 'reached' if reached else 'fallen' if not score['upright'] else 'finished' if self.done else 'running'
        self.state = self.snapshot(neural,decoder,images)
        self.wall_seconds += time.perf_counter()-started
        self.state['playback_speed'] = float(self.world.data.time/self.wall_seconds)
        return self.state

    def start_taste(self, rate_hz=200):
        if not self.done or self.status != 'reached':
            raise ValueError('Reach the food zone before starting a sugar-taste assay')
        from brain.sugar_assay import SugarAssay
        SugarAssay.validate_rate(rate_hz)
        self.world.start_feeding()
        assay = SugarAssay(self.brain, rate_hz, paced=False)
        self.taste = assay
        self.done = False
        self.status = 'tasting'
        self.frames += 1
        self.state = {**self.state, 'status': 'tasting', 'done': False,
                      'frame': self.frames, 'brain': None, 'taste': self.taste_snapshot(),
                      'images': {**self.state['images'], 'body': self.world.feeding.image()}}
        return self.state

    def taste_snapshot(self):
        return {**self.taste.snapshot(), 'body_held': False, 'torso_legs_held': True,
                'proboscis': self.world.feeding.snapshot()}

    def snapshot(self,neural,decoder,images):
        w = self.world
        state = {'status':self.status,'trial_id':self.trial_id,'config':self.config.copy(),
                 'time':float(w.data.time),'frame':self.frames,'done':self.done,
                 'position':w.position.tolist(),'heading_deg':float(np.rad2deg(w.heading)),
                 'target':[0,0],'path':w.path.copy(),'score':w.score(),'decoder':decoder,
                 'odor':w.odor.tolist(),'antennae':w.antennae.tolist(),
                 'contrast':w.contrast.round(4).tolist(),'retina':w.readings.round(4).tolist(),
                 'sensory_time':max(0,float(w.data.time)-(.02 if neural else 0)),
                 'brain':neural,'playback_speed':0.0,'contact_bins':self.contact_bins}
        if images:state['images']=w.images()
        return state
