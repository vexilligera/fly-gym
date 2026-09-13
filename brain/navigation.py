"""Closed-loop visual experiment; all state stays on the compute worker."""
import time
import uuid
import numpy as np
from brain.visual_world import VisualWorld


class VisualNavigation:
    def __init__(self, brain):
        self.brain = brain
        self.world = VisualWorld()
        self.config = {}
        self.reset()

    def reset(self, condition='vision', heading_deg=0, target_deg=30, seed=1, duration=8):
        if condition not in ('vision', 'blind', 'shuffled', 'readout_off'):
            raise ValueError('Unknown visual condition')
        for value, low, high in [(heading_deg, -60, 60), (target_deg, -60, 60), (duration, .1, 20)]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
                raise ValueError('Heading/target must be −60…60 degrees; duration .1…20 seconds')
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 100000:
            raise ValueError('Seed must be an integer in 0…100000')
        self.config = dict(condition=condition, heading_deg=heading_deg, target_deg=target_deg, seed=seed, duration=duration)
        self.trial_id = uuid.uuid4().hex
        self.brain.reset()
        self.world.reset(heading_deg, target_deg, seed)
        self.world.vision()
        self.filtered = np.zeros(len(self.brain.visual.readout))
        self.bearings = self.world.bearings.ravel()[self.brain.visual.readout_pixels]
        self.decoder_mask = self.world.visual_mask.ravel()[self.brain.visual.readout_pixels]
        self.wall_started = time.perf_counter()
        self.wall_seconds = 0.0
        self.frames = 0
        self.done = False
        self.status = 'paused'
        self.state = self.snapshot(None, 0.0, 0.0, [0.0, 0.0], images=True)
        return self.state

    def step(self, images=True):
        if self.done:
            return self.state
        started = time.perf_counter()
        # Image at t → neural bin [t,t+20ms] → body step to t+20ms.
        contrast = self.world.vision()
        neural = self.brain.step_visual(contrast, self.config['condition'])
        counts = self.brain.last_delta[self.brain.visual.readout]
        self.filtered += (1 - np.exp(-.02/.1)) * (counts - self.filtered)
        weights = self.filtered * self.decoder_mask
        sx, sy = np.cos(self.bearings) @ weights, np.sin(self.bearings) @ weights
        mass = weights.sum()
        bearing = float(np.arctan2(sy, sx)) if mass > .1 else 0.0
        coherence = float(np.hypot(sx, sy) / mass) if mass > .1 else 0.0
        turn = float(np.clip(bearing * .8, -.65, .65)) if coherence > .25 else 0.0
        if self.config['condition'] == 'readout_off':
            turn = 0.0
        gains = [float(np.clip(.9 - turn, .15, 1.2)), float(np.clip(.9 + turn, .15, 1.2))]
        self.world.step(gains)
        if abs(self.world.data.time - neural['time']) > 1e-7:
            raise RuntimeError('Brain and body simulation clocks diverged')
        self.frames += 1
        self.world.path.append(self.world.position[:2].tolist())
        score = self.world.score()
        # Ground-truth position is restricted to evaluation/termination. It
        # never supplies a bearing or speed to the motor decoder above.
        reached = score['distance_mm'] < 3.0
        self.done = reached or neural['time'] >= self.config['duration'] - 1e-8
        self.status = 'reached' if reached else 'finished' if self.done else 'running'
        self.state = self.snapshot(neural, bearing, coherence, gains, images)
        self.wall_seconds += time.perf_counter() - started
        self.state['playback_speed'] = float(self.world.data.time / self.wall_seconds)
        return self.state

    def snapshot(self, neural, bearing, coherence, gains, images):
        world = self.world
        state = {
            'status': self.status, 'trial_id': self.trial_id, 'config': self.config.copy(), 'time': float(world.data.time),
            'frame': self.frames, 'done': self.done,
            'position': world.position.tolist(), 'heading_deg': float(np.rad2deg(world.heading)),
            'target': world.target.tolist(), 'path': world.path[-1001:], 'score': world.score(),
            'decoder': {'bearing_deg': float(np.rad2deg(bearing)), 'coherence': coherence,
                        'gains': gains, 'L2_spikes': int(self.brain.last_delta[self.brain.visual.readout].sum())},
            'contrast': world.contrast.round(4).tolist(),
            'retina': world.readings.round(4).tolist(),
            'sensory_time': max(0.0, float(world.data.time) - (.02 if neural else 0)),
            'brain': neural, 'playback_speed': float(world.data.time / self.wall_seconds) if self.wall_seconds else 0,
        }
        if images:
            state['images'] = world.images()
        return state
