"""Baseline → imposed sugar-GRN input → washout in a held-body assay."""
import math
import time


class SugarAssay:
    @staticmethod
    def validate_rate(rate_hz):
        if isinstance(rate_hz, bool) or not isinstance(rate_hz, (int, float)) or not math.isfinite(rate_hz) or not 0 <= rate_hz <= 200:
            raise ValueError('Sugar GRN input must be 0–200 Hz')

    def __init__(self, brain, rate_hz=200, paced=True):
        self.validate_rate(rate_hz)
        self.brain, self.rate_hz = brain, rate_hz
        self.paced = paced
        self.steps = 0
        self.trace = []
        self.bucket = []
        self.done = False
        self.phase = 'baseline'
        self.contact = True
        self.brain.reset()

    def step(self, contact=True):
        started = time.perf_counter()
        self.contact = bool(contact)
        before = self.steps * .02
        self.phase = 'baseline' if before < 2 else 'sugar' if before < 6 else 'washout'
        rate = self.rate_hz if self.phase == 'sugar' and self.contact else 0
        neural = self.brain.step_sugar(rate)
        self.steps += 1
        self.done = self.steps >= 400
        self.bucket.append(neural['sugar'])
        if self.steps % 5 == 0:
            self.trace.append({'time': self.steps * .02, 'input_rate_hz': rate,
                'GRN_hz': sum(x['GRN_hz'] for x in self.bucket) / len(self.bucket),
                'MN9_left_hz': sum(x['MN9_hz']['left'] for x in self.bucket) / len(self.bucket),
                'MN9_right_hz': sum(x['MN9_hz']['right'] for x in self.bucket) / len(self.bucket)})
            self.bucket.clear()
        # Deliberate 0.2× playback makes a short neural response watchable.
        # The LIF timestep and the 20 ms neural bins are unchanged.
        if self.paced:
            time.sleep(max(0, .1 - (time.perf_counter() - started)))
        return neural

    def snapshot(self):
        return {'time': self.steps * .02, 'duration': 8, 'phase': self.phase,
                'rate_hz': self.rate_hz, 'done': self.done, 'trace': self.trace.copy(),
                'body_held': True, 'vision_and_odor_input': False,
                'protocol': '2 s baseline, 4 s sugar GRN input, 2 s washout; neural state resets at assay start',
                'playback_speed': .2}
