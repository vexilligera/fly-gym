"""DesktopFly's reduced MaleCNS LIF model, ported for the MuJoCo transfer test.

Adapted from Denis Shiryaev's MIT-licensed LocomotorSim, revision
32b00011e83c3dc85fa3ea0b3934155b04f1635d. License and source are retained in
wasm/locomotor/vendor. Physiology and pooled sensory tuning are provisional.
The browser reference uses the unmodified upstream JavaScript implementation.
"""
from pathlib import Path
import json

import numba as nb
import numpy as np

GRAPH_PATH = Path(__file__).resolve().parents[1] / 'wasm/locomotor/vendor/locomotor_circuit.json'


@nb.njit(cache=True)
def _step(voltage, adaptation, rates, exc, inh, next_exc, next_inh, refractory,
          drive, sensory_drive, silenced, row_start, targets, weights, counts,
          baseline, kick, synapses):
    n = len(voltage)
    for i in range(n):
        exc[i] = exc[i] * .8187308 + next_exc[i]
        inh[i] = inh[i] * .9048374 + next_inh[i]
        next_exc[i] = 0.
        next_inh[i] = 0.
    for i in range(n):
        rates[i] *= .9048374
        adaptation[i] *= .9950125
        if silenced[i]:
            voltage[i] = 0.
            rates[i] = 0.
            continue
        if refractory[i] > 0:
            refractory[i] -= 1
            continue
        voltage[i] = max(-1., voltage[i] * .9512294 + exc[i] + inh[i]
                         + baseline + drive[i] + sensory_drive[i] - adaptation[i])
        if voltage[i] >= 1:
            voltage[i] = 0.
            refractory[i] = 2
            adaptation[i] += kick
            rates[i] += 95.16258
            counts[i] += 1
            if synapses:
                for e in range(row_start[i], row_start[i + 1]):
                    if weights[e] >= 0:
                        next_exc[targets[e]] += weights[e]
                    else:
                        next_inh[targets[e]] += weights[e]


class DesktopLocomotor:
    def __init__(self, graph=None):
        self.graph = graph or json.loads(GRAPH_PATH.read_text())
        self.neurons = self.graph['neurons']
        self.n = len(self.neurons)
        edges = np.asarray(self.graph['edges'], dtype=float)
        if not np.isfinite(edges).all() or not np.equal(edges[:, :2], np.floor(edges[:, :2])).all():
            raise ValueError('Invalid graph edges')
        pre, post = edges[:, :2].T.astype(int)
        if min(pre.min(), post.min()) < 0 or max(pre.max(), post.max()) >= self.n:
            raise ValueError('Invalid graph indices')
        input_total = np.zeros(self.n)
        for _, target, weight in edges:
            input_total[int(target)] += abs(weight)
        # Stable ordering retains the upstream edge accumulation order.
        order = np.argsort(pre, kind='stable')
        self.targets = post[order]
        self.weights = (2.4 * edges[:, 2] / np.maximum(60, input_total[post]))[order]
        self.row_start = np.r_[0, np.cumsum(np.bincount(pre, minlength=self.n))]
        for name in ('voltage', 'adaptation', 'rates', 'exc', 'inh', 'next_exc', 'next_inh', 'drive', 'sensory_drive'):
            setattr(self, name, np.zeros(self.n))
        self.refractory = np.zeros(self.n, dtype=np.int32)
        self.counts = np.zeros(self.n, dtype=np.int64)
        self.silenced = np.zeros(self.n, dtype=bool)
        self.motor = np.array([i for i, n in enumerate(self.neurons) if n['role'] == 'motor'])
        self.sensory = np.array([i for i, n in enumerate(self.neurons) if n['role'] == 'sensory'])
        self.motor_groups = {}
        for leg in range(6):
            for channel in ('coxa_promotor', 'coxa_anterior_rotator', 'coxa_remotor',
                            'coxa_posterior_rotator', 'trochanter_flexor', 'trochanter_extensor',
                            'tibia_flexor', 'tibia_extensor'):
                self.motor_groups[leg, channel] = np.array([i for i, n in enumerate(self.neurons)
                    if n['role'] == 'motor' and n['leg'] == leg and n['motorChannel'] == channel], dtype=int)
        self.synapses_enabled = True
        self.feedback_enabled = True
        self.sim_ms = 0

    def set_descending(self, cell_type, side, rate):
        if not np.isfinite(rate) or rate < 0:
            raise ValueError('Invalid stimulation')
        for i, n in enumerate(self.neurons):
            if n['role'] == 'descending' and n['type'] == cell_type and n['side'] == side:
                self.drive[i] = min(.35, rate * .004)

    def feedback(self, legs):
        self.sensory_drive[:] = 0.
        if not self.feedback_enabled:
            return
        for i in self.sensory:
            n = self.neurons[i]
            f = legs.get(n['leg'])
            if f is None:
                continue
            if n['sensoryKind'] in ('campaniform', 'contact'):
                value = min(1., f['load'] * 6) if f['contact'] else 0.
            elif n['sensoryKind'] == 'hair_plate':
                value = min(1., abs(f['hipAngle']) / .65 + abs(f['elevationVelocity']) / 20)
            else:
                value = min(1., abs(f['kneeVelocity']) / 20 + abs(f['hipVelocity']) / 16
                            + abs(f['kneeAngle'] - .95) * .35)
            self.sensory_drive[i] = value * .10

    def step(self, milliseconds=1):
        for _ in range(milliseconds):
            _step(self.voltage, self.adaptation, self.rates, self.exc, self.inh,
                  self.next_exc, self.next_inh, self.refractory, self.drive,
                  self.sensory_drive, self.silenced, self.row_start, self.targets,
                  self.weights, self.counts, .022, .01, self.synapses_enabled)
            self.sim_ms += 1

    def channel_rate(self, leg, channel):
        indices = self.motor_groups[leg, channel]
        return float(self.rates[indices].mean()) if len(indices) else 0.

    def tibia_activation(self, leg=1):
        rates = np.array([self.channel_rate(leg, channel) for channel in ('tibia_flexor', 'tibia_extensor')])
        return rates / (rates + 50.)
