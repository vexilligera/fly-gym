"""Explicit candidate sensory assignments; never an automatic model promotion."""
from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product

import numpy as np
from scipy.special import expit

from brain.leg_reflex import LegCircuit, LegFixture, ReflexParameters, ReflexProtocol

SEEDS = (1, 2, 3)
SETTINGS = {
    'nominal': {'max_hz': 150., 'midpoint_shift_deg': 0.},
    'gain_75': {'max_hz': 75., 'midpoint_shift_deg': 0.},
    'gain_300': {'max_hz': 300., 'midpoint_shift_deg': 0.},
    'midpoint_minus10': {'max_hz': 150., 'midpoint_shift_deg': -10.},
    'midpoint_plus10': {'max_hz': 150., 'midpoint_shift_deg': 10.},
}
ASSIGNMENTS = [''.join(letters) for letters in product('FE', repeat=4)]


def protocols():
    result = {}
    for delta in (-10, -30, -50, 10, 30, 50):
        result[f'{"flex" if delta < 0 else "extend"}_{abs(delta)}'] = ReflexProtocol(displacement_deg=delta)
    for delta in (-30, 30):
        result[f'{"flex" if delta < 0 else "extend"}_30_slow'] = ReflexProtocol(displacement_deg=delta, ramp_s=.2)
    result['sham'] = ReflexProtocol(displacement_deg=0)
    result['sham_slow'] = ReflexProtocol(displacement_deg=0, ramp_s=.2)
    return result


def protocol_names(setting):
    return list(protocols()) if setting == 'nominal' else ['flex_30', 'extend_30', 'sham']


def candidate_rates(angle_deg, assignment, setting, frozen_fit):
    if assignment not in ASSIGNMENTS:
        raise ValueError('Expected four F/E labels in sensory body-ID order')
    values = SETTINGS[setting]
    rates = []
    for letter in assignment:
        cohort = 'JR209' if letter == 'F' else 'JR688'
        _, _, midpoint, width = frozen_fit['cohorts'][cohort]['selected']['parameters']
        midpoint += values['midpoint_shift_deg']
        fraction = expit((-1 if letter == 'F' else 1)*(angle_deg-midpoint)/width)
        rates.append(values['max_hz']*fraction)
    return np.asarray(rates)


class AssignedLegCircuit(LegCircuit):
    """Same LIF dynamics with an explicit per-neuron external sensory input.

    Kept separate so the published baseline implementation and provenance stay
    intact. Regression tests compare states, spikes and activations to LegCircuit.
    """
    def step_rates(self, rates_hz, *, connected=True, sensory_on=True, motor_on=True):
        p = self.parameters
        drive = np.asarray(rates_hz, dtype=float)
        if drive.shape != (len(self.sensory),) or not np.isfinite(drive).all() or np.any(drive < 0) or np.any(drive*self.dt > 1):
            raise ValueError('Invalid per-sensory-neuron rate vector')
        if not sensory_on:
            drive = np.zeros_like(drive)
        counts = np.zeros(len(self.ids), dtype=int)
        motor = np.concatenate((self.flexor, self.extensor))
        for _ in range(10):
            self.current += self.queue[self.pointer]
            self.queue[self.pointer] = 0
            self.current *= np.exp(-self.dt/p.synaptic_tau_s)
            self.refractory = np.maximum(0, self.refractory-self.dt)
            free = self.refractory <= 0
            self.voltage[free] += self.dt/p.membrane_tau_s * (-52-self.voltage[free]+self.current[free])
            self.voltage[~free] = -52
            self.voltage[self.sensory] = -52
            fired = (self.voltage >= -45) & free
            fired[self.sensory] = self.rng.random(len(self.sensory)) < drive*self.dt
            if not motor_on:
                fired[motor] = False
                self.voltage[motor] = -52
                self.current[motor] = 0
            self.voltage[fired] = -52
            self.refractory[fired] = p.refractory_s
            if connected:
                self.queue[self.pointer] = self.weights @ fired.astype(float)
            self.pointer = (self.pointer + 1) % len(self.queue)
            counts += fired
        self.counts += counts
        decay = np.exp(-.001/p.motor_filter_s)
        self.rates = decay*self.rates + (1-decay)*counts/.001
        flexor = float(self.rates[self.flexor].mean())
        extensor = float(self.rates[self.extensor].mean())
        activation = np.clip(p.muscle_gain*np.array([flexor, extensor])/p.motor_rate_scale_hz, 0, 1)
        return activation, {'sensory_drive_hz': float(drive.mean()), 'sensory_drive_by_cell_hz': drive.tolist(),
                            'sensory_hz': float(self.rates[self.sensory].mean()),
                            'flexor_hz': flexor, 'extensor_hz': extensor,
                            'network_spikes': int(counts.sum())}


def run_candidate(graph, frozen_fit, assignment, setting, protocol, seed=1, condition='connected', legacy=False):
    if condition not in ('connected', 'disconnected', 'sensory_off', 'motor_silenced'):
        raise ValueError('Unknown circuit condition')
    fixture = LegFixture(protocol.rest_angle_deg, protocol.distal_mass_scale)
    brain = AssignedLegCircuit(graph, seed=seed)
    if graph['sensory_ids'] != sorted(graph['sensory_ids']):
        raise ValueError('Assignment letters require ascending sensory IDs')
    trace = []
    at_onset = None
    try:
        for step in range(1600):
            time_s = step*.001
            held = protocol.held_angle(time_s)
            if held is not None:
                fixture.hold(held)
            angle = fixture.angle()
            if legacy:
                p = brain.parameters
                rates = np.full(len(brain.sensory), p.sensory_max_hz*np.clip((angle-fixture.rest_angle)/p.sensory_span_deg, 0, 1))
            else:
                rates = candidate_rates(angle, assignment, setting, frozen_fit)
            if step == round(protocol.onset_s/.001):
                at_onset = brain.counts.copy()
            activation, neural = brain.step_rates(rates, connected=condition != 'disconnected',
                sensory_on=condition != 'sensory_off', motor_on=condition != 'motor_silenced')
            fixture.step(activation, held)
            if step % 5 == 0:
                trace.append({'time_s': float(fixture.data.time), 'angle_deg': fixture.angle(),
                              'held': held is not None, 'muscle_activation': activation.tolist(), **neural})
        return {'assignment': assignment, 'setting': setting, 'seed': seed, 'condition': condition,
                'protocol': asdict(protocol), 'duration_s': 1.6, 'trace': trace,
                'spikes_by_body_id': dict(zip(map(str, brain.ids), map(int, brain.counts))),
                'post_onset_spikes_by_body_id': dict(zip(map(str, brain.ids), map(int, brain.counts-at_onset))),
                'motor_spikes': int(brain.counts[np.r_[brain.flexor, brain.extensor]].sum()),
                'total_spikes': int(brain.counts.sum()), 'final_angle_deg': fixture.angle(),
                'fixture_counter_torque': fixture.counter_torque,
                'unknown_sign_edges_disabled': brain.unknown_sign_edges}
    finally:
        fixture.close()
