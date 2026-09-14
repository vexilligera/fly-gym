"""Check the candidate-input adapter, physical controls and complete sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from brain.leg_reflex import LegCircuit, ReflexProtocol
from brain.sensory_assignment import AssignedLegCircuit, candidate_rates, run_candidate, ASSIGNMENTS, SETTINGS, protocols
from scripts.run_assignment_sweep import jobs, frozen_config


def preflight(output):
    graph = json.loads((ROOT/'brain/calibration_data/leg_connectome.json').read_text())
    fit = json.loads((ROOT/'wasm/calibration/claw-cells-fit.json').read_text())
    for condition in ('connected', 'disconnected', 'sensory_off', 'motor_silenced'):
        old, new = LegCircuit(graph, seed=2), AssignedLegCircuit(graph, seed=2)
        for angle in np.r_[np.full(100, 100.), np.linspace(100, 130, 100), np.full(500, 130.), np.linspace(130, 80, 100)]:
            options = dict(connected=condition != 'disconnected', sensory_on=condition != 'sensory_off', motor_on=condition != 'motor_silenced')
            activation, _ = old.step(angle, 100., **options)
            vector = np.full(4, 150*np.clip((angle-100)/20, 0, 1))
            actual, _ = new.step_rates(vector, **options)
            np.testing.assert_array_equal(activation, actual)
        for name in ('voltage', 'current', 'refractory', 'queue', 'counts', 'rates'):
            np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
    for assignment in ASSIGNMENTS:
        for setting in SETTINGS:
            low = candidate_rates(50, assignment, setting, fit)
            high = candidate_rates(150, assignment, setting, fit)
            assert np.all(low >= 0) and np.all(high <= SETTINGS[setting]['max_hz'])
            for i, label in enumerate(assignment):
                assert (low[i] > high[i]) == (label == 'F')
    legacy = run_candidate(graph, fit, 'EEEE', 'nominal', ReflexProtocol(), legacy=True)
    published = json.loads((ROOT/'wasm/calibration/trials.json').read_text())['connected']
    assert legacy['motor_spikes'] == published['motor_spikes'] == 17
    assert legacy['spikes_by_body_id'] == published['spikes_by_body_id']
    np.testing.assert_allclose([r['angle_deg'] for r in legacy['trace']], [r['angle_deg'] for r in published['trace']], rtol=0, atol=1e-10)
    controls = {name: run_candidate(graph, fit, 'FEFE', 'gain_300', protocols()['extend_30'], condition=name)
                for name in ('disconnected', 'sensory_off', 'motor_silenced')}
    for name, trial in controls.items():
        assert trial['motor_spikes'] == 0
        np.testing.assert_allclose([r['angle_deg'] for r in trial['trace']], [r['angle_deg'] for r in controls['disconnected']['trace']], rtol=0, atol=1e-10)
    assert controls['sensory_off']['total_spikes'] == 0
    assert controls['motor_silenced']['total_spikes'] > 0
    checks = ['vector-input LIF exactly reproduces original states and spikes under all four controls',
              'all 16 assignments and five settings have correct per-cell tuning direction and bounded rates',
              'original physical default reproduces all body-ID spike counts and joint angles',
              'disconnected, sensory-off and motor-silenced mechanics match',
              'sensory-off network silent; motor-silenced network retains nonmotor activity']
    result = {'passed': True, 'checks': checks, 'legacy_motor_spikes': legacy['motor_spikes']}
    output.mkdir(parents=True, exist_ok=True)
    (output/'preflight.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def validate(output):
    config = json.loads((output/'config.json').read_text())
    expected = frozen_config()
    assert config['config_hash'] == hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    names = jobs()
    assert len(names) == 1066 and len({j['id'] for j in names}) == 1066
    index = {job['id']: job for job in names}
    assert {p.stem for p in (output/'trials').glob('*.json')} == set(index)
    for job in names:
        r = json.loads((output/'trials'/f'{job["id"]}.json').read_text())
        assert r['config_hash'] == config['config_hash']
        assert all(r[key] == job[key] for key in ('id', 'setting', 'assignment', 'seed', 'condition', 'protocol_name'))
        assert r['protocol'] == config['protocols'][job['protocol_name']]
        trace = r['trace']
        assert len(trace) == 320 and abs(trace[-1]['time_s']-1.596) < 1e-8
        for field in ('time_s', 'angle_deg', 'sensory_hz', 'flexor_hz', 'extensor_hz', 'sensory_drive_hz'):
            assert np.isfinite([p[field] for p in trace]).all()
        assert np.all(np.diff([p['time_s'] for p in trace]) > 0)
        if job['condition'] == 'disconnected':
            assert r['motor_spikes'] == 0
        assert sum(r['spikes_by_body_id'].values()) == r['total_spikes']
        for point in trace:
            assert np.all(np.asarray(point['sensory_drive_by_cell_hz']) >= 0)
            assert np.all(np.asarray(point['sensory_drive_by_cell_hz']) <= SETTINGS[job['setting']]['max_hz'])
    result = {'passed': True, 'trials_checked': len(names),
              'checks': ['complete Cartesian design and ten passive controls', 'all source/configuration hashes match',
                         'all clocks, rates and physical traces finite', 'disconnected motor populations silent',
                         'saved metadata, per-body counts and input bounds consistent'],
              'biological_mapping_selected': None}
    (output/'sweep-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/assignment-sweep')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    preflight(args.output) if args.preflight else validate(args.output)
