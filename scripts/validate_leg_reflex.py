"""Check scientific isolation, circuit provenance and measured causal controls."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from brain.leg_calibration import fit_observation, promotion_status
from brain.leg_reflex import LegFixture


def validate(directory=Path('outputs/leg-calibration')):
    graph = json.loads(Path('brain/calibration_data/leg_connectome.json').read_text())
    records = json.loads(Path('brain/calibration_data/leg_recordings.json').read_text())['imaging']
    report = json.loads((directory/'report.json').read_text())
    trials = json.loads((directory/'trials.json').read_text())
    neurons = {r['bodyId']: r for r in graph['neurons']}
    assert len(neurons) == len(graph['neurons'])
    assert all(e['weight'] > 0 and e['body_pre'] in neurons and e['body_post'] in neurons for e in graph['edges'])
    assert all(neurons[i]['rootSide'] == 'L' and neurons[i]['entryNerve'] == 'ProLN' for i in graph['sensory_ids'])
    assert all(neurons[i]['type'] == 'Ti flexor MN' for i in graph['flexor_ids'])
    assert all(neurons[i]['type'] == 'Ti extensor MN' for i in graph['extensor_ids'])
    assert trials['connected']['motor_spikes'] > 0
    assert report['causal_effect']['max_additional_flexion_deg'] > .1
    assert trials['disconnected']['motor_spikes'] == 0
    assert trials['sensory_off']['total_spikes'] == 0
    assert trials['motor_silenced']['motor_spikes'] == 0
    assert trials['motor_silenced']['total_spikes'] > 0
    disconnected = np.array([r['angle_deg'] for r in trials['disconnected']['trace']])
    for name in ('sensory_off', 'motor_silenced'):
        np.testing.assert_allclose([r['angle_deg'] for r in trials[name]['trace']], disconnected, atol=1e-10, rtol=0)
    original = fit_observation(records)
    poisoned = copy.deepcopy(records)
    for record in poisoned:
        if record['split'] != 'train':
            record['dff'] = [1000 if x is not None else None for x in record['dff']]
    altered = fit_observation(poisoned)
    np.testing.assert_array_equal(original['parameter_values'], altered['parameter_values'])
    assert not promotion_status(graph, original)['eligible']
    assert not report['promotion']['neural_parameters_fitted']
    # Anatomical angle and MJCF qpos have opposite orientations; test actual
    # muscle activation instead of assuming the sign from an actuator name.
    movements = []
    for action in ([0, 0], [.15, 0], [0, .15]):
        leg = LegFixture()
        try:
            for _ in range(100):
                leg.step(action)
            movements.append(leg.angle())
        finally:
            leg.close()
    assert movements[1] < movements[0] < movements[2], movements
    for trial in trials.values():
        assert np.isfinite([r['angle_deg'] for r in trial['trace']]).all()
        assert abs(trial['trace'][-1]['time_s']-(trial['duration_s']-.004)) < 1e-8
    result = {'passed': True, 'checks': ['left-leg anatomical selection', 'positive muscle directions',
        'circuit disconnection', 'sensory silencing', 'motor silencing', 'matched passive mechanics',
        'held-out data cannot affect fitted parameters', 'unverified cell mapping prevents promotion',
        'finite physics and matched clocks'], 'direct_muscle_final_angles_deg': movements}
    (directory/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    validate()
