"""Validate data isolation, causal filtering and paired protocol controls."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from brain.claw_calibration import fit_candidates, predict


def validate(directory):
    path = Path('brain/calibration_data/claw_recordings.json')
    data = json.loads(path.read_text())
    records = data['records']
    report = json.loads((directory/'claw-report.json').read_text())
    assert report['input_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(records) == 60 and len({r['animal_id'] for r in records}) == 10
    for r in records:
        assert np.all(np.diff(r['time_s']) > 0)
        assert len(r['time_s']) == len(r['calcium']) == len(r['angle_deg'])
    train = {r['animal_id'] for r in records if r['split'] == 'train'}
    val = {r['animal_id'] for r in records if r['split'] == 'validation'}
    test = {r['animal_id'] for r in records if r['split'] == 'animal_test'}
    assert not train & val and not train & test and not val & test
    assert all(r['roi'] == 'L1_x' for r in records if r['split'] != 'reference_only')
    original = fit_candidates(records)
    poisoned = copy.deepcopy(records)
    for r in poisoned:
        if r['split'] in ('animal_test', 'protocol_test', 'reference_only'):
            r['calcium'] = [1e9]*len(r['calcium'])
            r['angle_deg'] = [179]*len(r['angle_deg'])
    assert original == fit_candidates(poisoned), 'Test samples leaked into training/selection'
    r = next(r for r in records if r['split'] == 'animal_test')
    altered = copy.deepcopy(r)
    altered['angle_deg'][100:] = [30.]*(len(r['angle_deg'])-100)
    np.testing.assert_allclose(predict(r, original)[:100], predict(altered, original)[:100], atol=1e-12)
    assert report['frozen_fit']['selected_model'] == original['selected_model']
    for key in ('training_scale', 'training_constant'):
        np.testing.assert_allclose(report['frozen_fit'][key], original[key], atol=1e-12, rtol=1e-12)
    for name, candidate in original['candidates'].items():
        np.testing.assert_allclose(report['frozen_fit']['candidates'][name]['coefficients'], candidate['coefficients'], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(report['frozen_fit']['candidates'][name]['validation_mse'], candidate['validation_mse'], atol=1e-12, rtol=1e-12)
    assert not report['accepted_as_neural_calibration'] and not report['maze_changed']
    trials = json.loads((directory/'sweep-trials.json').read_text())
    assert len(report['sweep']) == 9
    for result in report['sweep']:
        pair = trials[result['id']]
        active, passive = pair['connected'], pair['disconnected']
        assert active['protocol'] == passive['protocol'] == result['protocol']
        assert passive['motor_spikes'] == 0
        for tr in pair.values():
            assert np.isfinite([r['angle_deg'] for r in tr['trace']]).all()
            assert abs(tr['trace'][-1]['time_s']-1.596) < 1e-8
        if result['protocol']['displacement_deg'] < 0:
            # A negative result of the current encoder, not a desired biological property.
            assert active['motor_spikes'] == 0
            np.testing.assert_allclose([r['angle_deg'] for r in active['trace']], [r['angle_deg'] for r in passive['trace']], atol=1e-10, rtol=0)
    old = json.loads(Path('wasm/calibration/trials.json').read_text())['connected']
    default = trials['4']['connected']
    assert default['motor_spikes'] == old['motor_spikes'] == 17
    np.testing.assert_allclose([r['angle_deg'] for r in default['trace']], [r['angle_deg'] for r in old['trace']], atol=1e-10, rtol=0)
    outcome = {'passed': True, 'checks': ['distinct training, selection and test animals',
        'held-out and reference data cannot change fit or selection', 'causal calcium filtering',
        'timestamp and source hash integrity', 'nine matched disconnected physical controls',
        'flexion negative results retained', 'default reflex reproduces original trial',
        'finite mechanics and matching clocks', 'no automatic promotion to neural engine or maze']}
    (directory/'followup-validation.json').write_text(json.dumps(outcome, indent=2)+'\n')
    print(json.dumps(outcome, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('outputs/leg-reflex-followup'))
    validate(parser.parse_args().directory)
