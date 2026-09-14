"""Verify source integrity, held-out isolation, causal predictions and scores."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from brain.claw_cell_calibration import load_records, fit_models, evaluate, predict, weights, PRIMARY
from scripts.prepare_claw_cell_recordings import read_source, NumericUnpickler


def equivalent(first, second):
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equivalent(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for x, y in zip(first, second):
            equivalent(x, y)
    elif isinstance(first, (int, float)) and not isinstance(first, bool):
        np.testing.assert_allclose(first, second, rtol=1e-7, atol=1e-8)
    else:
        assert first == second, (first, second)


def validate(output):
    directory = ROOT/'brain/calibration_data'
    metadata, records = load_records(directory)
    report = json.loads((output/'claw-cells-report.json').read_text())
    assert len(records) == 262
    assert sum(r['cohort'] in PRIMARY for r in records) == 156
    assert len({r['animal_id'] for r in records if r['cohort'] in PRIMARY}) == 13
    for name, digest in report['input_sha256'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    assert report['fit_sha256'] == hashlib.sha256((output/'claw-cells-fit.json').read_bytes()).hexdigest()
    for r in records:
        assert len(r['time_s']) == len(r['calcium']) == len(r['angle_deg'])
        assert np.isfinite(r['calcium']).all() and np.isfinite(r['angle_deg']).all()
        assert np.all(np.diff(r['time_s']) > 0)
    for cohort in PRIMARY:
        splits = {s: {r['animal_id'] for r in records if r['cohort'] == cohort and r['split'] == s}
                  for s in ('train', 'validation', 'animal_test', 'protocol_test')}
        assert not splits['train'] & splits['validation']
        assert not (splits['train'] | splits['validation']) & splits['animal_test']
        assert splits['protocol_test'] == splits['train'] | splits['validation']
        assert all(r['order'] == 2 for r in records if r['cohort'] == cohort and r['split'] in ('train', 'validation'))
        training = [r for r in records if r['cohort'] == cohort and r['split'] == 'train']
        weight = weights(training)
        for animal in splits['train']:
            contribution = sum(w*w*len(r['time_s']) for w, r in zip(weight, training) if r['animal_id'] == animal)
            np.testing.assert_allclose(contribution, 1/len(splits['train']))
    # Poison every held-out/reference stimulus and response. Refitting and
    # validation selection must remain numerically identical to the saved fit.
    poisoned = copy.deepcopy(records)
    for r in poisoned:
        if r['split'] in ('animal_test', 'protocol_test', 'reference_only'):
            r['calcium'][:] = 1e6
            r['angle_deg'][:] = 179
    fitted = fit_models(poisoned)
    equivalent(report['frozen_fit'], fitted)
    for cohort in PRIMARY:
        r = next(r for r in records if r['cohort'] == cohort and r['split'] == 'animal_test')
        altered = copy.deepcopy(r)
        cut = len(r['time_s'])//2
        altered['angle_deg'][cut:] = 30
        np.testing.assert_allclose(predict(r, fitted)[:cut], predict(altered, fitted)[:cut], atol=1e-12)
    equivalent(report['evaluation'], evaluate(records, report['frozen_fit']))
    # Independently recompute equal-fly aggregation, rather than pooling cells.
    for cohort in PRIMARY:
        rows = [r for r in report['evaluation']['fly_scores'] if r['cohort'] == cohort and r['split'] == 'animal_test']
        ratio = np.mean([r['mse'] for r in rows])/np.mean([r['constant_mse'] for r in rows])
        np.testing.assert_allclose(ratio, report['evaluation']['groups'][cohort]['animal_test']['relative_mse'])
    try:
        NumericUnpickler(io.BytesIO(b'cos\nsystem\n.')).load()
    except pickle.UnpicklingError:
        pass
    else:
        raise AssertionError('Non-numeric pickle global accepted')
    raw_path = ROOT/'outputs/leg-calibration-source/mamiya2023'/metadata['source_file']
    source_checked = False
    if raw_path.exists():
        source, _ = read_source(raw_path)
        with np.load(directory/'claw_cells.npz', allow_pickle=False) as imported:
            assert set(source) == set(imported.files)
            for key in source:
                np.testing.assert_array_equal(source[key], imported[key])
        source_checked = True
    assert report['mapping']['verified_recording_matches'] == 0
    assert all(v is None for v in report['mapping']['assigned_tuning'].values())
    assert not report['accepted_as_neural_calibration'] and not report['maze_changed']
    prior = json.loads((ROOT/'wasm/calibration/claw-report.json').read_text())
    assert hashlib.sha256((ROOT/'brain/leg_reflex.py').read_bytes()).hexdigest() == prior['implementation_sha256']['brain/leg_reflex.py']
    result = {'passed': True, 'raw_archive_round_trip_checked': source_checked,
              'checks': ['source and fit hashes', '262 finite region traces with original timestamps',
                         'whole-fly split separation and correct reverse-order transfer group',
                         'equal weight per fly, then per region',
                         'poisoning test and reference inputs/outputs cannot alter fitting or selection',
                         'future joint angles cannot affect earlier predictions',
                         'all region scores and equal-fly aggregates recomputed',
                         'unsupported pickle globals rejected',
                         'no unverified EM tuning assignment or physiological promotion',
                         'original physical reflex implementation unchanged']}
    (output/'claw-cells-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/claw-cell-validation')
    validate(parser.parse_args().output)
