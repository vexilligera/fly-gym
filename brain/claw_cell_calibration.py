"""Fly-separated observation models for Mamiya 2023 claw cell-body calcium.

These functions do not assign physiology to MaleCNS neurons or convert calcium
to firing rates. See CLAW_CELL_VALIDATION.md for the fixed analysis protocol.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.signal import lfilter
from scipy.special import expit

TAUS = (0., .3, 1., 3.)
PRIMARY = ('JR209', 'JR688')
PARAMETER_NAMES = ('offset_drr', 'gain_drr', 'midpoint_deg', 'width_deg')
LOWER = np.array([-2., 0., 0., 1.])
UPPER = np.array([5., 20., 180., 90.])


def load_records(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'claw_cells.json').read_text())
    path = directory/'claw_cells.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['npz_sha256']:
        raise ValueError('Imported data hash mismatch')
    records = []
    with np.load(path, allow_pickle=False) as arrays:
        for file in metadata['files']:
            key, cohort, order = file['key'], file['cohort'], file['order']
            info = metadata['cohorts'][cohort]
            for i, fly in enumerate(arrays[f'{key}_fly_id'][0]):
                fly = int(fly)
                if info.get('reference_only'):
                    split = 'reference_only'
                elif fly in info['animal_test']:
                    split = 'animal_test'
                elif order == 3:
                    split = 'protocol_test'
                else:
                    split = 'train' if fly in info['train'] else 'validation'
                records.append({'id': f'{key}_region{i+1}', 'cohort': cohort,
                                'fly': fly, 'animal_id': f'{cohort}:{fly}',
                                'region': i+1, 'order': order, 'split': split,
                                'time_s': arrays[f'{key}_time_s'],
                                'angle_deg': arrays[f'{key}_angle_deg'][:, i],
                                'calcium': arrays[f'{key}_drr'][:, i],
                                'x_position': arrays[f'{key}_x_position'][:, i],
                                'y_position': arrays[f'{key}_y_position'][:, i]})
    return metadata, records


def steady(angle, parameters, cohort):
    offset, gain, midpoint, width = parameters
    sign = -1 if cohort == 'JR209' else 1
    return offset + gain*expit(sign*(np.asarray(angle)-midpoint)/width)


def relax(signal, time, tau):
    """Causal first-order observation filter, at the source sampling interval."""
    signal, time = np.asarray(signal), np.asarray(time)
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, rtol=1e-3, atol=1e-9):
        raise ValueError('Expected regular source sampling')
    if tau == 0:
        return signal.copy()
    alpha = -np.expm1(-dt/tau)
    zi = np.expand_dims((1-alpha)*signal[0], axis=0)
    return lfilter([alpha], [1, -(1-alpha)], signal, axis=0, zi=zi)[0]


def weights(records):
    counts = Counter(r['animal_id'] for r in records)
    return np.array([1/np.sqrt(len(counts)*counts[r['animal_id']]*len(r['time_s'])) for r in records])


def animal_mse(records, predictions):
    values = {}
    for r, prediction in zip(records, predictions):
        values.setdefault(r['animal_id'], []).append(float(np.mean((prediction-r['calcium'])**2)))
    return float(np.mean([np.mean(v) for v in values.values()]))


def fit_models(records):
    result = {'cohorts': {}, 'neural_parameters_fitted': False, 'promoted_to_circuit': False}
    for cohort in PRIMARY:
        train = [r for r in records if r['cohort'] == cohort and r['split'] == 'train']
        validation = [r for r in records if r['cohort'] == cohort and r['split'] == 'validation']
        if not train or not validation:
            raise ValueError('Training and validation flies required')
        time = train[0]['time_s']
        assert all(np.array_equal(r['time_s'], time) for r in train)
        angles = np.column_stack([r['angle_deg'] for r in train])
        target = np.column_stack([r['calcium'] for r in train])
        weight = weights(train)
        constant = float(sum(np.sum(r['calcium'])*w*w for r, w in zip(train, weight)))
        # Same old encoder, with a nonnegative fluorescence gain. A negative
        # gain would silently turn this comparator into a flexion sensor.
        old = np.clip((angles-100)/20, 0, 1)
        design = np.stack([np.ones_like(old), old], axis=-1)
        legacy = lsq_linear((design*weight[None, :, None]).reshape(-1, 2),
                            (target*weight).ravel(), bounds=([-np.inf, 0], [np.inf, np.inf]))
        candidates = []
        for tau in TAUS:
            def residual(parameters):
                prediction = relax(steady(angles, parameters, cohort), time, tau)
                return ((prediction-target)*weight).ravel()
            fits = [least_squares(residual, [0, 2, midpoint, 15], bounds=(LOWER, UPPER),
                                  max_nfev=500, ftol=1e-9, xtol=1e-9, gtol=1e-9)
                    for midpoint in (45., 90., 135.)]
            fitted = min(fits, key=lambda x: x.cost)
            if not fitted.success or not np.isfinite(fitted.x).all():
                raise RuntimeError(f'Fit failed for {cohort}, tau={tau}')
            candidate = {'tau_s': tau, 'parameters': fitted.x.tolist(),
                         'training_mse': float(2*fitted.cost),
                         'parameters_at_bound': [name for name, x, low, high in zip(PARAMETER_NAMES, fitted.x, LOWER, UPPER)
                                                 if abs(x-low) < 1e-3 or abs(x-high) < 1e-3]}
            predictions = [relax(steady(r['angle_deg'], fitted.x, cohort), r['time_s'], tau) for r in validation]
            candidate['validation_mse'] = animal_mse(validation, predictions)
            candidates.append(candidate)
        chosen = min(candidates, key=lambda c: c['validation_mse'])
        result['cohorts'][cohort] = {'selected': chosen, 'candidates': candidates,
                                   'constant_drr': constant, 'legacy_coefficients': legacy.x.tolist(),
                                   'train_flies': sorted({r['animal_id'] for r in train}),
                                   'validation_flies': sorted({r['animal_id'] for r in validation}),
                                   'parameter_names': PARAMETER_NAMES}
    return result


def predict(record, fit, model='separate'):
    cohort = record['cohort']
    if model == 'wrong_driver':
        cohort = next(c for c in PRIMARY if c != cohort)
    values = fit['cohorts'][cohort]
    if model == 'constant':
        return np.full(len(record['time_s']), values['constant_drr'])
    if model == 'legacy':
        offset, gain = values['legacy_coefficients']
        return offset + gain*np.clip((record['angle_deg']-100)/20, 0, 1)
    chosen = values['selected']
    return relax(steady(record['angle_deg'], chosen['parameters'], cohort), record['time_s'], chosen['tau_s'])


def describe_tuning(record):
    low = record['angle_deg'] <= 60
    high = record['angle_deg'] >= 120
    low_mean = float(record['calcium'][low].mean()) if low.any() else None
    high_mean = float(record['calcium'][high].mean()) if high.any() else None
    contrast = low_mean-high_mean if low_mean is not None and high_mean is not None else None
    return {'low_angle_mean_drr': low_mean, 'high_angle_mean_drr': high_mean,
            'low_minus_high_drr': contrast,
            'observed_preference': 'low_angle' if contrast is not None and contrast > 0 else 'high_angle'}


def evaluate(records, fit):
    scores = []
    for r in records:
        if r['cohort'] not in PRIMARY:
            continue
        row = {k: r[k] for k in ('id', 'cohort', 'fly', 'animal_id', 'region', 'order', 'split')}
        row['tuning'] = describe_tuning(r)
        row['models'] = {}
        for model in ('separate', 'constant', 'legacy', 'wrong_driver'):
            prediction = predict(r, fit, model)
            correlation = np.corrcoef(prediction, r['calcium'])[0, 1] if np.std(prediction) > 1e-10 and np.std(r['calcium']) > 1e-10 else None
            row['models'][model] = {'mse': float(np.mean((prediction-r['calcium'])**2)),
                                    'pearson_r': float(correlation) if correlation is not None else None}
        baseline = row['models']['constant']['mse']
        for model in row['models'].values():
            model['relative_mse'] = model['mse']/max(baseline, 1e-12)
        scores.append(row)
    by_fly = []
    for animal, split in sorted({(s['animal_id'], s['split']) for s in scores}):
        rows = [s for s in scores if s['animal_id'] == animal and s['split'] == split]
        baseline = float(np.mean([r['models']['constant']['mse'] for r in rows]))
        mse = float(np.mean([r['models']['separate']['mse'] for r in rows]))
        by_fly.append({'animal_id': animal, 'cohort': rows[0]['cohort'], 'split': split,
                       'region_traces': len(rows), 'mse': mse, 'constant_mse': baseline,
                       'relative_mse': mse/max(baseline, 1e-12),
                       'legacy_mse': float(np.mean([r['models']['legacy']['mse'] for r in rows])),
                       'wrong_driver_mse': float(np.mean([r['models']['wrong_driver']['mse'] for r in rows])),
                       'median_region_r': float(np.median([r['models']['separate']['pearson_r'] for r in rows if r['models']['separate']['pearson_r'] is not None])),
                       'region_traces_beating_constant': sum(r['models']['separate']['relative_mse'] < 1 for r in rows)})
    groups = {}
    for cohort in PRIMARY:
        groups[cohort] = {}
        for split in ('train', 'validation', 'animal_test', 'protocol_test'):
            rows = [r for r in by_fly if r['cohort'] == cohort and r['split'] == split]
            baseline = float(np.mean([r['constant_mse'] for r in rows]))
            groups[cohort][split] = {'flies': len(rows), 'region_traces': sum(r['region_traces'] for r in rows),
                'relative_mse': float(np.mean([r['mse'] for r in rows])/max(baseline, 1e-12)),
                'legacy_relative_mse': float(np.mean([r['legacy_mse'] for r in rows])/max(baseline, 1e-12)),
                'wrong_driver_relative_mse': float(np.mean([r['wrong_driver_mse'] for r in rows])/max(baseline, 1e-12)),
                'flies_beating_constant': sum(r['relative_mse'] < 1 for r in rows),
                'median_fly_correlation': float(np.median([r['median_region_r'] for r in rows]))}
    return {'region_scores': scores, 'fly_scores': by_fly, 'groups': groups}
