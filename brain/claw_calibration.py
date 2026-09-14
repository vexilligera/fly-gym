"""Animal-separated calcium observation benchmark for pooled claw axons.

No parameters from this benchmark are interpreted as spikes or promoted to the
MaleCNS leg circuit. The split and hypotheses are in REFLEX_FOLLOWUP.md.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

MODELS = ('angle', 'angle_history', 'legacy_encoder')
KERNEL = {'rise_s': .03, 'decay_s': .30}


def design(record, model, shift_s=0.):
    time = np.asarray(record['time_s'], dtype=float)
    angle = np.asarray(record['angle_deg'], dtype=float)
    valid = np.isfinite(angle)
    if not valid.any():
        raise ValueError('No valid angle samples')
    angle = np.interp(time, time[valid], angle[valid])
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, rtol=1e-3):
        raise ValueError('Irregular sampling needs explicit time-domain filtering')
    if shift_s:
        angle = np.roll(angle, round(shift_s/dt))
    x = (angle-90)/90
    if model == 'legacy_encoder':
        basis = np.column_stack([np.ones(len(x)), np.clip((angle-100)/20, 0, 1)])
    else:
        basis = np.column_stack([x**i for i in range(5)])
        if model == 'angle_history':
            direction = np.zeros(len(x))
            velocity = np.r_[0, np.diff(angle)/dt]
            for i in range(1, len(x)):
                direction[i] = np.sign(velocity[i]) if abs(velocity[i]) > 5 else direction[i-1]
            basis = np.column_stack([basis] + [direction*x**i for i in range(3)])
        elif model != 'angle':
            raise ValueError('Unknown observation model')
    t = np.arange(0, 10*KERNEL['decay_s']+dt, dt)
    kernel = np.exp(-t/KERNEL['decay_s'])-np.exp(-t/KERNEL['rise_s'])
    kernel /= kernel.sum()
    # Constant prehistory at the initial feature value; no look-ahead.
    padded = np.vstack([np.repeat(basis[:1], len(kernel)-1, axis=0), basis])
    result = fftconvolve(padded, kernel[:, None], mode='full', axes=0)
    return result[len(kernel)-1:len(kernel)-1+len(basis)]


def mask(record):
    return (np.asarray(record['analyze']) == 1) & np.isfinite(np.asarray(record['calcium'], dtype=float)) & np.isfinite(np.asarray(record['angle_deg'], dtype=float))


def fit_candidates(records):
    """Read training targets and validation targets only; never score tests here."""
    training = [r for r in records if r['split'] == 'train']
    validation = [r for r in records if r['split'] == 'validation']
    if not training or not validation:
        raise ValueError('Training and validation animals are required')
    scale = float(max(np.asarray(r['calcium'], dtype=float)[mask(r)].max() for r in training))
    if scale <= 0:
        raise ValueError('Expected a positive training calcium maximum')
    constant = float(np.mean([np.asarray(r['calcium'])[mask(r)].mean()/scale for r in training]))
    candidates = {}
    for model in MODELS:
        xs, ys = [], []
        for r in training:
            valid = mask(r)
            xs.append(design(r, model)[valid]/np.sqrt(valid.sum()))
            ys.append(np.asarray(r['calcium'])[valid]/scale/np.sqrt(valid.sum()))
        coeff = np.linalg.lstsq(np.vstack(xs), np.concatenate(ys), rcond=None)[0]
        errors = []
        for r in validation:
            valid = mask(r)
            errors.append(float(np.mean((design(r, model)[valid]@coeff-np.asarray(r['calcium'])[valid]/scale)**2)))
        candidates[model] = {'coefficients': coeff.tolist(), 'validation_mse': float(np.mean(errors))}
    # Legacy encoder is an explanatory comparator, not a candidate to promote.
    selected = min(('angle', 'angle_history'), key=lambda name: candidates[name]['validation_mse'])
    return {'selected_model': selected, 'candidates': candidates,
            'training_scale': scale, 'training_constant': constant, 'kernel': KERNEL,
            'train_animals': sorted({r['animal_id'] for r in training}),
            'validation_animals': sorted({r['animal_id'] for r in validation}),
            'neural_parameters_fitted': False, 'promoted_to_circuit': False}


def predict(record, fit, model=None, shift_s=0.):
    model = model or fit['selected_model']
    return design(record, model, shift_s)@np.asarray(fit['candidates'][model]['coefficients'])


def evaluate(records, fit):
    scores = []
    for r in records:
        if r['split'] == 'reference_only':
            continue
        valid = mask(r)
        target = np.asarray(r['calcium'], dtype=float)[valid]/fit['training_scale']
        baseline_mse = float(np.mean((target-fit['training_constant'])**2))
        row = {k: r[k] for k in ('id', 'animal_id', 'protocol', 'split')}
        row.update(n_samples=int(valid.sum()), constant_mse=baseline_mse, models={})
        for name in MODELS:
            prediction = predict(r, fit, name)[valid]
            mse = float(np.mean((prediction-target)**2))
            correlation = np.corrcoef(prediction, target)[0, 1]
            row['models'][name] = {'mse': mse, 'relative_mse': mse/max(baseline_mse, 1e-12),
                                   'pearson_r': float(correlation) if np.isfinite(correlation) else None}
        shifted = predict(r, fit, shift_s=1.)[valid]
        row['shifted_input_mse'] = float(np.mean((shifted-target)**2))
        scores.append(row)
    groups = {}
    for split in ('train', 'validation', 'animal_test', 'protocol_test'):
        selected = [s for s in scores if s['split'] == split]
        model = fit['selected_model']
        groups[split] = {'trials': len(selected), 'animals': len({s['animal_id'] for s in selected}),
            'relative_mse': float(np.mean([s['models'][model]['mse'] for s in selected])/np.mean([s['constant_mse'] for s in selected])),
            'median_pearson_r': float(np.median([s['models'][model]['pearson_r'] for s in selected])),
            'trials_beating_constant': sum(s['models'][model]['relative_mse'] < 1 for s in selected)}
    return {'scores': scores, 'groups': groups}
