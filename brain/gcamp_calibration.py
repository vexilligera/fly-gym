"""Offline GCaMP observation model and grouped train/test calibration.

Fluorescence scaling/decay are measurement parameters, not neuronal physiology.
The candidate neural parameters are selected separately by training loss only.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter

DEFAULT_DATA = Path(__file__).parent/'calibration_data/shiu2022_taste.json'


def load_data(path=DEFAULT_DATA):
    data = json.loads(Path(path).read_text())
    if data['schema'] != 'gcamp-taste-pilot-v1':
        raise ValueError('Unsupported GCaMP dataset')
    for cell in data['cells']:
        t = np.asarray(cell['time_s'], float)
        if not np.isfinite(t).all() or not (np.diff(t) > 0).all():
            raise ValueError('Observation times must be finite and increasing')
        ownership = {}
        for trace in cell['traces']:
            if len(trace['dff']) != len(t):
                raise ValueError('Trace length differs from observation times')
            if trace['split'] not in ('train', 'test'):
                raise ValueError('Unknown data split')
            if trace['fly_id'] in ownership and ownership[trace['fly_id']] != trace['split']:
                raise ValueError('One fly appears in both training and test data')
            ownership[trace['fly_id']] = trace['split']
    return data


def calcium_response(rates_hz, dt_s, tau_s):
    """Causal unit-DC-gain low-pass observation model, sampled at bin ends.

    This effective calcium/indicator kernel is an approximation. It does not
    assert that fluorescence is a direct voltage or spike-count measurement.
    """
    rates = np.asarray(rates_hz, float)
    if rates.ndim != 1 or not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError('Expected finite, nonnegative firing rates')
    if not np.isfinite([dt_s, tau_s]).all() or dt_s <= 0 or tau_s <= 0:
        raise ValueError('Sampling interval and decay must be positive')
    decay = np.exp(-dt_s/tau_s)
    return lfilter([1-decay], [1, -decay], rates)


def observations(data, cell, split):
    baseline = data['protocol']['baseline_frames']
    rows, ids = [], []
    for trace in cell['traces']:
        if trace['tastant'] != 'sugar' or trace['split'] != split:
            continue
        y = np.array([np.nan if v is None else v for v in trace['dff']], float)
        bl = y[baseline[0]:baseline[1]+1]
        if not np.isfinite(bl).all():
            raise ValueError('Missing pre-stimulus baseline')
        # Per-trial baseline uses only pre-stimulus data, including for test flies.
        rows.append(y-np.mean(bl))
        ids.append(trace['fly_id'])
    if len(rows) < 2:
        raise ValueError('Need at least two flies in each split')
    return np.array(rows), ids


def evaluation_mask(data, cell):
    t = np.asarray(cell['time_s'])
    lo, hi = data['protocol']['evaluation_window_s']
    return (t >= lo-1e-9) & (t <= hi+1e-9)


def predicted_trace(data, cell, simulation, tau_s):
    times = np.asarray(simulation['time_s'])
    dt = float(simulation['dt_s'])
    if len(times) < 2 or not np.allclose(np.diff(times), dt, rtol=0, atol=1e-8):
        raise ValueError('Simulation must contain uniformly sampled bins')
    rates = np.asarray(simulation['rates_hz'][cell['name']], float)
    if len(rates) != len(times):
        raise ValueError('Rates and simulation time differ in length')
    calcium = calcium_response(rates, dt, tau_s)
    query = np.asarray(cell['time_s']) - data['protocol']['stimulus_on_s'] + simulation['stimulus_on_s']
    selected = evaluation_mask(data, cell)
    if query[selected].max() > times[-1]+1e-8:
        raise ValueError('Simulation does not cover the evaluation window')
    return np.interp(query, times, calcium, left=0, right=calcium[-1])


def fit_observation(data, cell, simulation):
    y, _ = observations(data, cell, 'train')
    mask = evaluation_mask(data, cell)
    valid = np.isfinite(y) & mask[None, :]
    # This scale is computed from training data and reused for held-out errors.
    norm = max(.05, float(np.sqrt(np.mean(y[valid]**2))))

    def evaluate(log_tau):
        tau = float(np.exp(log_tau))
        x = predicted_trace(data, cell, simulation, tau)
        xx = np.broadcast_to(x, y.shape)[valid]
        yy = y[valid]
        gain = max(0., float(xx@yy / (xx@xx))) if xx@xx > 1e-20 else 0.
        loss = float(np.mean((yy-gain*xx)**2)/norm**2)
        return loss, gain, tau

    low, high = np.log(.2), np.log(10.)
    optimum = minimize_scalar(lambda v: evaluate(v)[0], bounds=(low, high), method='bounded')
    loss, gain, tau = min((evaluate(v) for v in [low, optimum.x, high]), key=lambda x: x[0])
    return {'effective_decay_s': tau, 'fluorescence_gain': gain,
            'training_normalizer': norm, 'train_nmse': loss,
            'decay_at_bound': tau <= .201 or tau >= 9.99}


def score(data, cell, simulation, observation, split):
    pred = observation['fluorescence_gain'] * predicted_trace(data, cell, simulation, observation['effective_decay_s'])
    y, ids = observations(data, cell, split)
    mask = evaluation_mask(data, cell)
    errors = []
    for fly, row in zip(ids, y):
        valid = mask & np.isfinite(row)
        mse = float(np.mean((row[valid]-pred[valid])**2))
        errors.append({'fly_id': fly, 'nmse': mse/observation['training_normalizer']**2})
    return {'nmse': float(np.mean([e['nmse'] for e in errors])), 'flies': errors,
            'prediction_dff': pred.tolist()}


def fit_candidates(data, simulations):
    fitted = []
    for simulation in simulations:
        cells = {}
        for cell in data['cells']:
            obs = fit_observation(data, cell, simulation)
            cells[cell['name']] = {'observation': obs,
                                  'train': score(data, cell, simulation, obs, 'train')}
        fitted.append({'candidate': simulation['candidate'], 'cells': cells,
                       'train_nmse': float(np.mean([c['train']['nmse'] for c in cells.values()]))})
    # Freeze selection before any held-out response is scored.
    selected = int(np.argmin([f['train_nmse'] for f in fitted]))
    for fitted_case, simulation in zip(fitted, simulations):
        for cell in data['cells']:
            entry = fitted_case['cells'][cell['name']]
            entry['test'] = score(data, cell, simulation, entry['observation'], 'test')
        fitted_case['test_nmse'] = float(np.mean([c['test']['nmse'] for c in fitted_case['cells'].values()]))
    return fitted, selected


def paired_bootstrap(baseline, selected, replicates=5000):
    """Resample held-out flies within cell types, weighting each type equally."""
    differences = []
    for name, base in baseline['cells'].items():
        old = {x['fly_id']: x['nmse'] for x in base['test']['flies']}
        new = {x['fly_id']: x['nmse'] for x in selected['cells'][name]['test']['flies']}
        if old.keys() != new.keys():
            raise ValueError('Unpaired validation flies')
        differences.append(np.array([old[k]-new[k] for k in sorted(old)]))
    rng = np.random.default_rng(783)
    samples = np.mean([rng.choice(d, (replicates, len(d))).mean(axis=1) for d in differences], axis=0)
    return {'metric': 'baseline minus selected normalized MSE; positive favors selected',
            'mean': float(np.mean([d.mean() for d in differences])),
            'interval_95': np.quantile(samples, [.025, .975]).tolist(),
            'replicates': replicates, 'warning': 'Small held-out sample; interval covers fly variability, not model or protocol uncertainty.'}
