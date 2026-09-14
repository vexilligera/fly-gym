"""Measured leg-response fitting, with explicit limits on parameter promotion."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


def observation(record, parameters, sample_rate=None):
    """Effective angle -> calcium response. No identification with CNS neurons.

    A phenomenological calibration reference only: its tau combines neural and
    indicator dynamics and its gain cannot be converted to absolute spike Hz.
    """
    angle = np.asarray(record['angle_deg'], dtype=float)
    observed = np.asarray(record['dff'], dtype=float)
    valid_angle = np.isfinite(angle)
    if not valid_angle.any():
        raise ValueError('No measured angles')
    index = np.arange(len(angle))
    filled = np.interp(index, index[valid_angle], angle[valid_angle])
    gain, threshold, width, tau = parameters
    drive = expit((filled - threshold) / width)
    filtered = np.empty_like(drive)
    filtered[0] = drive[0]
    decay = np.exp(-1 / ((sample_rate or record['sample_rate_hz']) * tau))
    for i in range(1, len(drive)):
        filtered[i] = decay * filtered[i-1] + (1-decay) * drive[i]
    # Baseline frames end before the first >3 degree change in the source angle.
    first = int(np.flatnonzero(valid_angle)[0])
    changes = np.flatnonzero(np.abs(filled - filled[first]) > 3)
    end = int(changes[0]) if len(changes) else min(first + 20, len(angle))
    baseline = np.arange(first, max(first+1, end))
    baseline = baseline[np.isfinite(observed[baseline])]
    if not len(baseline):
        raise ValueError('No pre-movement fluorescence baseline')
    prediction = gain * (filtered - np.mean(filtered[baseline]))
    target = observed - np.mean(observed[baseline])
    mask = valid_angle & np.isfinite(target)
    return prediction, target, mask


def fit_observation(records, sample_rate=7.57):
    training = [r for r in records if r['split'] == 'train']
    if not training:
        raise ValueError('No training records')
    def residual(parameters):
        out = []
        for record in training:
            predicted, target, valid = observation(record, parameters, sample_rate)
            out.append((predicted[valid]-target[valid]) / np.sqrt(valid.sum()))
        return np.concatenate(out)
    lower = [.001, 0, 2, .02]
    upper = [10, 180, 100, 10]
    starts = ([1, 80, 25, .5], [2, 120, 15, 1], [.5, 40, 40, .2])
    fits = [least_squares(residual, x, bounds=(lower, upper), max_nfev=300) for x in starts]
    result = min(fits, key=lambda x: np.sum(x.fun**2))
    scores = []
    for record in records:
        predicted, target, valid = observation(record, result.x, sample_rate)
        mse = float(np.mean((predicted[valid]-target[valid])**2))
        baseline = float(np.mean(target[valid]**2))
        scores.append({'id': record['id'], 'split': record['split'],
                       'mse': mse, 'zero_response_mse': baseline,
                       'relative_mse': mse / max(baseline, 1e-12),
                       'n_samples': int(valid.sum())})
    bounds = [name for name, x, lo, hi in zip(
        ('fluorescence_gain', 'angle_midpoint_deg', 'angle_width_deg', 'effective_tau_s'),
        result.x, lower, upper) if min(x-lo, hi-x) < .001*(hi-lo)]
    return {'parameters': dict(zip(
        ('fluorescence_gain', 'angle_midpoint_deg', 'angle_width_deg', 'effective_tau_s'),
        map(float, result.x))), 'parameter_values': result.x.tolist(),
        'sample_rate_hz': sample_rate, 'scores': scores, 'parameters_at_bound': bounds,
        'optimizer_success': bool(result.success),
        'meaning': 'Effective angle-to-GCaMP response fit; not a connectome, firing-rate or muscle calibration.'}


def promotion_status(connectome, observation_fit):
    """Fail closed until comparable neuron and motor-unit identities exist."""
    reasons = []
    if not connectome['mapping'].get('13Balpha'):
        reasons.append('Recorded 13Balpha cells have no verified match to MaleCNS body IDs.')
    reasons.append('No measured spike-to-force calibration for the pooled muscle adapter.')
    reasons.append('Individual claw-neuron angle tuning and omitted circuit inputs remain unresolved.')
    if observation_fit['parameters_at_bound']:
        reasons.append('An effective observation parameter reached its fit bound.')
    if any(s['relative_mse'] >= 1 for s in observation_fit['scores'] if s['split'] == 'protocol_test'):
        reasons.append('The frozen observation fit fails to beat zero response on several held-out swing records.')
    return {'eligible': False, 'reasons': reasons,
            'live_maze_changed': False, 'neural_parameters_fitted': False}
