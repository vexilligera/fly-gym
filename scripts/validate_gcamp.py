"""Check causal fluorescence conversion, grouped fitting, and data provenance.

Pass --cuda on an allocated GPU to check gain changes and exact baseline recovery.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.gcamp_calibration import load_data, calcium_response, fit_candidates, predicted_trace


def validate():
    data = load_data()
    assert len(data['cells']) == 3
    assert sum(len(c['traces']) for c in data['cells']) == 63
    x = np.r_[np.zeros(10), 100., np.zeros(20)]
    y = calcium_response(x, .02, .7)
    assert not y[:10].any() and y[10] > 0
    np.testing.assert_allclose(y[11:]/y[10:-1], np.exp(-.02/.7))
    for bad in ([np.nan], [-1.], [[1.]]):
        try:
            calcium_response(bad, .02, .7)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid rates accepted')
    # Synthetic fixture tests only numerical recovery; it is never used as data.
    fixture = deepcopy(data)
    times = np.arange(1, 1101)*.02
    rates = np.where((times > 1.2) & (times <= 8.4), 40., 0.)
    simulation = {'candidate': {'synaptic_gain': 1., 'sugar_input_hz': 100.},
                  'dt_s': .02, 'time_s': times.tolist(), 'stimulus_on_s': 1.2,
                  'rates_hz': {c['name']: rates.tolist() for c in fixture['cells']}}
    for c in fixture['cells']:
        expected = .08*predicted_trace(fixture, c, simulation, 2.)
        for tr in c['traces']:
            tr['dff'] = expected.tolist()
    delayed = deepcopy(simulation)
    delayed['candidate']['synaptic_gain'] = 1.25
    delayed['rates_hz'] = {name: np.r_[np.zeros(100), values[:-100]].tolist()
                           for name, values in simulation['rates_hz'].items()}
    fit, selected = fit_candidates(fixture, [simulation, delayed])
    assert selected == 0
    for c in fit[0]['cells'].values():
        assert abs(c['observation']['effective_decay_s']-2.) < 1e-3
        assert abs(c['observation']['fluorescence_gain']-.08) < 1e-4
        assert c['test']['nmse'] < 1e-9
    changed = deepcopy(fixture)
    for c in changed['cells']:
        for tr in c['traces']:
            if tr['split'] == 'test':
                tr['dff'] = [10*v for v in tr['dff']]
    altered, selected_after = fit_candidates(changed, [simulation, delayed])
    assert selected_after == selected
    for name in fit[0]['cells']:
        assert fit[0]['cells'][name]['observation'] == altered[0]['cells'][name]['observation']
    assert altered[0]['test_nmse'] > fit[0]['test_nmse']
    print('GCaMP data, causality, parameter recovery, and held-out isolation passed')


def cuda():
    from brain.cuda_engine import CudaEngine
    import cupy as cp
    pre, post, weights = np.array([0]), np.array([1]), np.array([150.])
    engine = CudaEngine(2, pre, post, weights, np.array([0]))
    events = np.zeros((200, 1), np.uint8)
    events[0, 0] = 1
    engine.set_synaptic_gain(.75)
    engine.reset()
    weak = engine.step([0], [], events=events)
    engine.set_synaptic_gain(1.25)
    engine.reset()
    strong = engine.step([0], [], events=events)
    assert strong[1] > weak[1], (weak, strong)
    def run():
        engine.reset()
        result = engine.step([180], [])
        return result, cp.asnumpy(engine.v), cp.asnumpy(engine.g)
    engine.set_synaptic_gain(1.)
    baseline = run()
    engine.set_synaptic_gain(1.25)
    run()
    engine.set_synaptic_gain(1.)
    restored = run()
    for a, b in zip(baseline, restored):
        np.testing.assert_array_equal(a, b)
    for bad in [0, 3, np.nan, True, '1', [1]]:
        try:
            engine.set_synaptic_gain(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid gain accepted')
    print('CUDA gain intervention and exact baseline recovery passed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()
    validate()
    if args.cuda:
        cuda()
