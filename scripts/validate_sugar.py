"""Validate released sugar inputs, recurrent responses, and CUDA input compatibility."""
import ast
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.sugar_input import SUGAR_IDS, MN9_IDS, SugarInput


def main():
    tree = ast.parse((ROOT / 'external/fly-brain/code/benchmark.py').read_text())
    spec = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(k, ast.Name) and k.id == 'EXPERIMENTS' for k in n.targets))['sugar']
    assert list(SUGAR_IDS) == [str(i) for i in spec['neu_exc']]
    with (ROOT / 'brain/data/neuron_annotations.tsv').open() as f:
        annotations = {r['root_id']: r for r in csv.DictReader(f, delimiter='\t')}
    ids = list(annotations)
    sugar = SugarInput(ids, annotations)
    assert len(sugar.receptors) == 21 and not set(sugar.motor.values()) & set(sugar.receptors)

    import cupy as cp
    from brain.cuda_engine import CudaEngine
    pre = np.array([0, 2, 3, 4, 5, 1], dtype=np.int32)
    post = np.array([3, 3, 4, 5, 1, 3], dtype=np.int32)
    weights = np.array([100, -40, 100, 70, -20, 30], dtype=np.int32)
    base = CudaEngine(6, pre, post, weights, np.array([0, 2]))
    extended = CudaEngine(6, pre, post, weights, np.array([0, 1, 2]), primary_inputs=[0, 2])
    for _ in range(100):
        a = base.step([180, 80], [])
        b = extended.step([180, 0, 80], [])
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(cp.asnumpy(base.v), cp.asnumpy(extended.v))
        np.testing.assert_array_equal(cp.asnumpy(base.g), cp.asnumpy(extended.g))
    del base, extended

    from brain.model import ConnectomeBrain
    from brain.sugar_assay import SugarAssay
    brain = ConnectomeBrain(backend='cuda')
    assert brain.metadata['neurons'] == 138639 and brain.metadata['connections'] == 15091983
    trials = []
    for rate in [0, 50, 100, 200]:
        assay = SugarAssay(brain, rate, paced=False)
        on_indices = set(brain.sugar.receptors.tolist()) if rate else set()
        started = time.perf_counter()
        while not assay.done:
            n = assay.step()
            assert np.isfinite(brain.last_delta).all()
            expected = on_indices if assay.phase == 'sugar' else set()
            assert set(n['activity']['input_indices']) == expected
            if assay.phase == 'baseline' or rate == 0:
                assert n['spikes'] == 0
        assert n['time'] == 8 and len(assay.trace) == 80
        assert [r['input_rate_hz'] for r in assay.trace] == [0]*20 + [rate]*40 + [0]*20
        on = assay.trace[20:60]
        mean = {k: float(np.mean([r[k] for r in on])) for k in ['GRN_hz', 'MN9_left_hz', 'MN9_right_hz']}
        if rate == 200:
            assert mean['MN9_right_hz'] > 0 and mean['GRN_hz'] > 0
        trials.append({'input_rate_hz': rate, 'mean_on_rates_hz': mean,
                       'trace': assay.trace, 'wall_seconds': time.perf_counter()-started})
        print(json.dumps({k:v for k,v in trials[-1].items() if k != 'trace'}), flush=True)
    result = {'passed': True, 'source_cohort_verified': True,
              'legacy_input_counts_and_voltages_identical': True,
              'direct_input_only_to_sugar_GRNs': True, 'no_taste_control_silent': True,
              'metadata': brain.sugar.metadata, 'trials': trials}
    (ROOT / 'outputs/sugar-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Sugar validation passed', flush=True)


if __name__ == '__main__':
    main()
