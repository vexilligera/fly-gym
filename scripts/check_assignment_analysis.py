"""Check distinguishability metrics and recompute saved physical contrasts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from brain.sensory_assignment import ASSIGNMENTS, SETTINGS, protocols
from scripts.analyze_assignment_sweep import pairwise, summarize_pairs


def synthetic_checks():
    groups = {}
    for a in ASSIGNMENTS:
        groups[('nominal', a, 'test')] = {'evoked_angle': np.zeros((3, 10)),
            'evoked_motor': np.zeros((3, 2, 10)), 'post_release': np.ones(10, dtype=bool)}
    identical = pairwise(groups, 'nominal', ['test'])
    assert len(identical) == 120
    assert summarize_pairs(identical)['joint_separated_pairs'] == 0
    # A constant 1-degree difference in one candidate separates exactly its
    # 15 pairs; adding identical motor signals cannot create other separation.
    groups[('nominal', 'FFFF', 'test')]['evoked_angle'][:] = 1
    altered = pairwise(groups, 'nominal', ['test'])
    assert summarize_pairs(altered)['angle_separated_pairs'] == 15
    assert summarize_pairs(altered)['motor_separated_pairs'] == 0
    # One exceptional seed must not be called separation in every seed.
    groups[('nominal', 'FFFF', 'test')]['evoked_angle'][0] = 0
    altered = pairwise(groups, 'nominal', ['test'])
    involving = [r for r in altered if 'FFFF' in (r['a'], r['b'])]
    assert all(r['by_protocol']['test']['minimum_seed_angle_rms_deg'] == 0 for r in involving)
    assert all(r['angle_separated'] for r in involving)
    print('Synthetic identical-signal, known-difference and seed-consistency checks passed.', flush=True)


def validate(output):
    report = json.loads((output/'assignment-report.json').read_text())
    assert report['biological_mapping_selected'] is None and not report['measured_downstream_targets_used']
    assert report['rejected_biological_mappings'] == []
    assert len(report['groups']) == 256 and len(report['pairwise']) == 120
    assert report['preflight']['passed'] and report['sweep_validation']['passed']
    hashes = json.loads((output/'assignment-trial-hashes.json').read_text())
    for name, digest in hashes['trials'].items():
        assert hashlib.sha256((output/'trials'/f'{name}.json').read_bytes()).hexdigest() == digest
    # Recompute selected contrasts directly from raw coordinate/rate traces.
    choices = [('nominal', 'FFFF', 'flex_50'), ('nominal', 'EEEE', 'extend_50'),
               ('nominal', 'FEFE', 'flex_30_slow'), ('gain_75', 'EFFE', 'extend_30'),
               ('gain_300', 'FEEF', 'flex_30'), ('midpoint_minus10', 'EEEE', 'flex_30'),
               ('midpoint_plus10', 'FFFF', 'extend_30')]
    def read(name):
        return json.loads((output/'trials'/f'{name}.json').read_text())
    for setting, assignment, name in choices:
        sham_name = 'sham_slow' if name.endswith('_slow') else 'sham'
        passive = read(f'passive__{name}')['trace']
        passive_sham = read(f'passive__{sham_name}')['trace']
        changes = []
        for seed in (1, 2, 3):
            active = read(f'{setting}__{assignment}__{name}__{seed}')['trace']
            sham = read(f'{setting}__{assignment}__{sham_name}__{seed}')['trace']
            changes.append([(a['angle_deg']-s['angle_deg'])-(p['angle_deg']-q['angle_deg'])
                            for a, s, p, q in zip(active, sham, passive, passive_sham)])
        mask = np.array([p['time_s'] >= protocols()[name].release_s for p in passive])
        mean = np.mean(changes, axis=0)
        item = next(r for r in report['groups'] if (r['setting'], r['assignment'], r['protocol']) == (setting, assignment, name))
        np.testing.assert_allclose(item['max_mean_evoked_angle_deg'], np.max(np.abs(mean[mask])), atol=1e-10)
        np.testing.assert_allclose(item['angle_rms_deg'], np.sqrt(np.mean(mean[mask]**2)), atol=1e-10)
    for setting, summary in report['sensitivity_at_30_deg'].items():
        assert summary['pairs'] == 120
        assert 0 <= summary['angle_separated_pairs'] <= summary['joint_separated_pairs'] <= 120
    view = json.loads((output/'assignment-traces.json').read_text())
    assert len(view) == 256
    for r in view:
        assert r['setting'] in SETTINGS and r['assignment'] in ASSIGNMENTS
        assert r['protocol'] in protocols()
        length = len(r['time_s'])
        for field in ('angle_mean_deg', 'angle_sd_deg', 'flexor_mean_hz', 'extensor_mean_hz'):
            assert len(r[field]) == length and np.isfinite(r[field]).all()
        assert np.all(np.asarray(r['angle_sd_deg']) >= 0)
    # Input neurons are externally imposed and must not be proposed as an
    # independently informative downstream measurement.
    sensory = set(report['config']['sensory_body_id_order'])
    assert all(r['body_id'] not in sensory for r in report['suggested_downstream_readouts'])
    result = {'passed': True, 'checks': ['identical candidates produce no separation',
        'one known one-degree difference separates exactly 15 pairs',
        'an exceptional seed is not mistaken for all-seed agreement',
        'all raw-trial hashes match the analyzed outputs',
        'seven physical contrasts recomputed directly from saved raw traces',
        'all 256 display traces finite and consistent',
        'suggested downstream readouts exclude imposed sensory inputs',
        'no biological target fitting, rejection or automatic mapping selection']}
    (output/'assignment-analysis-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/assignment-sweep')
    parser.add_argument('--synthetic-only', action='store_true')
    args = parser.parse_args()
    synthetic_checks()
    if not args.synthetic_only:
        validate(args.output)
