"""Summarize candidate contrasts and experimental distinguishability, without fitting."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from brain.sensory_assignment import ASSIGNMENTS, SETTINGS, SEEDS, protocols, protocol_names, candidate_rates
from scripts.run_assignment_sweep import jobs

ANGLE_RESOLUTION = .5
MOTOR_RESOLUTION = 5.


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def pairwise(groups, setting, names):
    rows = []
    for a, b in combinations(ASSIGNMENTS, 2):
        by_protocol = {}
        for name in names:
            x, y = groups[(setting, a, name)], groups[(setting, b, name)]
            mask = x['post_release']
            angle = x['evoked_angle'][:, mask]-y['evoked_angle'][:, mask]
            motor = x['evoked_motor'][:, :, mask]-y['evoked_motor'][:, :, mask]
            by_protocol[name] = {
                'angle_rms_deg': float(np.sqrt(np.mean(angle.mean(axis=0)**2))),
                'motor_rms_hz': float(np.sqrt(np.mean(motor.mean(axis=0)**2))),
                'minimum_seed_angle_rms_deg': float(np.sqrt(np.mean(angle**2, axis=1)).min()),
                'minimum_seed_motor_rms_hz': float(np.sqrt(np.mean(motor**2, axis=(1, 2))).min()),
                'joint_separated_every_seed': bool(np.all(
                    (np.sqrt(np.mean(angle**2, axis=1)) > ANGLE_RESOLUTION) |
                    (np.sqrt(np.mean(motor**2, axis=(1, 2))) > MOTOR_RESOLUTION)))}
        angle_max = max(p['angle_rms_deg'] for p in by_protocol.values())
        motor_max = max(p['motor_rms_hz'] for p in by_protocol.values())
        rows.append({'a': a, 'b': b, 'by_protocol': by_protocol,
                     'max_angle_rms_deg': angle_max, 'max_motor_rms_hz': motor_max,
                     'angle_separated': angle_max > ANGLE_RESOLUTION,
                     'motor_separated': motor_max > MOTOR_RESOLUTION,
                     'joint_separated': angle_max > ANGLE_RESOLUTION or motor_max > MOTOR_RESOLUTION})
    return rows


def summarize_pairs(rows):
    return {'pairs': len(rows), 'angle_separated_pairs': sum(r['angle_separated'] for r in rows),
            'motor_separated_pairs': sum(r['motor_separated'] for r in rows),
            'joint_separated_pairs': sum(r['joint_separated'] for r in rows),
            'close_pairs': [[r['a'], r['b']] for r in rows if not r['joint_separated']]}


def analyze(output):
    config = json.loads((output/'config.json').read_text())
    graph = json.loads((ROOT/'brain/calibration_data/leg_connectome.json').read_text())
    fit = json.loads((ROOT/'wasm/calibration/claw-cells-fit.json').read_text())
    ids = [n['bodyId'] for n in graph['neurons']]
    neuron_by_id = {n['bodyId']: n for n in graph['neurons']}
    fields = ('angle_deg', 'flexor_hz', 'extensor_hz', 'sensory_hz')
    trials, hashes = {}, {}
    for job in jobs():
        path = output/'trials'/f'{job["id"]}.json'
        raw = path.read_bytes()
        r = json.loads(raw)
        assert r['config_hash'] == config['config_hash']
        hashes[job['id']] = hashlib.sha256(raw).hexdigest()
        trials[job['id']] = {'values': np.array([[point[field] for point in r['trace']] for field in fields]),
                            'time': np.array([point['time_s'] for point in r['trace']]),
                            'counts': np.array([r['post_onset_spikes_by_body_id'][str(i)] for i in ids]),
                            'motor_spikes': r['motor_spikes']}
    groups, summaries, view = {}, [], []
    for setting in SETTINGS:
        for assignment in ASSIGNMENTS:
            for name in protocol_names(setting):
                if name.startswith('sham'):
                    continue
                protocol = protocols()[name]
                sham_name = 'sham_slow' if name.endswith('_slow') else 'sham'
                active = [trials[f'{setting}__{assignment}__{name}__{s}'] for s in SEEDS]
                sham = [trials[f'{setting}__{assignment}__{sham_name}__{s}'] for s in SEEDS]
                passive = trials[f'passive__{name}']
                passive_sham = trials[f'passive__{sham_name}']
                time = passive['time']
                a = np.array([r['values'] for r in active])
                s = np.array([r['values'] for r in sham])
                evoked_angle = (a[:, 0]-passive['values'][0])-(s[:, 0]-passive_sham['values'][0])
                evoked_motor = a[:, 1:3]-s[:, 1:3]
                count_response = (np.array([r['counts'] for r in active])-np.array([r['counts'] for r in sham]))/(1.6-protocol.onset_s)
                mask = time >= protocol.release_s
                angle_mean = evoked_angle.mean(axis=0)
                angle_sd = evoked_angle.std(axis=0, ddof=1)
                peak = np.flatnonzero(mask)[np.argmax(np.abs(angle_mean[mask]))]
                key = (setting, assignment, name)
                groups[key] = {'evoked_angle': evoked_angle, 'evoked_motor': evoked_motor,
                               'post_release': mask, 'count_response_hz': count_response}
                summaries.append({'setting': setting, 'assignment': assignment, 'protocol': name,
                    'max_mean_evoked_angle_deg': float(np.max(np.abs(angle_mean[mask]))),
                    'signed_restoring_at_peak_deg': float(-np.sign(protocol.displacement_deg)*angle_mean[peak]),
                    'angle_rms_deg': float(np.sqrt(np.mean(angle_mean[mask]**2))),
                    'seed_angle_sd_rms_deg': float(np.sqrt(np.mean(angle_sd[mask]**2))),
                    'per_seed_peak_evoked_angle_deg': np.max(np.abs(evoked_angle[:, mask]), axis=1).tolist(),
                    'mean_total_motor_spikes': float(np.mean([r['motor_spikes'] for r in active])),
                    'mean_sham_motor_spikes': float(np.mean([r['motor_spikes'] for r in sham])),
                    'mean_evoked_motor_spikes': float(np.mean([r['motor_spikes']-s['motor_spikes'] for r, s in zip(active, sham)])),
                    'resting_assumed_sensory_hz': candidate_rates(100, assignment, setting, fit).tolist()})
                # Display at 10 ms; analyses use every source sample.
                keep = np.unique(np.r_[np.arange(0, len(time), 2), len(time)-1])
                series = {'time_s': time, 'angle_mean_deg': angle_mean, 'angle_sd_deg': angle_sd,
                          'flexor_mean_hz': evoked_motor[:, 0].mean(axis=0),
                          'extensor_mean_hz': evoked_motor[:, 1].mean(axis=0)}
                view.append({'setting': setting, 'assignment': assignment, 'protocol': name,
                             'release_s': protocol.release_s,
                             **{k: np.round(v[keep], 6).tolist() for k, v in series.items()}})
    main_names = [n for n in protocol_names('nominal') if not n.startswith('sham')]
    pairs = pairwise(groups, 'nominal', main_names)
    sensitivity = {setting: summarize_pairs(pairwise(groups, setting, ['flex_30', 'extend_30'])) for setting in SETTINGS}
    protocol_comparison = []
    for name in main_names:
        ps = [r['by_protocol'][name] for r in pairs]
        protocol_comparison.append({'protocol': name,
            'angle_separated_pairs': sum(r['angle_rms_deg'] > ANGLE_RESOLUTION for r in ps),
            'motor_separated_pairs': sum(r['motor_rms_hz'] > MOTOR_RESOLUTION for r in ps),
            'joint_separated_pairs': sum(r['angle_rms_deg'] > ANGLE_RESOLUTION or r['motor_rms_hz'] > MOTOR_RESOLUTION for r in ps),
            'every_seed_joint_separated_pairs': sum(r['joint_separated_every_seed'] for r in ps),
            'median_pair_angle_rms_deg': float(np.median([r['angle_rms_deg'] for r in ps])),
            'median_pair_motor_rms_hz': float(np.median([r['motor_rms_hz'] for r in ps]))})
    # A suggested measurable downstream cell, excluding the four imposed inputs.
    readouts = []
    for name in main_names:
        responses = np.array([groups[('nominal', a, name)]['count_response_hz'] for a in ASSIGNMENTS])
        means = responses.mean(axis=1)
        for index, body in enumerate(ids):
            if body in graph['sensory_ids']:
                continue
            readouts.append({'protocol': name, 'body_id': body, 'type': neuron_by_id[body]['type'],
                'superclass': neuron_by_id[body]['superclass'],
                'candidate_mean_rate_range_hz': float(np.ptp(means[:, index])),
                'candidate_mean_rate_sd_hz': float(means[:, index].std()),
                'median_within_mapping_seed_sd_hz': float(np.median(responses[:, :, index].std(axis=1, ddof=1))),
                'mean_evoked_hz_by_assignment': dict(zip(ASSIGNMENTS, means[:, index].tolist()))})
    readouts.sort(key=lambda r: (-r['candidate_mean_rate_sd_hz'], r['body_id'], r['protocol']))
    report = {'config': config, 'runtime': json.loads((output/'runtime.json').read_text()),
        'preflight': json.loads((output/'preflight.json').read_text()),
        'sweep_validation': json.loads((output/'sweep-validation.json').read_text()),
        'summary': summarize_pairs(pairs), 'pairwise': pairs, 'groups': summaries,
        'sensitivity_at_30_deg': sensitivity, 'protocol_comparison': protocol_comparison,
        'suggested_downstream_readouts': readouts[:20], 'biological_mapping_selected': None,
        'measured_downstream_targets_used': False, 'rejected_biological_mappings': [],
        'assumed_resolution': {'angle_rms_deg': ANGLE_RESOLUTION, 'pooled_motor_rms_hz': MOTOR_RESOLUTION},
        'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations': ['This is a sensitivity sweep, not fitting to matched biological reflex recordings.',
            'Calcium curve shape does not identify spike rate, sensory dynamics, or individual-cell tuning.',
            'Common random numbers couple comparisons; three seeds are not biological replicates.',
            'The 0.5 degree / 5 Hz thresholds are hypothetical measurement resolutions, not measured noise.',
            'Pairwise closeness is not transitive and does not establish equivalent classes of neurons.',
            'Suggested readouts and stimulus choices depend on the provisional circuit, gain and muscle model.',
            'The original neural encoder and live maze remain unchanged.']}
    dump(output/'assignment-report.json', report)
    dump(output/'assignment-trial-hashes.json', {'config_hash': config['config_hash'], 'trials': hashes})
    (output/'assignment-traces.json').write_text(json.dumps(view, separators=(',', ':'), allow_nan=False)+'\n')
    plot(report, output)
    print(json.dumps({'summary': report['summary'], 'sensitivity': sensitivity,
                      'protocols': protocol_comparison, 'top_readout': readouts[0]}, indent=2))
    return report


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    names = [p['protocol'] for p in report['protocol_comparison']]
    matrix = np.array([[next(r['max_mean_evoked_angle_deg'] for r in report['groups'] if r['setting'] == 'nominal' and r['assignment'] == a and r['protocol'] == n) for n in names] for a in ASSIGNMENTS])
    image = axes[0].imshow(matrix, aspect='auto', cmap='YlOrBr')
    axes[0].set(yticks=range(16), yticklabels=ASSIGNMENTS, xticks=range(len(names)),
                xticklabels=[n.replace('_', ' ') for n in names], title='Candidate circuit contribution to movement', ylabel='Provisional F/E assignment in body-ID order')
    axes[0].tick_params(axis='x', rotation=55)
    fig.colorbar(image, ax=axes[0], label='Peak |mean evoked circuit contrast| (degrees)')
    x = np.arange(len(names))
    axes[1].bar(x-.18, [p['angle_separated_pairs'] for p in report['protocol_comparison']], width=.36, label='Angle >0.5° RMS', color='#b44226')
    axes[1].bar(x+.18, [p['joint_separated_pairs'] for p in report['protocol_comparison']], width=.36, label='Angle or pooled motor >5 Hz RMS', color='#456f62')
    axes[1].set(xticks=x, xticklabels=[n.replace('_', ' ') for n in names], ylim=(0, 125),
                ylabel='Candidate pairs separated / 120', title='Hypothetical resolution · not biological validation')
    axes[1].tick_params(axis='x', rotation=55)
    axes[1].legend(fontsize=8)
    fig.savefig(output/'assignment-sweep.png', dpi=170)
    fig.savefig(output/'assignment-sweep.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/assignment-sweep')
    analyze(parser.parse_args().output)
