"""Fit the new claw benchmark and run paired physical protocol sensitivity tests."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from brain.claw_calibration import fit_candidates, evaluate, predict, mask
from brain.leg_reflex import ReflexProtocol, run_reflex

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def plot(dataset, fit, scores, sweep, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    # First held-out animal, both protocols, chosen by ID rather than fit quality.
    for ax, protocol in zip(axes[0], ('ramp_hold_flex_first', 'ramp_hold_ext_first')):
        r = next(r for r in dataset['records'] if r['animal_id'] == 9 and r['roi'] == 'L1_x' and r['protocol'] == protocol)
        t = np.asarray(r['time_s'])
        ax.plot(t, np.asarray(r['calcium'])/fit['training_scale'], color='#596873', lw=1, label='Recorded GCaMP6f')
        ax.plot(t, predict(r, fit), color='#b44226', lw=1.5, label='Frozen claw response fit')
        ax.plot(t, predict(r, fit, 'legacy_encoder'), color='#84a69a', lw=1, label='Extension-only comparator')
        ax.set(title=f'Unseen fly 9 · {protocol.replace("ramp_hold_", "").replace("_", " ")}',
               xlabel='Recording time (s)', ylabel='Calcium / training maximum')
        ax.legend(fontsize=8)
    ax = axes[1, 0]
    for split, color in [('animal_test', '#b44226'), ('protocol_test', '#456f62')]:
        rows = [s for s in scores['scores'] if s['split'] == split]
        ax.scatter(np.arange(len(rows)), [s['models'][fit['selected_model']]['relative_mse'] for s in rows], label=split.replace('_',' '), color=color)
    ax.axhline(1, ls='--', color='#777', label='Training-constant baseline')
    ax.set(title='Held-out prediction error', xlabel='Trial within test group', ylabel='MSE / constant-baseline MSE')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    if sweep:
        labels = [r['name'] for r in sweep]
        ax.barh(labels, [r['max_angle_difference_deg'] for r in sweep], color=['#b44226' if r['protocol']['displacement_deg'] > 0 else '#64748b' for r in sweep])
        ax.set(title='Physical circuit contribution · seed 1', xlabel='Maximum |connected − disconnected| (degrees)')
    fig.savefig(output/'claw-reflex.png', dpi=170)
    fig.savefig(output/'claw-reflex.pdf')
    plt.close(fig)


def run(output, calcium_only=False):
    output.mkdir(parents=True, exist_ok=True)
    data_path = ROOT/'brain/calibration_data/claw_recordings.json'
    dataset = json.loads(data_path.read_text())
    fit = fit_candidates(dataset['records'])
    # Persist the frozen fit before evaluating test responses.
    dump(output/'claw-fit.json', fit)
    scores = evaluate(dataset['records'], fit)
    view_traces = []
    for r in dataset['records']:
        if r['split'] == 'reference_only':
            continue
        valid = mask(r)
        row = {k:r[k] for k in ('id','animal_id','protocol','split','time_s','angle_deg')}
        calcium = np.asarray(r['calcium'], dtype=float)/fit['training_scale']
        row['calcium'] = [float(v) if ok else None for v, ok in zip(calcium, valid)]
        row['prediction'] = predict(r, fit).tolist()
        row['legacy_prediction'] = predict(r, fit, 'legacy_encoder').tolist()
        view_traces.append(row)
    (output/'claw-traces.json').write_text(json.dumps(view_traces, separators=(',', ':'), allow_nan=False)+'\n')
    print(json.dumps({'selected': fit['selected_model'], 'validation': {n: r['validation_mse'] for n,r in fit['candidates'].items()}, 'test_groups': scores['groups']}, indent=2), flush=True)
    sweep, trials = [], {}
    if not calcium_only:
        graph = json.loads((ROOT/'brain/calibration_data/leg_connectome.json').read_text())
        p = ReflexProtocol()
        protocols = [(f'{"Extend" if d>0 else "Flex"} {abs(d)}°', replace(p, displacement_deg=d)) for d in (-30,-20,-10,10,20,30)]
        protocols += [('Extend 20° · fast', replace(p, ramp_s=.05)),
                      ('Extend 20° · slow', replace(p, ramp_s=.2)),
                      ('Extend 20° · 2× mass', replace(p, distal_mass_scale=2))]
        for index, (name, protocol) in enumerate(protocols):
            pair = {c: run_reflex(graph, c, protocol=protocol) for c in ('connected','disconnected')}
            active, passive = pair['connected'], pair['disconnected']
            delta = np.asarray([r['angle_deg'] for r in active['trace']])-np.asarray([r['angle_deg'] for r in passive['trace']])
            released = np.asarray([not r['held'] for r in active['trace']])
            signed_restore = -np.sign(protocol.displacement_deg)*delta[released]
            peak = int(np.argmax(np.abs(delta[released])))
            result = {'id': str(index), 'name': name, 'protocol': asdict(protocol),
                      'release_s': protocol.release_s, 'seed': 1,
                      'motor_spikes': active['motor_spikes'],
                      'disconnected_motor_spikes': passive['motor_spikes'],
                      'max_angle_difference_deg': float(np.max(np.abs(delta[released]))),
                      'signed_restoring_difference_at_peak_deg': float(signed_restore[peak]),
                      'active_final_angle_deg': active['final_angle_deg'],
                      'passive_final_angle_deg': passive['final_angle_deg']}
            sweep.append(result)
            trials[str(index)] = pair
            print(f"{name}: {result['motor_spikes']} motor spikes; max circuit contribution {result['max_angle_difference_deg']:.4f} degrees", flush=True)
    report = {'status': 'Sensory response benchmark and physical sensitivity test; physiological circuit calibration not accepted',
              'dataset': {k:v for k,v in dataset.items() if k != 'records'},
              'recording_count': len(dataset['records']), 'primary_recording_count': 20, 'primary_animal_count': 10, 'all_roi_unique_animal_count': None,
              'frozen_fit': fit, 'evaluation': scores, 'sweep': sweep,
              'input_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
              'fit_sha256': hashlib.sha256((output/'claw-fit.json').read_bytes()).hexdigest(),
              'implementation_sha256': {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('brain/claw_calibration.py','brain/leg_reflex.py','scripts/continue_leg_reflex.py')},
              'runtime': {'python':sys.version, 'numpy':np.__version__, 'scipy':__import__('scipy').__version__, 'mujoco':__import__('mujoco').__version__},
              'accepted_as_neural_calibration': False, 'maze_changed': False,
              'limitations': [
                  'Only two animals form the final independent-animal test; no population confidence claim.',
                  'Opposite-order tests on animals 1–8 share animals with training or validation.',
                  'Pooled claw axon calcium cannot identify individual MaleCNS sensory-cell receptive fields.',
                  'The fixed calcium kernel and fitted fluorescence coefficients do not identify absolute firing rates.',
                  'Physical sweeps retain the original provisional extension-only encoder; the calcium fit is not injected into the circuit.',
                  'Distal load is a sensitivity test with tibia/tarsus mass and inertia doubled, not a measured biological load.',
                  'Motor recruitment, spike-to-force scaling and the missing circuit inputs remain uncalibrated.',
              ]}
    dump(output/'claw-report.json', report)
    if trials:
        (output/'sweep-trials.json').write_text(json.dumps(trials, separators=(',', ':'), allow_nan=False)+'\n')
    plot(dataset, fit, scores, sweep, output)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/leg-reflex-followup')
    parser.add_argument('--calcium-only', action='store_true')
    args = parser.parse_args()
    run(args.output, args.calcium_only)
