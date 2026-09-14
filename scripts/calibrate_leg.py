"""Run the leg calibration fixture, held-out data comparisons and causal controls."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from brain.leg_calibration import fit_observation, observation, promotion_status
from brain.leg_reflex import ReflexParameters, run_reflex

ROOT = Path(__file__).resolve().parents[1]


def plot_results(records, fits, trials, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    for ax, name in zip(axes[0], ('rampandhold-3', 'extfirstswings-1')):
        record = next(r for r in records if r['id'] == name)
        pred, target, mask = observation(record, fits[0]['parameter_values'])
        time = np.arange(len(pred))/record['sample_rate_hz']
        ax.plot(time, np.where(mask, target, np.nan), color='#575f6d', lw=1, label='Recorded GCaMP6f')
        ax.plot(time, np.where(mask, pred, np.nan), color='#ba4e29', lw=1.8, label='Fit frozen after training')
        ax.set(title=f'Held-out {name}', xlabel='Time (s)', ylabel='Baseline-centered ΔF/F')
        ax.legend(fontsize=8)
    for condition, color in [('connected', '#b44226'), ('disconnected', '#64748b')]:
        trace = trials[condition]['trace']
        time = [r['time_s'] for r in trace]
        axes[1, 0].plot(time, [r['angle_deg'] for r in trace], label=condition, color=color)
    axes[1, 0].axvspan(.3, .6, color='#e8bc64', alpha=.25, label='Imposed extension')
    axes[1, 0].set(xlabel='Simulation time (s)', ylabel='Femur–tibia angle (degrees)', title='Physical effect of the neural circuit')
    axes[1, 0].legend(fontsize=8)
    tr = trials['connected']['trace']
    for name, color in [('sensory_hz', '#64748b'), ('flexor_hz', '#b44226'), ('extensor_hz', '#459578')]:
        axes[1, 1].plot([r['time_s'] for r in tr], [r[name] for r in tr], label=name.replace('_hz',''), color=color)
    axes[1, 1].set(xlabel='Simulation time (s)', ylabel='Filtered population mean (Hz)', title='Simulated activity · provisional physiology')
    axes[1, 1].legend(fontsize=8)
    fig.savefig(output/'calibration.png', dpi=170)
    fig.savefig(output/'calibration.pdf')
    plt.close(fig)


def run(output, render=False):
    output.mkdir(parents=True, exist_ok=True)
    graph_path = ROOT/'brain/calibration_data/leg_connectome.json'
    recordings_path = ROOT/'brain/calibration_data/leg_recordings.json'
    graph = json.loads(graph_path.read_text())
    recordings = json.loads(recordings_path.read_text())
    fits = [fit_observation(recordings['imaging'], rate) for rate in (7.57, 8.01)]
    trials = {}
    for condition in ('connected', 'disconnected', 'sensory_off', 'motor_silenced'):
        video = output/f'{condition}.mp4' if render and condition in ('connected', 'disconnected') else None
        trial = run_reflex(graph, condition, video=video)
        trials[condition] = trial
        print(f"{condition}: {trial['motor_spikes']} motor spikes, {trial['total_spikes']} total spikes", flush=True)
    active = np.array([r['angle_deg'] for r in trials['connected']['trace']])
    passive = np.array([r['angle_deg'] for r in trials['disconnected']['trace']])
    released = np.array([not r['held'] for r in trials['connected']['trace']])
    # Independent seeds test the causal effect without refitting anything.
    sensitivity = []
    for seed, sign in ((2, -1), (3, -1), (1, 1)):
        trial = run_reflex(graph, seed=seed, parameters=replace(ReflexParameters(), glutamate_sign=sign))
        delta = np.array([r['angle_deg'] for r in trial['trace']])-passive
        sensitivity.append({'seed': seed, 'glutamate_sign': sign,
                            'max_additional_flexion_deg': float(-delta[released].min()),
                            'motor_spikes': trial['motor_spikes']})
    behavior_summary = []
    for r in recordings['behavior']:
        stim, control = (np.asarray(r[k], dtype=float) for k in ('stim_angle_deg', 'control_angle_deg'))
        # Methods: mean final 200 ms of 720 ms activation minus preceding 200 ms.
        stim_delta = float(np.nanmean(stim[216:276])-np.nanmean(stim[:60]))
        control_delta = float(np.nanmean(control[216:276])-np.nanmean(control[:60]))
        behavior_summary.append({'id': r['id'], 'condition': r['condition'], 'split': r['split'],
                                 'stim_change_deg': stim_delta, 'control_change_deg': control_delta,
                                 'stim_minus_control_deg': stim_delta-control_delta})
    report = {
        'title': 'Left-front-leg calibration fixture',
        'status': 'Prototype wired; physiological calibration not accepted',
        'connectome': graph['statistics'], 'dataset': graph['dataset'],
        'input_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (graph_path, recordings_path)},
        'implementation_hashes': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('brain/leg_reflex.py', 'brain/leg_calibration.py', 'scripts/calibrate_leg.py')},
        'runtime': {'python': sys.version, 'numpy': np.__version__,
                    'mujoco': __import__('mujoco').__version__, 'scipy': __import__('scipy').__version__},
        'circuit_selection': graph['selection'], 'anatomy_mapping': graph['mapping'],
        'calcium_fits': fits,
        'promotion': promotion_status(graph, fits[0]),
        'causal_effect': {'max_additional_flexion_deg': float(np.max((passive-active)[released])),
                          'motor_spikes': trials['connected']['motor_spikes'],
                          'disconnected_motor_spikes': trials['disconnected']['motor_spikes'],
                          'sensory_off_total_spikes': trials['sensory_off']['total_spikes'],
                          'motor_silenced_motor_spikes': trials['motor_silenced']['motor_spikes']},
        'sensitivity': sensitivity, 'published_behavior': behavior_summary,
        'behavior_comparison_status': 'Recorded 13Balpha stimulation is a distinct experiment. No direct numerical equivalence to the simulated mechanical perturbation is claimed.',
        'limitations': graph['limitations'] + recordings['notes'] + [
            'All four selected claw-type sensory cells are provisionally extension-tuned; their individual receptive fields are not known.',
            'LIF parameters, synaptic gain and motor-to-muscle gain are provisional. None were fitted to these calcium traces.',
            'Glutamate is modeled as inhibitory within the CNS in the baseline; a positive-sign sensitivity run is reported. Motor-to-muscle excitation is positive.',
            'Three neurons have unclear transmitter predictions: their outgoing edges remain in the anatomical file but are disabled in dynamics.',
            'The physical test clamps other joints, counterbalances the resting tibia, imposes a 20-degree extension, then releases it.',
            'Passive mechanics also return the leg. The reported neural effect is the difference against a matched disconnected control.',
            'The pooled flexor/extensor muscle mapping omits recruitment differences among motor units.',
            'This is an isolated leg experiment; the live maze still uses its previous brain and body controller.',
        ],
    }
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'trials.json').write_text(json.dumps(trials, allow_nan=False)+'\n')
    plot_results(recordings['imaging'], fits, trials, output)
    print(json.dumps(report['causal_effect'], indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'outputs/leg-calibration')
    p.add_argument('--render', action='store_true')
    args = p.parse_args()
    run(args.output, args.render)
