"""Run the full CUDA connectome against a small published GCaMP calibration set.

Use --simulate on the allocated GPU, then --fit and --plot on either host.
No server state, live profile, anatomical count, or connection sign is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.gcamp_calibration import (DEFAULT_DATA, load_data, fit_candidates,
                                     observations, score, paired_bootstrap)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False, separators=(',', ':'))+'\n')


def simulate(data, path, candidates, seed):
    from brain.model import ConnectomeBrain
    brain = ConnectomeBrain(backend='cuda')
    indices = {c['name']: [brain.index[i] for i in c['flywire_ids']] for c in data['cells']}
    onset = 1.2
    pulse = data['protocol']['stimulus_off_s']-data['protocol']['stimulus_on_s']
    dt = .02
    count = round((onset+pulse+12)/dt)
    start, stop = round(onset/dt), round((onset+pulse)/dt)
    sugar_slots = np.searchsorted(brain.inputs, brain.sugar.receptors)
    simulations = []
    for gain, rate in candidates:
        began = time.perf_counter()
        brain.reset()
        brain.engine.set_synaptic_gain(gain)
        # Common random input across candidates, then a different-seed check.
        brain.engine.rng = np.random.default_rng(seed-1)
        brain.engine.secondary_rng = np.random.default_rng(seed)
        rates = np.zeros(len(brain.inputs))
        traces = {name: [] for name in indices}
        total_spikes = 0
        for i in range(count):
            rates[sugar_slots] = rate if start <= i < stop else 0
            delta = brain.engine.step(rates, [])
            total_spikes += int(delta.sum())
            if i < start and delta.any():
                raise AssertionError('The released zero-input baseline must remain silent')
            for name, ix in indices.items():
                traces[name].append(float(delta[ix].mean()/dt))
        simulations.append({'candidate': {'synaptic_gain': gain, 'sugar_input_hz': rate},
                            'seed': seed, 'dt_s': dt, 'stimulus_on_s': onset,
                            'stimulus_off_s': onset+pulse,
                            'time_s': ((np.arange(count)+1)*dt).tolist(),
                            'rates_hz': traces, 'total_spikes': total_spikes})
        write_json(path, {'schema': 'gcamp-grid-v1', 'complete': len(simulations)==len(candidates),
                          'dataset_sha256': hashlib.sha256(DEFAULT_DATA.read_bytes()).hexdigest(),
                          'metadata': brain.metadata, 'simulations': simulations,
                          'input_cohort': brain.sugar.metadata,
                          'protocol_note': 'Released unilateral 21-cell sugar cohort is a proxy for bilateral physical taste stimulation. Analytically resting prehistory precedes the simulated 1.2 s baseline.'})
        print(json.dumps({'candidate': simulations[-1]['candidate'],
                          'seconds': round(time.perf_counter()-began, 2),
                          'completed': len(simulations), 'total': len(candidates)}), flush=True)


def fit(data, grid_path, output):
    grid = json.loads(grid_path.read_text())
    if not grid['complete'] or grid['dataset_sha256'] != hashlib.sha256(DEFAULT_DATA.read_bytes()).hexdigest():
        raise ValueError('Incomplete grid or mismatched data')
    fitted, selected_index = fit_candidates(data, grid['simulations'])
    baseline_index = next(i for i, c in enumerate(fitted)
                          if c['candidate'] == {'synaptic_gain': 1., 'sugar_input_hz': 200.})
    baseline, selected = fitted[baseline_index], fitted[selected_index]
    result = {'schema': 'gcamp-fit-v1', 'scope': 'Offline sugar-circuit calibration pilot; not whole-brain validation',
              'applied_to_live_model': False, 'dataset_source': data['source'],
              'dataset_sha256': grid['dataset_sha256'],
              'baseline_index': baseline_index, 'selected_index': selected_index,
              'candidates': fitted, 'held_out_comparison': paired_bootstrap(baseline, selected),
              'selection': 'Lowest mean normalized training MSE, with equal cell-type weighting. Test flies were scored only after selection.',
              'limits': [data['matching'], data['protocol']['timing_caveat'], grid['protocol_note'],
                         'Fitted decay and fluorescence gain belong to the observation model, not to neural membrane dynamics.',
                         'Three cell types and one sugar concentration do not uniquely identify neural gain versus input strength.',
                         'Water/bitter records are preserved but excluded from this sugar-only fit.',
                         'These experiments informed the original model literature; this is a held-out split for this fit, not wholly new external validation.']}
    write_json(output, result)
    print(json.dumps({'baseline': baseline['candidate'], 'selected': selected['candidate'],
                      'baseline_test_nmse': baseline['test_nmse'], 'selected_test_nmse': selected['test_nmse'],
                      'comparison': result['held_out_comparison']}), flush=True)


def seed_check(data, output_dir):
    report_path = output_dir/'fit.json'
    report = json.loads(report_path.read_text())
    cases = [report['candidates'][i] for i in [report['baseline_index'], report['selected_index']]]
    candidates = list(dict.fromkeys((c['candidate']['synaptic_gain'], c['candidate']['sugar_input_hz']) for c in cases))
    destination = output_dir/'seed-check.json'
    simulate(data, destination, candidates, seed=144)
    simulations = json.loads(destination.read_text())['simulations']
    checks = []
    for case in cases:
        simulation = next(s for s in simulations if s['candidate'] == case['candidate'])
        cells = {c['name']: score(data, c, simulation, case['cells'][c['name']]['observation'], 'test')
                 for c in data['cells']}
        checks.append({'candidate': case['candidate'], 'test_nmse': float(np.mean([v['nmse'] for v in cells.values()])),
                       'cells': cells})
    report['different_seed_check'] = {'seed': 144, 'refit_observation_parameters': False,
                                      'baseline': checks[0], 'selected': checks[1]}
    write_json(report_path, report)
    print(json.dumps({'seed': 144, 'baseline_test_nmse': checks[0]['test_nmse'],
                      'selected_test_nmse': checks[1]['test_nmse']}), flush=True)


def plot(data, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads((output_dir/'fit.json').read_text())
    baseline = report['candidates'][report['baseline_index']]
    selected = report['candidates'][report['selected_index']]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for ax, cell in zip(axes, data['cells']):
        t = np.asarray(cell['time_s'])-data['protocol']['stimulus_on_s']
        y, ids = observations(data, cell, 'test')
        mean = np.nanmean(y, axis=0)
        sem = np.nanstd(y, axis=0, ddof=1)/np.sqrt(np.isfinite(y).sum(axis=0))
        ax.axvspan(0, 7.2, color='#efb25e', alpha=.16)
        ax.errorbar(t, mean, yerr=sem, color='#343d4a', fmt='o', markersize=3,
                    linewidth=.8, label='Held-out GCaMP mean ± SEM')
        for case, color, style, label in [(baseline, '#b66b22', '--', 'Baseline + fitted readout'),
                                         (selected, '#008676', '-', 'Selected model + fitted readout')]:
            ax.plot(t, case['cells'][cell['name']]['test']['prediction_dff'], style, color=color, lw=1.8, label=label)
        ax.set(title=f"{cell['name']} · {len(ids)} held-out flies", xlabel='Seconds from sugar onset', xlim=(-6, 19.2))
        ax.spines[['top', 'right']].set_visible(False)
        ax.axhline(0, color='#c8ced5', linewidth=.6)
    axes[0].set_ylabel('Baseline-corrected ΔF/F')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .07), ncol=3, frameon=False, fontsize=9)
    params = selected['candidate']
    fig.suptitle('GCaMP calibration pilot: predictions on flies excluded from fitting', fontsize=14, x=.5, y=.99)
    fig.text(.5, .915, f"Selected coupling gain {params['synaptic_gain']:g}×, sugar input {params['sugar_input_hz']:g} Hz. "
             f"Test normalized error: {baseline['test_nmse']:.3f} → {selected['test_nmse']:.3f}.", ha='center', fontsize=10)
    fig.text(.02, .02, 'Source: Shiu, Sterne et al., eLife 2022, Figure 3 source data (CC BY 4.0). '
             'Pilot assumptions and limits are recorded in fit.json. Live model unchanged.', fontsize=8)
    fig.tight_layout(rect=(0, .17, 1, .89))
    fig.savefig(output_dir/'gcamp-calibration.png', dpi=180)
    fig.savefig(output_dir/'gcamp-calibration.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulate', action='store_true')
    p.add_argument('--fit', action='store_true')
    p.add_argument('--seed-check', action='store_true')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/gcamp-calibration')
    args = p.parse_args()
    if not any([args.simulate, args.fit, args.seed_check, args.plot]):
        p.error('Choose --simulate, --fit, --seed-check, and/or --plot')
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_data()
    grid = args.output/'grid.json'
    if args.simulate:
        simulate(data, grid, [(g, r) for g in [.75, 1., 1.25] for r in [50., 100., 200.]], seed=43)
    if args.fit:
        fit(data, grid, args.output/'fit.json')
    if args.seed_check:
        seed_check(data, args.output)
    if args.plot:
        plot(data, args.output)


if __name__ == '__main__':
    main()
