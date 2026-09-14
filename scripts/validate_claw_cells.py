"""Fit frozen claw response curves and report independent-fly validation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from brain.claw_cell_calibration import load_records, fit_models, evaluate, predict, steady, PRIMARY


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def figure(records, fit, evaluation, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    for column, cohort in enumerate(PRIMARY):
        # Lowest held-out fly and column, selected without looking at fit quality.
        r = min((r for r in records if r['cohort'] == cohort and r['split'] == 'animal_test' and r['order'] == 3), key=lambda r: (r['fly'], r['region']))
        ax = axes[0, column]
        ax.plot(r['time_s'], r['calcium'], color='#596873', lw=1.3, label='Recorded GCaMP7f / tdTomato')
        ax.plot(r['time_s'], predict(r, fit), color='#b44226', lw=1.5, label='Frozen driver-specific curve')
        ax.plot(r['time_s'], predict(r, fit, 'legacy'), color='#84a69a', lw=1.1, label='Extension-only comparator')
        ax.set(title=f'{cohort} · unseen fly {r["fly"]}, region {r["region"]}\nStarts flexed · no test gain/baseline fit', xlabel='Source time (s)', ylabel='Fluorescence ratio change (ΔR/R)')
        ax.legend(fontsize=8)
        ax = axes[1, column]
        rows = [r for r in evaluation['fly_scores'] if r['cohort'] == cohort and r['split'] in ('animal_test', 'protocol_test')]
        ax.bar(np.arange(len(rows)), [r['relative_mse'] for r in rows], color=['#b44226' if r['split'] == 'animal_test' else '#54786b' for r in rows])
        ax.axhline(1, color='#6b7470', ls='--', lw=1)
        ax.set(xticks=np.arange(len(rows)), xticklabels=[r['animal_id'].split(':')[1] for r in rows], xlabel='Fly ID · orange: unseen flies; green: reverse-order transfer', ylabel='MSE / training-constant MSE', title='Each fly has equal weight · below 1 is better')
    fig.savefig(output/'claw-cells.png', dpi=170)
    fig.savefig(output/'claw-cells.pdf')
    plt.close(fig)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    directory = ROOT/'brain/calibration_data'
    metadata, records = load_records(directory)
    fit = fit_models(records)
    dump(output/'claw-cells-fit.json', fit)  # Freeze before any test evaluation.
    print(json.dumps({'selected': {c: fit['cohorts'][c]['selected'] for c in PRIMARY}}, indent=2), flush=True)
    evaluation = evaluate(records, fit)
    mapping = json.loads((directory/'claw_cell_mapping.json').read_text())
    report = {'dataset': metadata, 'frozen_fit': fit, 'evaluation': evaluation, 'mapping': mapping,
              'scope': 'Driver-specific calcium observation model; no individual EM neuron assignment.',
              'accepted_as_neural_calibration': False, 'maze_changed': False,
              'runtime': {'python': sys.version, 'numpy': np.__version__, 'scipy': __import__('scipy').__version__, 'host': platform.node()},
              'input_sha256': {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ('claw_cells.json', 'claw_cells.npz', 'claw_cell_mapping.json')},
              'implementation_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('brain/claw_cell_calibration.py', 'scripts/validate_claw_cells.py')},
              'fit_sha256': hashlib.sha256((output/'claw-cells-fit.json').read_bytes()).hexdigest(),
              'limitations': ['Only two independent test flies per driver; region traces are not independent biological replicates.',
                  'Source preprocessing used trial fluorescence; source-normalized traces were not used in our fits.',
                  'Effective relaxation time is selected from four candidates; it is not measured GCaMP or neuronal kinetics.',
                  'Some segmented regions contain more than one cell; JR688 can occasionally label flexion cells.',
                  'Raw fluorescence gain varies across cells/flies. No test-specific scaling is fitted.',
                  'The all-claw 73D10 cohort is preserved as reference only.',
                  'No exact MaleCNS body-ID match, absolute Hz calibration, muscle-force calibration or natural walking validation.']}
    dump(output/'claw-cells-report.json', report)
    print(json.dumps(evaluation['groups'], indent=2), flush=True)
    traces = []
    for r in records:
        if r['cohort'] not in PRIMARY:
            continue
        row = {k: r[k] for k in ('id', 'cohort', 'fly', 'animal_id', 'region', 'order', 'split')}
        # Display decimation only; fitting and every score use all source samples.
        indices = np.unique(np.r_[np.arange(0, len(r['time_s']), 4), len(r['time_s'])-1])
        for key in ('time_s', 'angle_deg', 'calcium'):
            row[key] = r[key][indices].tolist()
        row['prediction'] = predict(r, fit)[indices].tolist()
        row['legacy_prediction'] = predict(r, fit, 'legacy')[indices].tolist()
        traces.append(row)
    angles = np.linspace(15, 180, 166)
    view = {'display_stride': 4, 'traces': traces, 'angle_grid_deg': angles.tolist(),
            'steady_curves': {c: steady(angles, fit['cohorts'][c]['selected']['parameters'], c).tolist() for c in PRIMARY}}
    (output/'claw-cells-traces.json').write_text(json.dumps(view, separators=(',', ':'), allow_nan=False)+'\n')
    figure(records, fit, evaluation, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/claw-cell-validation')
    args = parser.parse_args()
    run(args.output)
