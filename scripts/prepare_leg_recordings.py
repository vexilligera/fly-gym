"""Preserve Agrawal 2020 leg recordings, provenance and fixed evaluation splits.

Download Imaging_data.zip and Behavior_data.zip from the public Dryad page.
The browser download works without login; Dryad's API download requires auth.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def finite_list(x):
    return [float(v) if np.isfinite(v) else None for v in np.asarray(x)]


def prepare(directory):
    imaging_path = directory / 'Imaging_data.zip'
    behavior_path = directory / 'Behavior_data.zip'
    imaging = zipfile.ZipFile(imaging_path)
    behavior = zipfile.ZipFile(behavior_path)
    records = []
    for protocol in ('rampandhold', 'extfirstswings', 'flexfirstswings'):
        member = f'Imaging data/13Balpha_{protocol}.mat'
        data = loadmat(io.BytesIO(imaging.read(member)), simplify_cells=True)
        for i, (angle, dff, fly, stimulus) in enumerate(zip(
                data['LegAngle'], data['DFF'], data['fly'], data['stimulus'])):
            assert len(angle) == len(dff)
            records.append({
                'id': f'{protocol}-{int(fly)}', 'cell_type': '13Balpha',
                'source_member': member, 'source_cell_index_matlab': i + 1,
                'source_fly_label': int(fly), 'protocol': str(stimulus),
                'split': ('train' if i < 2 else 'test') if protocol == 'rampandhold' else 'protocol_test',
                'sample_rate_hz': 7.57,
                'angle_deg': finite_list(angle), 'dff': finite_list(dff),
            })
    data = loadmat(io.BytesIO(behavior.read('Behavior data/Agrawal_2020_fig_3.mat')),
                   simplify_cells=True)['Fig_3']
    movements = []
    for condition in ('OFFBALL', 'ONBALL'):
        group = f'{condition}_13B_20847xCsChrimson'
        for i, fly in enumerate(data[group]['DATA']):
            joint = fly['FeTi_jointangles']
            movements.append({
                'id': f'{condition.lower()}-{i+1}',
                'source_field': f'Fig_3.{group}.DATA({i+1}).FeTi_jointangles',
                'condition': 'unloaded' if condition == 'OFFBALL' else 'loaded',
                'split': ('train' if i < 3 else 'test') if condition == 'OFFBALL' else 'load_test',
                'sample_rate_hz': 300, 'time_origin_note': 'Assumed first sample -0.2 s from methods; raw export has no timestamp or laser channel.',
                'stimulus_duration_s': .72,
                'stim_angle_deg': finite_list(joint['stim']['avg']),
                'control_angle_deg': finite_list(joint['control']['avg']),
                'stim_trials': int(np.asarray(joint['stim']['all']).shape[1]),
                'control_trials': int(np.asarray(joint['control']['all']).shape[1]),
            })
    return {
        'source': 'Agrawal et al., eLife 2020, 10.7554/eLife.60299',
        'doi': 'https://doi.org/10.5061/dryad.k3j9kd55t', 'license': 'CC0-1.0',
        'files': {p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                           'bytes': p.stat().st_size} for p in (imaging_path, behavior_path)},
        'notes': [
            'Raw arrays are preserved; missing samples are null.',
            'Imaging README specifies 7.57 Hz unless otherwise specified; article methods say 8.01 Hz. Both rates are evaluated.',
            'Fly labels in different imaging protocols are not confirmed to identify the same individuals. Protocol tests are not independent-animal validation.',
            'Calcium traces are GCaMP6f. They do not provide absolute spike rates.',
            'Behavior is headless flies with optogenetic 13Balpha activation, not direct motor-neuron stimulation.',
            'The model-to-13Balpha identity is unresolved; these recordings cannot calibrate named MaleCNS cells yet.',
        ],
        'imaging': records, 'behavior': movements,
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('outputs/leg-calibration-source'))
    p.add_argument('--output', type=Path, default=Path('brain/calibration_data/leg_recordings.json'))
    args = p.parse_args()
    result = prepare(args.source)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"Preserved {len(result['imaging'])} calcium traces and {len(result['behavior'])} fly movement records")
