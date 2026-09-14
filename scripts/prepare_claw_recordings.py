"""Import public, processed proprioceptor calcium traces without resampling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def prepare(path):
    rows = pq.read_table(path).to_pylist()
    groups = {}
    for row in rows:
        groups.setdefault(row['trial'], []).append(row)
    records = []
    for name, group in groups.items():
        group.sort(key=lambda r: r['frame'])
        first = group[0]
        for key in ('roi', 'animal_id', 'stimulus_type', 'driver'):
            assert len({r[key] for r in group}) == 1
        animal = int(first['animal_id'])
        if first['roi'] != 'L1_x':
            split = 'reference_only'
        elif animal >= 9:
            split = 'animal_test'
        elif first['stimulus_type'] == 'ramp_hold_ext_first':
            split = 'protocol_test'
        else:
            split = 'train' if animal <= 6 else 'validation'
        record = {'id': name, 'animal_id': animal, 'roi': first['roi'],
                  'driver': first['driver'], 'protocol': first['stimulus_type'],
                  'split': split}
        for target, key in [('time_s', 'time'), ('frame', 'frame'),
                            ('angle_deg', 'L1C_flex'), ('calcium', 'calcium'),
                            ('analyze', 'analyze')]:
            record[target] = [float(r[key]) if r[key] is not None and np.isfinite(r[key]) else None for r in group]
        assert np.all(np.diff(record['time_s']) > 0)
        records.append(record)
    return {'source_doi': 'https://doi.org/10.5061/dryad.gqnk98t16',
            'original_article': 'https://doi.org/10.1016/j.neuron.2018.09.009',
            'release_article': 'https://doi.org/10.1038/s41586-025-09554-2',
            'source_file': path.name, 'source_download': 'https://datadryad.org/downloads/file_stream/4330032',
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_bytes': path.stat().st_size, 'license': 'CC0-1.0',
            'indicator': 'GCaMP6f',
            'signal_units': 'Processed calcium as released; no conversion to spikes or absolute calcium concentration.',
            'notes': [
                'ROI recordings pool claw axons with mixed flexion and extension tuning.',
                'The original Mamiya experiment imaged the right front leg; the later release uses L1 column/ROI names. No specimen/laterality/body-ID match is asserted.',
                'Source time, angle and calcium values are preserved, including tracking angles slightly above 180 degrees.',
                'The release is already processed. Its preprocessing cannot be undone; our additional normalization uses training data only.',
                'Animal labels across X/Y/Z ROIs are not confirmed to identify the same individuals. Only X is used; Y/Z are not counted as additional independent animals.',
            ], 'records': records}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('outputs/leg-calibration-source/dallmann2025/claw_magnet_Mamiya2018.parquet'))
    p.add_argument('--output', type=Path, default=Path('brain/calibration_data/claw_recordings.json'))
    args = p.parse_args()
    result = prepare(args.source)
    args.output.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n')
    print(f"Imported {len(result['records'])} traces; primary X-branch cohort has 10 animal labels")
