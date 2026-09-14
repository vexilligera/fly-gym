"""Import the public Mamiya 2023 numeric summaries with strict provenance."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = '866e8487b51b8e2cbc1c9ec7d875a33f918249d14c76c115e37601b05cf59862'
FIELDS = ('x_position', 'y_position', 'drr', 'source_normalized_drr', 'angle_deg', 'fly_id', 'time_s')
COHORTS = {
    'JR209': {'regions_per_order': 50, 'flies': 7, 'train': [1, 2, 3, 4], 'validation': [5], 'animal_test': [6, 7], 'expected_tuning': 'flexion'},
    'JR688': {'regions_per_order': 28, 'flies': 6, 'train': [1, 2, 3], 'validation': [4], 'animal_test': [5, 6], 'expected_tuning': 'extension'},
    '73D10': {'regions_per_order': 53, 'flies': 7, 'reference_only': True, 'expected_tuning': 'mixed'},
}


class NumericUnpickler(pickle.Unpickler):
    """Only NumPy numeric array construction is permitted in this known archive."""
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
                   ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported global: {module}.{name}')
        return allowed[module, name]


def read_source(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('Source archive differs from the reviewed Dryad release')
    arrays, files = {}, []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(('trial_type2', 'trial_type3')):
                continue
            cohort = next(c for c in COHORTS if c in name)
            order = int(name[-1])
            key = f'{cohort}_{order}'
            content = archive.read(name)
            values = NumericUnpickler(io.BytesIO(content)).load()
            if not isinstance(values, list) or len(values) != len(FIELDS):
                raise ValueError('Unexpected summary format')
            for field, a in zip(FIELDS, values):
                if not isinstance(a, np.ndarray) or a.dtype.kind not in 'fi' or not np.isfinite(a).all():
                    raise ValueError(f'Non-finite/non-numeric source: {name}, {field}')
                arrays[f'{key}_{field}'] = a
            n_time, n_regions = values[2].shape
            assert all(a.shape == (n_time, n_regions) for a in values[:5])
            assert values[5].shape == (1, n_regions) and values[6].shape == (n_time,)
            assert np.all(np.diff(values[6]) > 0)
            assert n_regions == COHORTS[cohort]['regions_per_order']
            assert np.array_equal(np.unique(values[5]), np.arange(1, COHORTS[cohort]['flies']+1))
            files.append({'key': key, 'cohort': cohort, 'order': order, 'member': name,
                          'member_sha256': hashlib.sha256(content).hexdigest(),
                          'samples_per_region': n_time, 'regions': n_regions,
                          'dt_s': float(np.median(np.diff(values[6])))})
    assert len(files) == 6 and len(arrays) == 42
    return arrays, files


def prepare(source, output):
    arrays, files = read_source(source)
    output.mkdir(parents=True, exist_ok=True)
    target = output/'claw_cells.npz'
    np.savez_compressed(target, **arrays)
    metadata = {
        'source_doi': 'https://doi.org/10.5061/dryad.dbrv15f6q',
        'article_doi': 'https://doi.org/10.1016/j.neuron.2023.07.009',
        'download': 'https://datadryad.org/downloads/file_stream/2453318',
        'source_file': source.name, 'source_sha256': SOURCE_SHA, 'source_bytes': source.stat().st_size,
        'license': 'CC0-1.0', 'indicator': 'GCaMP7f + tdTomato',
        'units': 'Processed fluorescence ratio change, delta R/R; not Hz.',
        'sex': 'female', 'leg': 'right front',
        'cohorts': COHORTS, 'files': files, 'npz_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'notes': ['Source arrays are preserved exactly; normalized traces are reference only.',
                  'Regions sometimes merge multiple cells. Column correspondence across orders is not independently verified.',
                  'Type 2 starts extended; type 3 starts flexed. Times retain the source alignment.',
                  'Fly IDs are namespaced by driver; never count regions as independent flies.',
                  'No recording has an established MaleCNS body-ID match.'],
    }
    (output/'claw_cells.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'files': len(files), 'region_traces': sum(f['regions'] for f in files),
                      'primary_flies': 13, 'reference_flies': 7, 'npz_bytes': target.stat().st_size}))


def audit_mapping(output):
    import pyarrow.feather as feather
    path = ROOT/'brain/data/male-cns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather'
    graph_path = ROOT/'brain/calibration_data/leg_connectome.json'
    graph = json.loads(graph_path.read_text())
    rows = feather.read_table(path).to_pylist()
    selected = [r for r in rows if r['bodyId'] in graph['sensory_ids']]
    assert len(selected) == 4
    audit = {'source_page': 'https://janelia-flyem.github.io/male-cns/download/',
             'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'graph_sha256': hashlib.sha256(graph_path.read_bytes()).hexdigest(),
             'annotation_fields_checked': list(selected[0]), 'neurons': selected,
             'type_synonym_evidence': {t: [{'bodyId': r['bodyId'], 'synonyms': r['synonyms']}
                 for r in rows if r['type'] == t and r['synonyms']] for t in ('SNpp50', 'SNpp51')},
             'verified_recording_matches': 0, 'assigned_tuning': {str(r['bodyId']): None for r in selected},
             'limits': ['None of the four exact records has a flexion/extension label or peripheral soma position.',
                        'MANC body IDs and VFB identifiers are anatomical references, not links to a recorded ROI.',
                        'The physiology experiment used female right legs; this fixture uses a male left leg.',
                        'No morphology registration, genetic-driver overlap measurement or neuron identity match has been established.']}
    (output/'claw_cell_mapping.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/leg-calibration-source/mamiya2023/2023_Mamiya_etal_Proprioceptor_feature_selectivity.zip')
    parser.add_argument('--output', type=Path, default=ROOT/'brain/calibration_data')
    args = parser.parse_args()
    prepare(args.source, args.output)
    audit_mapping(args.output)
