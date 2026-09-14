"""Extract a reproducible, explicitly reduced MaleCNS leg reflex circuit.

Run on the cluster after downloading the three official v1.0 Feather tables.
No FlyWire IDs, synthetic edges, or inferred 13B-alpha identities are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES = {
    'annotations': 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
    'weights': 'connectome-weights-male-cns-v1.0-minconf-0.5.feather',
    'transmitters': 'body-neurotransmitters-male-cns-v1.0.feather',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract(source: Path):
    annotations = feather.read_table(source / FILES['annotations']).to_pylist()
    by_id = {r['bodyId']: r for r in annotations}
    sensory = sorted(r['bodyId'] for r in annotations
                     if r['rootSide'] == 'L' and r['entryNerve'] == 'ProLN'
                     and r['type'] in ('SNpp50', 'SNpp51'))
    flexor = sorted(r['bodyId'] for r in annotations
                    if r['superclass'] == 'vnc_motor' and r['somaSide'] == 'L'
                    and r['subclass'] == 'fl' and r['type'] == 'Ti flexor MN')
    extensor = sorted(r['bodyId'] for r in annotations
                      if r['superclass'] == 'vnc_motor' and r['somaSide'] == 'L'
                      and r['subclass'] == 'fl' and r['type'] == 'Ti extensor MN')
    assert sensory and flexor and extensor, 'Required annotated populations absent'
    motor = flexor + extensor
    graph = feather.read_table(source / FILES['weights'], memory_map=True)
    sensory_out = graph.filter(pc.is_in(graph['body_pre'], pa.array(sensory)))
    motor_in = graph.filter(pc.is_in(graph['body_post'], pa.array(motor)))
    intermediates = set(sensory_out['body_post'].to_pylist()) & set(motor_in['body_pre'].to_pylist())
    intermediates = {i for i in intermediates if by_id.get(i, {}).get('superclass') == 'vnc_intrinsic'}
    ids = sorted(set(sensory + motor) | intermediates)
    subset = graph.filter(pc.and_(pc.is_in(graph['body_pre'], pa.array(ids)),
                                 pc.is_in(graph['body_post'], pa.array(ids))))
    nts = feather.read_table(source / FILES['transmitters'])
    nts = nts.filter(pc.is_in(nts['body'], pa.array(ids)))
    nt_by_id = {r['body']: r for r in nts.to_pylist()}
    fields = ('bodyId', 'type', 'instance', 'superclass', 'somaSide', 'rootSide',
              'somaNeuromere', 'entryNerve', 'status', 'statusLabel', 'mancBodyid')
    neurons = [{**{k: by_id[i][k] for k in fields},
                'transmitter': nt_by_id.get(i, {}).get('consensus_nt'),
                'transmitter_confidence': nt_by_id.get(i, {}).get('predicted_nt_confidence')}
               for i in ids]
    edges = sorted(subset.to_pylist(), key=lambda r: (r['body_pre'], r['body_post']))
    incoming = graph.filter(pc.is_in(graph['body_post'], pa.array(ids)))
    all_input = int(pc.sum(incoming['weight']).as_py())
    included = sum(r['weight'] for r in edges)
    return {
        'dataset': 'male-cns:v1.0', 'license': 'CC-BY-4.0',
        'source_page': 'https://janelia-flyem.github.io/male-cns/download/',
        'sources': {k: {'url': BASE + n, 'sha256': digest(source / n)} for k, n in FILES.items()},
        'selection': 'All annotated VNC intrinsic neurons on two-edge paths from left ProLN SNpp50/51 to left front-leg Ti flexor/extensor MNs; retain every induced edge.',
        'sensory_ids': sensory, 'flexor_ids': flexor, 'extensor_ids': extensor,
        'neurons': neurons, 'edges': edges,
        'statistics': {'raw_segment_edges': graph.num_rows, 'neurons': len(ids),
                       'edges': len(edges), 'synapses': included,
                       'retained_incoming_synapse_fraction': included / all_input},
        'mapping': {
            'SNpp50_SNpp51': 'FeCO claw annotations are present on these types elsewhere in the release. Left ProLN selects the front leg. Individual extension/flexion tuning remains unresolved.',
            'flexor': 'Ti flexor MN population -> LFTibia_flex_93434, pooled motor-unit approximation.',
            'extensor': 'Ti extensor MN population -> LFTibia_extensor_93932, pooled motor-unit approximation.',
            '13Balpha': None,
        },
        'limitations': ['A reduced circuit, not the entire VNC or CNS.',
                       'Many inputs from outside the selected circuit are omitted.',
                       'The 13Balpha recording cohort has no verified body-ID match here.',
                       'Motor pools are not resolved into experimentally identified slow/intermediate/fast motor units.',
                       'Transmitter predictions do not determine postsynaptic receptor physiology.'],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('brain/data/male-cns-v1.0'))
    p.add_argument('--output', type=Path, default=Path('brain/calibration_data/leg_connectome.json'))
    args = p.parse_args()
    result = extract(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['statistics'], indent=2))


if __name__ == '__main__':
    main()
