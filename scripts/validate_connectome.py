"""Meaningful full-release data and intervention checks against a local backend."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'external/fly-brain/data'
    def api(action, payload=None):
        request = urllib.request.Request(f'http://127.0.0.1:{args.port}/api/brain/{action}')
        if payload is not None:
            request.data = json.dumps(payload).encode()
            request.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    status = api('status')
    assert status['status'] == 'ready', status
    metadata = status['metadata']
    assert (metadata['neurons'], metadata['connections'], metadata['synapse_count_sum']) == (138639, 15091983, 54492922)
    with (source / '2025_Completeness_783.csv').open() as f:
        ids = np.array([int(row[0]) for row in list(csv.reader(f))[1:]], dtype=np.int64)
    with (root / 'brain/data/neuron_annotations.tsv').open() as f:
        annotations = {int(row['root_id']): row for row in csv.DictReader(f, delimiter='\t')}
    geometry = api('geometry')
    assert geometry['schema'] == 'flywire-v783-anchors-v1'
    assert (geometry['neuron_count'], geometry['positioned_count'], geometry['unpositioned_count']) == (138639, 138625, 14)
    placed = np.asarray(geometry['indices'])
    positions = np.asarray(geometry['positions_um']).reshape(-1, 3)
    assert len(set(placed)) == len(placed) == len(positions)
    expected_placed, expected_positions = [], []
    for index, rid in enumerate(ids):
        row = annotations.get(int(rid), {})
        xyz = [float(row.get('pos_' + axis, '') or 'nan') * scale for axis, scale in zip('xyz', [.004, .004, .04])]
        if np.isfinite(xyz).all():
            expected_placed.append(index)
            expected_positions.append(xyz)
    assert np.array_equal(placed, expected_placed)
    assert np.allclose(positions, expected_positions, atol=.00051, rtol=0), 'Anchor coordinates must preserve source voxel scaling'
    assert api(f'neuron/{int(placed[0])}')['id'] == str(ids[placed[0]])
    unplaced = sorted(set(range(len(ids))) - set(placed))
    assert api(f'neuron/{unplaced[0]}')['id'] == str(ids[unplaced[0]])
    print(f'Geometry: all {len(placed):,} anchors mapped to the correct simulated neuron and physical coordinates', flush=True)
    checked = 0
    for batch in pq.ParquetFile(source / '2025_Connectivity_783.parquet').iter_batches(batch_size=250000, columns=['Presynaptic_ID','Postsynaptic_ID','Presynaptic_Index','Postsynaptic_Index']):
        cols = [batch.column(i).to_numpy() for i in range(4)]
        assert np.array_equal(ids[cols[2]], cols[0])
        assert np.array_equal(ids[cols[3]], cols[1])
        checked += len(cols[0])
    assert checked == metadata['connections']
    results = {}
    for stimulus, silence, steps in [('none',False,2),('walk',False,5),('walk',True,3),('odor',False,3),('reverse',False,4)]:
        api('reset', {})
        runs = [api('step', {'stimulus':stimulus,'silence':silence,'rate_hz':100,'odor':.5}) for _ in range(steps)]
        for run in runs:
            activity = run['activity']
            indices, counts = activity['indices'], activity['spike_counts']
            assert len(indices) == len(counts) == run['active_neurons']
            assert len(set(indices)) == len(indices)
            assert sum(counts) == run['spikes']
            assert all(0 <= i < len(ids) for i in indices + activity['input_indices'])
            assert all(isinstance(c, int) and c > 0 for c in counts)
            by_index = dict(zip(indices, counts))
            for top in run['top_neurons']:
                assert top['id'] == str(ids[top['index']])
                assert by_index[top['index']] == top['spikes']
            counts_by_id = {str(ids[i]): count for i, count in by_index.items()}
            for group, members in metadata['groups'].items():
                rate = sum(counts_by_id.get(rid, 0) for rid in members) / len(members) / run['step_seconds']
                assert abs(rate - run['rates_hz'][group]) < .001
        if stimulus == 'none' or silence:
            assert all(r['spikes'] == 0 and r['gains'] == [0,0] for r in runs)
        elif stimulus == 'walk':
            assert max(r['active_neurons'] for r in runs) > 2
            assert max(sum(r['gains']) for r in runs) > .2
        elif stimulus == 'odor':
            assert max(r['active_neurons'] for r in runs) > 68
            assert any(r['rates_hz']['ORN_DM1_left'] > 0 for r in runs)
        elif stimulus == 'reverse':
            assert min(sum(r['gains']) for r in runs) < -.1
        results[stimulus + ('_silenced' if silence else '')] = runs
        print(f'{stimulus} silence={silence}: passed, peak active={max(r["active_neurons"] for r in runs)}', flush=True)
    api('reset', {})
    hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'2025_Completeness_783.csv',source/'2025_Connectivity_783.parquet',root/'brain/data/neuron_annotations.tsv']}
    output = root/'brain/validation.json'
    output.write_text(json.dumps({'passed':True,'metadata':metadata,'connection_id_rows_validated':checked,'anchor_positions_validated':len(placed),'sparse_activity_validated':True,'sha256':hashes,'interventions':results},indent=2))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
