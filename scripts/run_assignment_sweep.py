"""Run the frozen 16-assignment physical sensitivity design with resumable trials."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.sensory_assignment import ASSIGNMENTS, SETTINGS, SEEDS, protocols, protocol_names, run_candidate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_config():
    sources = ['brain/calibration_data/leg_connectome.json', 'wasm/calibration/claw-cells-fit.json',
               'brain/leg_reflex.py', 'brain/sensory_assignment.py', 'brain/ASSIGNMENT_SWEEP.md']
    return {'version': 1, 'source_hashes': {p: digest(ROOT/p) for p in sources},
            'assignments': ASSIGNMENTS, 'settings': SETTINGS, 'seeds': SEEDS,
            'protocols': {n: asdict(p) for n, p in protocols().items()},
            'sensory_body_id_order': json.loads((ROOT/sources[0]).read_text())['sensory_ids'],
            'duration_s': 1.6, 'biological_mapping_selected': None, 'maze_changed': False}


def jobs():
    result = []
    for setting in SETTINGS:
        for assignment in ASSIGNMENTS:
            for name in protocol_names(setting):
                for seed in SEEDS:
                    result.append({'id': f'{setting}__{assignment}__{name}__{seed}',
                                   'setting': setting, 'assignment': assignment, 'protocol_name': name,
                                   'seed': seed, 'condition': 'connected'})
    for name in protocols():
        result.append({'id': f'passive__{name}', 'setting': 'nominal', 'assignment': 'EEEE',
                       'protocol_name': name, 'seed': 1, 'condition': 'disconnected'})
    return result


def initialize():
    global GRAPH, FIT
    GRAPH = json.loads((ROOT/'brain/calibration_data/leg_connectome.json').read_text())
    FIT = json.loads((ROOT/'wasm/calibration/claw-cells-fit.json').read_text())


def worker(job, output, config_hash):
    start = time.monotonic()
    result = run_candidate(GRAPH, FIT, job['assignment'], job['setting'],
                           protocols()[job['protocol_name']], job['seed'], job['condition'])
    result.update(id=job['id'], protocol_name=job['protocol_name'], config_hash=config_hash,
                  wall_time_s=time.monotonic()-start)
    path = Path(output)/'trials'/f'{job["id"]}.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n')
    temporary.replace(path)
    return {'id': job['id'], 'motor_spikes': result['motor_spikes'], 'wall_time_s': result['wall_time_s']}


def run(output, workers):
    output.mkdir(parents=True, exist_ok=True)
    (output/'trials').mkdir(exist_ok=True)
    config = frozen_config()
    encoded = json.dumps(config, sort_keys=True, separators=(',', ':'))
    config_hash = hashlib.sha256(encoded.encode()).hexdigest()
    path = output/'config.json'
    if path.exists() and json.loads(path.read_text())['config_hash'] != config_hash:
        raise ValueError('Output belongs to a different design; choose a new output directory')
    path.write_text(json.dumps({'config_hash': config_hash, **config}, indent=2)+'\n')
    todo = []
    for job in jobs():
        existing = output/'trials'/f'{job["id"]}.json'
        if existing.exists():
            if json.loads(existing.read_text())['config_hash'] != config_hash:
                raise ValueError(f'Stale trial: {existing}')
        else:
            todo.append(job)
    print(json.dumps({'total_trials': len(jobs()), 'remaining': len(todo), 'workers': workers,
                      'host': platform.node(), 'config_hash': config_hash}), flush=True)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        futures = [pool.submit(worker, job, str(output), config_hash) for job in todo]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if count % 32 == 0 or count == len(todo):
                print(json.dumps({'completed_now': count, 'remaining': len(todo)-count,
                                  'elapsed_s': round(time.monotonic()-started, 1), 'latest': result}), flush=True)
    (output/'runtime.json').write_text(json.dumps({'host': platform.node(), 'python': sys.version,
        'numpy': __import__('numpy').__version__, 'scipy': __import__('scipy').__version__,
        'mujoco': __import__('mujoco').__version__, 'workers': workers,
        'wall_time_s_this_invocation': time.monotonic()-started, 'trial_count': len(jobs()),
        'config_hash': config_hash}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/assignment-sweep')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('Use 1–8 workers within the existing allocation')
    run(args.output, args.workers)
