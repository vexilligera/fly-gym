"""FlyWire v783 LIF network using the released Shiu/Eon Brian2 equations.

All neurons and all connection rows in the release are instantiated. The
descending-rate to CPG decoder below is an explicit local demonstration model,
not a reconstructed VNC or a biologically validated body interface.
"""
from pathlib import Path
from collections import Counter
import csv
import gc
import time
import os
import platform

import numpy as np
import pyarrow.parquet as pq
import brian2 as b

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'external/fly-brain/data'
STEP_MS = 20
EON_COMMIT = 'a3db62f9436074e485c0278290c2164ed6150808'


class ConnectomeBrain:
    def __init__(self, progress=print, backend='brian2'):
        started = time.perf_counter()
        progress('Reading FlyWire v783 neurons and annotations')
        with (DATA / '2025_Completeness_783.csv').open() as f:
            self.ids = [r[0] for r in list(csv.reader(f))[1:]]
        self.index = {root_id: i for i, root_id in enumerate(self.ids)}
        with (ROOT / 'brain/data/neuron_annotations.tsv').open() as f:
            self.annotations = {r['root_id']: r for r in csv.DictReader(f, delimiter='\t') if r['root_id'] in self.index}
        # Immutable positions use exactly the same indices as the simulated
        # neuron table. These are annotation anchors, not somas or morphologies.
        positioned, positions = [], []
        for i, rid in enumerate(self.ids):
            annotation = self.annotations.get(rid, {})
            xyz = [float(annotation.get('pos_' + axis, '') or 'nan') * scale
                   for axis, scale in zip('xyz', (.004, .004, .04))]
            if np.isfinite(xyz).all():
                positioned.append(i)
                positions.extend(round(value, 3) for value in xyz)
        self.geometry = {
            'schema': 'flywire-v783-anchors-v1', 'units': 'micrometers',
            'neuron_count': len(self.ids), 'positioned_count': len(positioned),
            'unpositioned_count': len(self.ids) - len(positioned),
            'indices': positioned, 'positions_um': positions,
            'source_voxel_size_nm': [4, 4, 40],
            'description': 'FlyWire annotation anchors, usually on the neuron backbone; not neuron shapes.',
        }
        self.groups = {}
        for name in ['DNp09', 'DNa02', 'MDN', 'ORN_DM1']:
            for side in ['left', 'right']:
                self.groups[f'{name}_{side}'] = np.array([
                    self.index[rid] for rid, r in self.annotations.items()
                    if r['cell_type'] == name and r['side'] == side
                ], dtype=np.int32)
        if any(len(indices) == 0 for indices in self.groups.values()):
            raise ValueError('Required v783 neuron types/sides were not found.')
        progress('Loading every released connection (15 million rows)')
        columns = ['Presynaptic_Index', 'Postsynaptic_Index', 'Excitatory x Connectivity', 'Connectivity']
        table = pq.read_table(DATA / '2025_Connectivity_783.parquet', columns=columns)
        pre = table[columns[0]].to_numpy().astype(np.int32)
        post = table[columns[1]].to_numpy().astype(np.int32)
        weights = table[columns[2]].to_numpy().astype(np.float64)
        if pre.min() < 0 or post.min() < 0 or max(pre.max(), post.max()) >= len(self.ids):
            raise ValueError('Connection indices do not match the neuron table.')
        self.metadata = {
            'neurons': len(self.ids), 'connections': len(pre),
            'synapse_count_sum': int(table[columns[3]].to_numpy().sum()),
            'positive_connections': int((weights > 0).sum()),
            'negative_connections': int((weights < 0).sum()),
            'zero_connections': int((weights == 0).sum()),
            'annotation_matches': len(self.annotations),
            'release': 'FlyWire v783 / Eon public model', 'commit': EON_COMMIT,
            'dt_ms': .1, 'batch_ms': STEP_MS,
            'groups': {key: [self.ids[i] for i in indices] for key, indices in self.groups.items()},
            'neurotransmitters': dict(Counter(r['top_nt'] or 'unknown' for r in self.annotations.values())),
            'decoder': 'DNp09 forward; MDN reverse; DNa02 differential steering. Demonstration gains, no VNC connectome.',
            'backend': backend, 'hostname': platform.node(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        }
        del table
        from brain.vision_input import VisualInput
        self.visual = VisualInput(self.ids, self.annotations, pre, post, weights)
        self.metadata['vision'] = self.visual.metadata
        self.odor_projection = {side: np.array([
            self.index[rid] for rid,r in self.annotations.items()
            if r['cell_type']=='DM1_lPN' and r['side']==side
        ],dtype=np.int32) for side in ('left','right')}
        self.metadata['olfaction'] = {
            'input': '68 ORN_DM1 cells; food-odor proxy, not sugar receptors',
            'projection': 'DM1_lPN cells monitored downstream, not directly stimulated',
            'encoder': 'Normalized antenna odor → saturating common rate and 16× bilateral contrast gain → Poisson input; engineered approximation.',
            'chemistry': 'No volatile sucrose, receptor binding, gustatory circuit, feeding, or metabolism model.',
        }
        self.inputs = np.unique(np.concatenate([*self.groups.values(), self.visual.receptors]))
        self.input_index = {int(i): j for j, i in enumerate(self.inputs)}
        self.engine = None
        if backend == 'cuda':
            from brain.cuda_engine import CudaEngine
            progress('Constructing full recurrent network on the allocated CUDA GPU')
            self.engine = CudaEngine(len(self.ids), pre, post, weights, self.inputs)
            self.metadata.update(self.engine.metadata)
        elif backend == 'brian2':
            self._create_brian_network(pre, post, weights, progress)
        else:
            raise ValueError('Unknown brain backend')
        del pre, post, weights
        gc.collect()
        self.filtered = dict.fromkeys(self.groups, 0.0)
        self.previous_counts = np.zeros(len(self.ids), dtype=np.int32)
        self.last_delta = self.previous_counts.copy()
        self.metadata['load_seconds'] = round(time.perf_counter() - started, 2)
        progress('Full connectome ready')

    def _create_brian_network(self, pre, post, weights, progress):
        progress('Constructing full recurrent Brian2 network')
        b.prefs.codegen.target = 'cython'
        b.defaultclock.dt = .1 * b.ms
        b.seed(42)
        self.neurons = b.NeuronGroup(
            len(self.ids),
            '''dv/dt = (-52*mV - v + g)/(20*ms) : volt (unless refractory)
               dg/dt = -g/(5*ms) : volt (unless refractory)
               rfc : second
               silenced : boolean''',
            threshold='v > -45*mV and not silenced',
            reset='v = -52*mV; g = 0*mV', refractory='rfc', method='linear',
        )
        self.neurons.v = -52 * b.mV
        self.neurons.g = 0 * b.mV
        self.neurons.rfc = 2.2 * b.ms
        self.synapses = b.Synapses(self.neurons, self.neurons, 'w : volt', on_pre='g_post += w', delay=1.8*b.ms)
        self.synapses.connect(i=pre, j=post)
        self.synapses.w = weights * .275 * b.mV
        # Explicit input neurons; we do not add a baseline current to the brain.
        self.poisson = b.PoissonGroup(len(self.inputs), rates=0*b.Hz)
        self.input_synapses = b.Synapses(self.poisson, self.neurons, on_pre='v_post += 68.75*mV')
        self.input_synapses.connect(i=np.arange(len(self.inputs)), j=self.inputs)
        # Counts for all neurons, without retaining an unbounded spike history.
        self.monitor = b.SpikeMonitor(self.neurons, record=False)
        self.network = b.Network(self.neurons, self.synapses, self.poisson, self.input_synapses, self.monitor)
        progress('Compiling Brian2 neuron and synapse dynamics')
        self.network.run(0*b.ms, namespace={})
        self.network.store('rest')

    def reset(self):
        if self.engine is not None:
            self.engine.reset()
        else:
            self.network.restore('rest', restore_random_state=True)
        self.filtered = dict.fromkeys(self.groups, 0.0)
        self.previous_counts[:] = 0
        self.last_delta[:] = 0
        return {'time': 0, 'reset': True}

    def neuron_info(self, index):
        if not 0 <= index < len(self.ids):
            raise ValueError('Neuron index is outside this release')
        rid = self.ids[index]
        annotation = self.annotations.get(rid, {})
        return {
            'index': index, 'id': rid,
            **{key: annotation.get(key, '') for key in
               ('cell_type', 'super_class', 'side', 'top_nt')},
        }

    def step(self, stimulus='walk', rate_hz=100.0, odor=0.5, silence=False):
        if not isinstance(silence, bool):
            raise ValueError('silence must be a boolean')
        if stimulus not in ('none', 'walk', 'left', 'right', 'reverse', 'odor'):
            raise ValueError('Unknown stimulation preset')
        rate_hz, odor = float(rate_hz), float(odor)
        if not np.isfinite(rate_hz) or not 0 <= rate_hz <= 250 or not np.isfinite(odor) or not 0 <= odor <= 1:
            raise ValueError('Rate must be 0–250 Hz and odor must be 0–1.')
        rates = np.zeros(len(self.inputs))
        targets = []
        if stimulus in ('walk', 'left', 'right'):
            targets += [('DNp09_left', rate_hz), ('DNp09_right', rate_hz)]
        if stimulus in ('left', 'right'):
            targets.append((f'DNa02_{stimulus}', rate_hz))
        if stimulus == 'reverse':
            targets += [('MDN_left', rate_hz), ('MDN_right', rate_hz)]
        if stimulus == 'odor':
            # Illustrative sensory encoding only: normalized odor exposure is
            # converted to ORN input rate. No molecule binding/diffusion model.
            odor_rate = 150 * odor
            targets += [('ORN_DM1_left', odor_rate), ('ORN_DM1_right', odor_rate)]
        for key, rate in targets:
            for i in self.groups[key]: rates[self.input_index[int(i)]] = rate
        return self._advance(rates, stimulus, rate_hz, odor, silence)

    def step_visual(self, contrast, condition='vision'):
        rates = np.zeros(len(self.inputs))
        slots = np.searchsorted(self.inputs, self.visual.receptors)
        rates[slots] = self.visual.rates(contrast, condition)
        return self._advance(rates, 'visual_' + condition, 180.0, 0.0, False)

    def step_multisensory(self, contrast, odor, vision=True, olfaction=True):
        if not isinstance(vision,bool) or not isinstance(olfaction,bool):
            raise ValueError('Sensory switches must be booleans')
        from brain.olfactory_input import odor_input_rates
        odor_rates = odor_input_rates(odor)
        if not olfaction: odor_rates[:] = 0
        rates = np.zeros(len(self.inputs))
        rates[np.searchsorted(self.inputs,self.visual.receptors)] = self.visual.rates(contrast,'vision' if vision else 'blind')
        for side, rate in zip(('left','right'),odor_rates):
            rates[np.searchsorted(self.inputs,self.groups['ORN_DM1_'+side])] = rate
        result = self._advance(rates,'vision_and_olfaction',180.0,float(np.mean(odor)),False)
        result['olfaction'] = {
            'input_rates_hz': odor_rates.tolist(),
            'ORN_rates_hz': [float(self.last_delta[self.groups['ORN_DM1_'+side]].mean()/.02) for side in ('left','right')],
            'PN_spikes': [int(self.last_delta[self.odor_projection[side]].sum()) for side in ('left','right')],
        }
        return result

    def _advance(self, rates, stimulus, rate_hz, odor, silence):
        silenced_ids = np.concatenate([self.groups['DNp09_left'], self.groups['DNp09_right']])
        started = time.perf_counter()
        if self.engine is not None:
            delta = self.engine.step(rates, silenced_ids if silence else [])
            simulation_time = self.engine.steps / 10000
        else:
            self.poisson.rates = rates*b.Hz
            self.neurons.rfc[self.inputs] = np.where(rates > 0, 0, 2.2)*b.ms
            self.neurons.silenced[silenced_ids] = bool(silence)
            self.network.run(STEP_MS*b.ms, namespace={})
            counts = np.asarray(self.monitor.count[:], dtype=np.int32)
            delta = counts - self.previous_counts
            self.previous_counts = counts.copy()
            simulation_time = float(self.network.t / b.second)
        elapsed = time.perf_counter() - started
        self.last_delta = delta
        alpha = 1 - np.exp(-STEP_MS / 100.0)
        raw = {}
        for key, indices in self.groups.items():
            raw[key] = float(delta[indices].mean() / (STEP_MS / 1000))
            self.filtered[key] += alpha * (raw[key] - self.filtered[key])
        f = self.filtered
        forward = (f['DNp09_left'] + f['DNp09_right'])/2/100
        reverse = (f['MDN_left'] + f['MDN_right'])/2/100
        turn = (f['DNa02_left'] - f['DNa02_right'])/100
        gains = np.clip([forward-reverse-.6*turn, forward-reverse+.6*turn], -1.2, 1.2)
        active = np.flatnonzero(delta)
        top = active[np.argsort(delta[active])[-8:][::-1]]
        return {
            'time': simulation_time, 'step_seconds': STEP_MS/1000,
            'wall_seconds': elapsed, 'spikes': int(delta.sum()), 'active_neurons': int(len(active)),
            'stimulus': stimulus, 'input_rate_hz': rate_hz, 'odor': odor,
            'rates_hz': {k:round(v, 3) for k,v in raw.items()},
            'filtered_rates_hz': {k:round(v, 3) for k,v in f.items()},
            'gains': gains.tolist(), 'silenced': bool(silence),
            # Lossless sparse counts from every neuron, in this completed
            # 20 ms bin. No subsampling, interpolated spikes, or activity trails.
            'activity': {'indices': active.tolist(), 'spike_counts': delta[active].tolist(),
                         'input_indices': self.inputs[rates > 0].tolist()},
            'top_neurons': [{'index':int(i), 'id':self.ids[i], 'spikes':int(delta[i]), 'type':self.annotations.get(self.ids[i],{}).get('cell_type','')} for i in top],
        }
