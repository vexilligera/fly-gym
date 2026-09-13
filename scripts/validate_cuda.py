"""Check CUDA against Brian2 under identical, explicit input spike schedules."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import brian2 as b
import cupy as cp
from brain.cuda_engine import CudaEngine


def main():
    b.prefs.codegen.target = 'numpy'
    b.defaultclock.dt = .1*b.ms
    pre = np.array([0, 0, 1, 2, 2, 3, 4, 5], dtype=np.int32)
    post = np.array([2, 4, 2, 3, 5, 4, 5, 2], dtype=np.int32)
    weights = np.array([100, 60, -90, 120, 70, -40, 90, 10], dtype=np.int32)
    inputs = np.array([0, 1], dtype=np.int32)
    gpu = CudaEngine(6, pre, post, weights, inputs)
    neurons = b.NeuronGroup(6, '''dv/dt=(-52*mV-v+g)/(20*ms):volt (unless refractory)
        dg/dt=-g/(5*ms):volt (unless refractory)
        rfc:second
        silenced:boolean''', threshold='v > -45*mV and not silenced',
        reset='v=-52*mV;g=0*mV', refractory='rfc', method='linear')
    neurons.v = -52*b.mV
    neurons.rfc = 2.2*b.ms
    syn = b.Synapses(neurons, neurons, 'w:volt', on_pre='g_post += w', delay=1.8*b.ms)
    syn.connect(i=pre, j=post)
    syn.w = weights*.275*b.mV
    source = b.SpikeGeneratorGroup(2, [], []*b.ms)
    inject = b.Synapses(source, neurons, on_pre='v_post += 68.75*mV')
    inject.connect(i=[0, 1], j=inputs)
    monitor = b.SpikeMonitor(neurons, record=False)
    network = b.Network(neurons, syn, source, inject, monitor)
    previous = np.zeros(6, dtype=np.int32)
    checks = []
    rng = np.random.default_rng(11)
    cases = [(False, False), (True, False), (True, True), (False, False), (True, False)]
    for drive, silence in cases:
        rates = np.array([250., 180.]) if drive else np.zeros(2)
        events = (rng.random((200, 2)) < rates[None, :]*.0001)
        if drive: events[[0, 1, 18, 21, 22, 199], 0] = True
        row, col = np.nonzero(events)
        source.set_spikes(col, (gpu.steps + row)*.1*b.ms)
        neurons.rfc[inputs] = np.where(rates > 0, 0, 2.2)*b.ms
        neurons.silenced = False
        neurons.silenced[3] = silence
        network.run(20*b.ms, namespace={})
        expected = np.asarray(monitor.count[:], dtype=np.int32) - previous
        previous += expected
        actual = gpu.step(rates, [3] if silence else [], events=events)
        voltage_error = float(np.max(np.abs(cp.asnumpy(gpu.v) - np.asarray(neurons.v[:]/b.mV))))
        conductance_error = float(np.max(np.abs(cp.asnumpy(gpu.g) - np.asarray(neurons.g[:]/b.mV))))
        assert np.array_equal(actual, expected), (actual, expected)
        assert voltage_error < 1e-9 and conductance_error < 1e-9, (voltage_error, conductance_error)
        checks.append({'drive':drive, 'silence':silence, 'counts':actual.tolist(),
                       'voltage_error_mV':voltage_error, 'conductance_error_mV':conductance_error})
    gpu.reset()
    assert gpu.steps == 0 and np.all(cp.asnumpy(gpu.v) == -52)
    assert not gpu.step(np.zeros(2), []).any()
    result = {'passed':True, 'gpu':gpu.metadata, 'identical_input_schedule_checks':checks,
              'scope':'Numerical/software comparison on an excitatory/inhibitory recurrent test network, including delays across batches, refractory changes, silencing and reset.'}
    output = Path(__file__).resolve().parents[1]/'outputs/cuda-validation.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
