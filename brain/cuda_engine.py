"""Interactive CUDA execution of the existing 0.1 ms LIF model.

This is a local execution backend, not another connectome or a trained model.
All connection rows are retained in outgoing CSR form. Integer synapse-count
accumulation makes parallel delivery order independent. Voltage/conductance use
float64 and the exact linear update, with Brian2's refractory/event schedule.
"""
import numpy as np
import cupy as cp


KERNELS = r'''
extern "C" __global__ void prepare(const long long* step, int* sizes) {
    sizes[*step % 19] = 0;
}

extern "C" __global__ void update(
    int n, const long long* step, double* v, double* g,
    long long* last_spike, const int* refractory, const unsigned char* silenced,
    unsigned char* eligible, int* incoming, int* history, int* sizes, int* counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long t = *step;
    bool active = t - last_spike[i] >= refractory[i];
    if (active) {
        const double em = 0.9950124791926823; // exp(-0.1/20)
        const double es = 0.9801986733067553; // exp(-0.1/5)
        v[i] = -52.0 + (v[i] + 52.0) * em + g[i] * (em-es) / 3.0;
        g[i] *= es;
    }
    if (active && !silenced[i] && v[i] > -45.0) {
        int slot = t % 19;
        int j = atomicAdd(sizes + slot, 1);
        history[slot*n+j] = i;
        counts[i]++;
        last_spike[i] = t;
        v[i] = -52.0;
        g[i] = 0.0;
        active = false; // incoming writes are masked after this threshold event
    }
    eligible[i] = active;
    incoming[i] = 0;
}

extern "C" __global__ void deliver(
    int n, const long long* step, const int* history, const int* sizes,
    const int* row, const int* post, const int* weight, int* incoming) {
    int slot = (*step + 1) % 19; // spikes emitted 18 timesteps (1.8 ms) ago
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = thread & 31;
    int warp = thread >> 5;
    int warp_count = gridDim.x * blockDim.x / 32;
    for (int j = warp; j < sizes[slot]; j += warp_count) {
        int pre = history[slot*n+j];
        for (int edge = row[pre] + lane; edge < row[pre+1]; edge += 32) {
            atomicAdd(incoming + post[edge], weight[edge]);
        }
    }
}

extern "C" __global__ void apply_inputs(
    int n, int m, int batch_step, long long* step, double* v, double* g,
    const unsigned char* eligible, const int* incoming, const int* input_slot,
    const unsigned char* events) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && eligible[i]) {
        g[i] += incoming[i] * 0.275;
        int slot = input_slot[i];
        if (slot >= 0 && events[batch_step*m+slot]) v[i] += 68.75;
    }
    // No other thread in this kernel reads the clock.
    if (i == 0) ++*step;
}
'''


class CudaEngine:
    def __init__(self, n, pre, post, weights, inputs, primary_inputs=None):
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('The CUDA backend requires a GPU allocated to this process.')
        if not np.equal(weights, np.rint(weights)).all():
            raise ValueError('CUDA delivery requires the release signed integer synapse counts.')
        if np.abs(weights).sum() >= np.iinfo(np.int32).max:
            raise ValueError('Synapse-count sum exceeds the integer delivery accumulator.')
        self.n, self.m = n, len(inputs)
        self.inputs_host = np.asarray(inputs, dtype=np.int32)
        # Additional sensory channels must not shift the existing navigation
        # input random stream, even while the new channels are inactive.
        primary = self.inputs_host if primary_inputs is None else np.asarray(primary_inputs)
        slot_by_id = {int(i): slot for slot, i in enumerate(self.inputs_host)}
        self.primary_slots = np.array([slot_by_id[int(i)] for i in primary], dtype=np.int32)
        self.secondary_slots = np.setdiff1d(np.arange(self.m), self.primary_slots)
        order = np.argsort(pre, kind='stable')
        row = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(np.bincount(pre, minlength=n), out=row[1:])
        self.row = cp.asarray(row)
        self.post = cp.asarray(post[order], dtype=cp.int32)
        self.weights = cp.asarray(weights[order], dtype=cp.int32)
        self.inputs = cp.asarray(self.inputs_host)
        input_slot = np.full(n, -1, dtype=np.int32)
        input_slot[self.inputs_host] = np.arange(self.m)
        self.input_slot = cp.asarray(input_slot)
        self.v, self.g = cp.empty(n, cp.float64), cp.empty(n, cp.float64)
        self.last_spike = cp.empty(n, cp.int64)
        self.refractory = cp.empty(n, cp.int32)
        self.silenced = cp.empty(n, cp.uint8)
        self.eligible = cp.empty(n, cp.uint8)
        self.incoming = cp.empty(n, cp.int32)
        self.history = cp.empty((19, n), cp.int32)
        self.sizes = cp.empty(19, cp.int32)
        self.counts = cp.empty(n, cp.int32)
        self.clock = cp.empty(1, cp.int64)
        self.events = cp.empty((200, self.m), cp.uint8)
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.reset()
        module = cp.RawModule(code=KERNELS, options=('--std=c++11', '--fmad=false'))
        prepare, update, deliver, apply = [module.get_function(name) for name in ('prepare', 'update', 'deliver', 'apply_inputs')]
        blocks = ((n + 255) // 256,)
        with self.stream:
            self.stream.begin_capture()
            for batch_step in range(200):
                prepare((1,), (1,), (self.clock, self.sizes))
                update(blocks, (256,), (np.int32(n), self.clock, self.v, self.g, self.last_spike,
                    self.refractory, self.silenced, self.eligible, self.incoming, self.history, self.sizes, self.counts))
                deliver((128,), (128,), (np.int32(n), self.clock, self.history, self.sizes, self.row, self.post, self.weights, self.incoming))
                apply(blocks, (256,), (np.int32(n), np.int32(self.m), np.int32(batch_step), self.clock,
                    self.v, self.g, self.eligible, self.incoming, self.input_slot, self.events))
            self.graph = self.stream.end_capture()
        props = cp.cuda.runtime.getDeviceProperties(0)
        self.metadata = {
            'gpu': props['name'].decode(), 'cupy_version': cp.__version__,
            'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
            'integration': 'Float64 exact linear LIF, 0.1 ms, 1.8 ms delay, Brian2 refractory/event order',
            'random_input': 'Independent Bernoulli rate*dt events; seed 42 for existing inputs, independent seed 43 for added sensory channels; differs from Brian2',
        }

    def reset(self):
        self.stream.synchronize()
        self.v.fill(-52)
        for a in (self.g, self.clock, self.sizes, self.counts, self.silenced, self.events):
            a.fill(0)
        self.refractory.fill(22)
        self.last_spike.fill(-1000000000)
        cp.cuda.get_current_stream().synchronize()
        self.steps = 0
        self.rng = np.random.default_rng(42)
        self.secondary_rng = np.random.default_rng(43)

    def step(self, rates, silenced, events=None):
        rates = np.asarray(rates)
        if events is None:
            events = np.zeros((200, self.m), dtype=np.uint8)
            events[:, self.primary_slots] = self.rng.random((200, len(self.primary_slots))) < rates[None, self.primary_slots] * .0001
            events[:, self.secondary_slots] = self.secondary_rng.random((200, len(self.secondary_slots))) < rates[None, self.secondary_slots] * .0001
        events = np.ascontiguousarray(events, dtype=np.uint8)
        if events.shape != (200, self.m):
            raise ValueError('Expected 200 input-event rows for the 20 ms batch.')
        # Tiny input/control arrays transfer once per batch, not per timestep.
        self.events.set(events)
        self.refractory[self.inputs] = cp.asarray(np.where(rates > 0, 0, 22), dtype=cp.int32)
        self.silenced.fill(0)
        if len(silenced): self.silenced[cp.asarray(silenced, dtype=cp.int32)] = 1
        self.counts.fill(0)
        cp.cuda.get_current_stream().synchronize()
        self.graph.launch(self.stream)
        self.stream.synchronize()
        self.steps += 200
        return cp.asnumpy(self.counts)
