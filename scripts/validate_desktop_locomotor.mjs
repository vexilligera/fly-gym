import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { LocomotorExperiment, PROTOCOLS, GEOMETRIES, legPoints } from '../wasm/locomotor/loop.js';
import { LocomotorSim, validateLocomotorCircuit } from '../wasm/locomotor/vendor/locomotor.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const page = root + 'wasm/locomotor/';
const graph = JSON.parse(fs.readFileSync(page + 'vendor/locomotor_circuit.json'));
const expected = {
  'locomotor_circuit.json': '8f76d94034dcf802453e3a0a8ed5342d122e57d37e2bb5ea28da66de0856f5d6',
  'locomotor.js': 'a00ad3935a555c09192f835a54ca89edc6173e5cf93c83797cd6b5bfc6123d2d',
  'legdynamics.js': 'b2315b40a6e9b024b5416bb3a0536cab2b74aa81ec77b85742815e0a61d283b1',
};
for (const [name, sha] of Object.entries(expected)) {
  assert.equal(crypto.createHash('sha256').update(fs.readFileSync(page + 'vendor/' + name)).digest('hex'), sha);
}
assert(validateLocomotorCircuit(graph));
assert.equal(graph.neurons.length, 1045); assert.equal(graph.edges.length, 17224);
assert.equal(graph.rawSynapseCounts.reduce((a, b) => a + b, 0), 708689);
assert(!validateLocomotorCircuit({ ...graph, edges: [[1045, 0, 1]] }));
for (const id of ['815843', '817680', '912317', '935383']) {
  const cell = graph.neurons.find((n) => n.id === id);
  assert.equal(cell.leg, 1); assert.equal(cell.sensoryDirection, null); assert.equal(cell.sensoryJoint, null);
}
const trials = [];
for (const key of Object.keys(PROTOCOLS)) {
  const experiment = new LocomotorExperiment(graph, key);
  while (!experiment.done) experiment.step();
  assert(experiment.summary.finite);
  assert.equal(experiment.cord.simMs, 10000);
  assert(experiment.summary.maxClockLagMs < 1);
  experiment.body.feedback.forEach((f, i) => {
    const toe = legPoints(GEOMETRIES[i], f).at(-1);
    assert(Math.hypot(toe[0] - f.footX, toe[1] - f.footY, toe[2] - f.footHeight) < 1e-12);
  });
  trials.push({ key, label: PROTOCOLS[key].label, duration_s: 10, summary: experiment.summary });
}
const trial = (key) => trials.find((x) => x.key === key).summary;
assert(trial('forward').lateForward > 5);
assert(trial('forward').contacts.every((n) => n >= 2));
assert(trial('backward').lateForward < -5);
assert(trial('left').lateYaw - trial('baseline').lateYaw > .1);
assert(trial('right').lateYaw - trial('baseline').lateYaw < -.1);
for (const key of ['cut', 'silenced', 'quiet']) {
  assert.equal(trial(key).motorSpikes, 0);
  assert(trial(key).latePath < 1e-10);
}
assert(trial('forward').sensorySpikes > 0);
assert.equal(trial('open').sensorySpikes, 0);
assert.notEqual(trial('forward').motorSpikes, trial('open').motorSpikes);

// Published upstream native result, rather than a second copy of our implementation.
const reference = new LocomotorExperiment(graph);
reference.config.duration = 8;
while (!reference.done) reference.step();
assert.equal(reference.summary.motorSpikes, 12868);
assert.equal(reference.summary.sensorySpikes, 34593);
assert.deepEqual(reference.summary.contacts, [20, 22, 19, 11, 16, 14]);
assert(Math.abs(reference.summary.forward - 10.643657) < 1e-6);

const schedules = [30, 60, 120].map((hz) => {
  const run = new LocomotorExperiment(graph);
  for (let i = 0; i < hz * 10; i++) run.advance(1 / hz);
  return { summary: run.summary, feedback: run.body.feedback, rates: [...run.cord.rates], position: run.position };
});
assert.deepEqual(schedules[0], schedules[1]); assert.deepEqual(schedules[1], schedules[2]);
const reset = new LocomotorExperiment(graph);
assert.equal(reset.time, 0); assert.equal(reset.cord.totalSpikes, 0);
assert.deepEqual(reset.position, [0, 0]); assert.equal(reset.trace.length, 0);

// Export a fixed-input upstream neural reference for an independent Python port.
const neural = new LocomotorSim(graph);
for (const side of ['left', 'right']) neural.setDescending('DNp09', side, 40);
neural.feedback = GEOMETRIES.map(() => ({ hipAngle: 0, hipVelocity: 0, kneeAngle: .95,
  kneeVelocity: 0, contact: false, load: 0, elevationVelocity: 0 }));
neural.feedback[1].kneeVelocity = 4;
neural.step(2000);
const neuralReference = { duration_ms: 2000, voltage: [...neural.voltage],
  rates: [...neural.rates], adaptation: [...neural.adaptation], excitatory: [...neural.excitatory],
  inhibitory: [...neural.inhibitory], refractory: [...neural.refractory],
  total_spikes: neural.totalSpikes, motor_spikes: neural.motorSpikes, sensory_spikes: neural.sensorySpikes };
fs.mkdirSync(root + 'outputs/desktop-locomotor', { recursive: true });
fs.writeFileSync(root + 'outputs/desktop-locomotor/neural-reference.json', JSON.stringify(neuralReference));
const report = { passed: true, runtime: { node: process.version, platform: process.platform },
  source_revision: '32b00011e83c3dc85fa3ea0b3934155b04f1635d',
  tests: ['pinned upstream code and data hashes', 'valid graph and raw contacts',
    'four sensory cells remain unresolved', 'forward recruitment and sustained contacts',
    'backward propulsion', 'paired steering after identical prefix',
    'synaptic disconnection', 'motor silencing', 'no-input baseline',
    'feedback changes activity', 'reference 8-second native result',
    '30/60/120 Hz frame independence', 'rendered endpoints equal sensed endpoints',
    'reset of all model state', 'finite mechanics and sub-millisecond neural clock lag'],
  reference_8s: reference.summary, trials,
  biological_validation: false, units: 'Body distances and forces are model units, not calibrated physical units.' };
fs.writeFileSync(page + 'report.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ passed: true, tests: report.tests, trials: trials.map((x) => ({
  key: x.key, motor_spikes: x.summary.motorSpikes, late_forward: x.summary.lateForward })) }, null, 2));
