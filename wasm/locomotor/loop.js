// The pinned DesktopFly neural and mechanical modules are unchanged.
// This wrapper adds an isolated, deterministic experiment clock and measurements.
import { LocomotorSim } from './vendor/locomotor.js';
import { SixLegDynamics, LegDynamics } from './vendor/legdynamics.js';

export const LEG_NAMES = ['RF', 'LF', 'RM', 'LM', 'RH', 'LH'];
export const GEOMETRIES = [
  [1, 3.1, 5.3, 0.95, 4.2, 4.8, 3.2],
  [-1, -3.1, 5.3, 0.95, 4.2, 4.8, 3.2],
  [1, 3.7, 2, -0.10, 4.8, 5.6, 3.8],
  [-1, -3.7, 2, -0.10, 4.8, 5.6, 3.8],
  [1, 3.3, -1.2, -0.95, 5.8, 7, 4.6],
  [-1, -3.3, -1.2, -0.95, 5.8, 7, 4.6],
].map(([side, attachX, attachY, yaw, femur, tibia, tarsus]) => ({
  side, attachX, attachY, attachZ: 4.5,
  baseYaw: side > 0 ? yaw : Math.PI - yaw, femur, tibia, tarsus,
}));

export const PROTOCOLS = {
  forward: { label: 'Forward · DNp09', forwardHz: 40, duration: 10 },
  backward: { label: 'Backward · MDN', forwardHz: 0, backwardHz: 70, duration: 10 },
  baseline: { label: 'Steering baseline · DNp09', forwardHz: 30, duration: 10 },
  left: { label: 'Left steering after 3 s', forwardHz: 30, leftHz: 70, onset: 3, duration: 10 },
  right: { label: 'Right steering after 3 s', forwardHz: 30, rightHz: 70, onset: 3, duration: 10 },
  quiet: { label: 'No descending or sensory input', forwardHz: 0, feedback: false, duration: 10 },
  cut: { label: 'Cut synaptic transmission', forwardHz: 40, synapses: false, duration: 10 },
  silenced: { label: 'Silence motor neurons', forwardHz: 40, motor: false, duration: 10 },
  open: { label: 'Disconnect sensory feedback', forwardHz: 40, feedback: false, duration: 10 },
};

// Joint endpoints use exactly the geometry used by the physical feedback.
export function legPoints(g, f) {
  const yaw = g.baseYaw + g.side * f.hipAngle;
  const points = [[g.attachX, g.attachY, g.attachZ]];
  const lengths = [g.femur, g.tibia, g.tarsus];
  const angles = [f.elevationAngle, f.elevationAngle - f.kneeAngle,
    f.elevationAngle - f.kneeAngle - LegDynamics.ankleAngle];
  for (let i = 0; i < 3; i++) {
    const p = points.at(-1), horizontal = lengths[i] * Math.cos(angles[i]);
    points.push([p[0] + Math.cos(yaw) * horizontal,
      p[1] + Math.sin(yaw) * horizontal, p[2] + lengths[i] * Math.sin(angles[i])]);
  }
  return points;
}

export class LocomotorExperiment {
  constructor(graph, protocol = 'forward') {
    if (!Object.hasOwn(PROTOCOLS, protocol)) throw new Error('Unknown experiment');
    this.graph = graph;
    this.protocol = protocol;
    this.config = { ...PROTOCOLS[protocol] };
    this.cord = new LocomotorSim(graph);
    this.body = new SixLegDynamics(GEOMETRIES);
    this.cord.synapsesEnabled = this.config.synapses !== false;
    this.cord.feedbackEnabled = this.config.feedback !== false;
    if (this.config.motor === false) this.cord.silenced = new Set(this.cord.indices('motor'));
    this.tick = 0;
    this.accumulator = 0;
    this.position = [0, 0];
    this.yaw = 0;
    this.path = [[0, 0]];
    this.trace = [];
    this.summary = { forward: 0, lateral: 0, yaw: 0, lateForward: 0, lateYaw: 0,
      latePath: 0, contacts: Array(6).fill(0), motorSpikes: 0, sensorySpikes: 0,
      networkSpikes: 0, finite: true, maxClockLagMs: 0 };
    this.roleIndices = new Map();
    graph.neurons.forEach((n, i) => {
      if (!this.roleIndices.has(n.role)) this.roleIndices.set(n.role, []);
      this.roleIndices.get(n.role).push(i);
    });
  }

  get time() { return this.tick / 120; }
  get done() { return this.tick >= Math.round(this.config.duration * 120); }

  step() {
    if (this.done) return;
    const { config: p, cord, body } = this;
    for (const side of ['left', 'right']) {
      cord.setDescending('DNp09', side, p.forwardHz || 0);
      const afterOnset = this.time + 1e-9 >= (p.onset || 0);
      cord.setDescending('MDN', side, afterOnset ? p.backwardHz || 0 : 0);
      for (const type of ['DNa01', 'DNa02']) {
        cord.setDescending(type, side, afterOnset ? p[side + 'Hz'] || 0 : 0);
      }
    }
    const before = body.feedback;
    cord.feedback = before;
    // Exact 8, 8, 9 ms schedule: neuron and body clocks agree every 25 ms.
    const targetMs = Math.floor((this.tick + 1) * 1000 / 120 + 1e-9);
    cord.step(targetMs - cord.simMs);
    const movement = body.advance(cord.commands, 1 / 120);
    const late = this.time + 1e-9 >= 3;
    this.position[0] += movement.lateral * Math.cos(this.yaw) - movement.forward * Math.sin(this.yaw);
    this.position[1] += movement.lateral * Math.sin(this.yaw) + movement.forward * Math.cos(this.yaw);
    this.yaw += movement.yaw;
    const s = this.summary;
    s.forward += movement.forward; s.lateral += movement.lateral; s.yaw += movement.yaw;
    if (late) {
      s.lateForward += movement.forward;
      s.lateYaw += movement.yaw;
      s.latePath += Math.hypot(movement.forward, movement.lateral);
    }
    const feedback = body.feedback;
    feedback.forEach((f, i) => {
      if (late && f.contact && !before[i].contact) s.contacts[i]++;
      s.finite &&= Object.values(f).every((v) => typeof v === 'boolean' || Number.isFinite(v));
      s.finite &&= f.footHeight >= -1e-8;
    });
    s.motorSpikes = cord.motorSpikes;
    s.sensorySpikes = cord.sensorySpikes;
    s.networkSpikes = cord.totalSpikes;
    this.tick++;
    s.maxClockLagMs = Math.max(s.maxClockLagMs, this.time * 1000 - cord.simMs);
    if (!s.finite || !this.position.every(Number.isFinite)) throw new Error('Nonfinite mechanical state');
    if (this.tick % 6 === 0) {
      this.path.push([...this.position]);
      const lf = cord.commands[1];
      this.trace.push({ time: this.time, flex: lf.flex, extend: lf.extend,
        knee: feedback[1].kneeAngle, contact: feedback[1].contact });
    }
  }

  advance(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0 || seconds > 1) throw new Error('Invalid clock increment');
    this.accumulator += seconds;
    while (this.accumulator + 1e-10 >= 1 / 120 && !this.done) {
      this.accumulator -= 1 / 120;
      this.step();
    }
  }

  snapshot() {
    return { protocol: this.protocol, time: this.time, neuralMs: this.cord.simMs, done: this.done,
      position: [...this.position], yaw: this.yaw, feedback: this.body.feedback,
      commands: this.cord.commands.map((c) => ({ ...c })), rates: this.cord.rates.slice(),
      roleRates: [...this.roleIndices].map(([role, ids]) => ({ role, cells: ids.length,
        rate: ids.reduce((s, i) => s + this.cord.rates[i], 0) / ids.length })),
      summary: { ...this.summary, contacts: [...this.summary.contacts] },
      path: this.path, trace: this.trace };
  }
}
