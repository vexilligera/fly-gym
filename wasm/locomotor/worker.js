import { LocomotorExperiment } from './loop.js';
let experiment, graph, running = false, speed = 1, previous = performance.now();
let lastPublish = 0;
function publish(extra = {}) {
  if (experiment) postMessage({ state: experiment.snapshot(), running, speed, ...extra });
}
self.onmessage = async ({ data }) => {
  try {
    if (data.action === 'init') {
      const response = await fetch('./vendor/locomotor_circuit.json');
      if (!response.ok) throw new Error('Cannot load the MaleCNS circuit');
      graph = await response.json();
      experiment = new LocomotorExperiment(graph);
      publish({ ready: true });
    } else if (data.action === 'reset') {
      running = false;
      experiment = new LocomotorExperiment(graph, data.protocol);
      publish({ reset: true });
    } else if (data.action === 'run') {
      if (!experiment) throw new Error('Circuit still loading');
      if (experiment.done) experiment = new LocomotorExperiment(graph, experiment.protocol);
      running = true;
      previous = performance.now();
      publish();
    } else if (data.action === 'pause') {
      running = false; publish();
    } else if (data.action === 'speed') {
      if (![0.25, 1, 2].includes(data.value)) throw new Error('Unsupported playback speed');
      speed = data.value;
    } else throw new Error('Unknown command');
  } catch (error) { running = false; postMessage({ error: error.message }); }
};
setInterval(() => {
  const now = performance.now();
  if (running && experiment) {
    try {
      // Hidden/overloaded tabs slow model time instead of inventing skipped states.
      experiment.advance(Math.min(0.05, (now - previous) / 1000) * speed);
      if (experiment.done) running = false;
      if (now - lastPublish >= 50 || !running) {
        publish(); lastPublish = now;
      }
    } catch (error) { running = false; postMessage({ error: error.message }); }
  }
  previous = now;
}, 8);
