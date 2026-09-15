import { BodyView } from './body-view.js';
import { PROTOCOLS, LEG_NAMES } from './loop.js';
const $ = (id) => document.getElementById(id);
const graph = await fetch('vendor/locomotor_circuit.json').then((r) => {
  if (!r.ok) throw new Error('Circuit download failed'); return r.json();
});
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let view, running = false, state, selectedNeuron = null, cells = [];
try { view = new BodyView($('stage')); }
catch (error) { $('stage-error').textContent = '3D view unavailable: ' + error.message; }
for (const [key, protocol] of Object.entries(PROTOCOLS)) {
  const option = new Option(protocol.label, key); $('protocol').add(option);
}
const tbody = $('motor-table').tBodies[0];
for (const leg of LEG_NAMES) {
  const row = tbody.insertRow();
  const name = row.insertCell(); name.innerHTML = '<i class="contact"></i>' + leg;
  for (let i = 0; i < 6; i++) { const cell = row.insertCell(); cell.className = 'activation'; cell.textContent = '0.00'; }
}
function send(action, extra = {}) { worker.postMessage({ action, ...extra }); }
$('run').onclick = () => send(running ? 'pause' : 'run');
$('reset').onclick = () => send('reset', { protocol: $('protocol').value });
$('protocol').onchange = () => send('reset', { protocol: $('protocol').value });
$('speed').onchange = () => send('speed', { value: Number($('speed').value) });
$('camera').onclick = () => view?.resetCamera();
document.addEventListener('visibilitychange', () => { if (document.hidden && running) send('pause'); });
worker.onerror = (error) => {
  running = false; $('run-state').textContent = 'Simulation error';
  $('stage-error').textContent = error.message; $('run').disabled = true;
};
const colors = { descending: '#5288b2', premotor: '#7b73b2', motor: '#c27535', sensory: '#488776', ascending: '#997aa1' };
const names = { descending: 'Descending', premotor: 'Interneurons', motor: 'Motor', sensory: 'Sensory', ascending: 'Ascending' };
const roles = Object.keys(colors);
const neuralCanvas = $('neural-map'), ctx = neuralCanvas.getContext('2d');
const ordered = roles.map((role) => ({ role, indices: graph.neurons.map((n, i) => n.role === role ? i : -1).filter((i) => i >= 0) }));
function drawNeurons() {
  ctx.clearRect(0, 0, neuralCanvas.width, neuralCanvas.height); cells = [];
  let y = 16;
  for (const { role, indices } of ordered) {
    ctx.fillStyle = '#68776b'; ctx.font = '11px system-ui';
    const rate = state.roleRates.find((r) => r.role === role).rate;
    ctx.fillText(names[role] + ' · ' + indices.length + ' cells · mean ' + rate.toFixed(1) + ' Hz', 5, y);
    y += 7;
    indices.forEach((idx, n) => {
      const x = 5 + n % 48 * 11, top = y + Math.floor(n / 48) * 10;
      ctx.fillStyle = '#eef1e9'; ctx.fillRect(x, top, 8, 7);
      ctx.globalAlpha = Math.min(1, state.rates[idx] / 80);
      ctx.fillStyle = colors[role]; ctx.fillRect(x, top, 8, 7); ctx.globalAlpha = 1;
      if (idx === selectedNeuron) { ctx.strokeStyle = '#243e2c'; ctx.strokeRect(x - 1, top - 1, 10, 9); }
      cells.push({ x, y: top, idx });
    });
    y += Math.ceil(indices.length / 48) * 10 + 17;
  }
  if (selectedNeuron !== null) {
    const n = graph.neurons[selectedNeuron];
    $('neuron-detail').textContent = n.id + ' · ' + n.type + ' · ' + names[n.role] +
      (n.leg === null ? '' : ' · ' + LEG_NAMES[n.leg]) + ' · ' + state.rates[selectedNeuron].toFixed(1) +
      ' Hz filtered rate. ' + (n.motorChannel || (n.role === 'sensory' ? 'Individual joint/direction tuning unresolved.' : n.side + ' side.'));
  }
}
neuralCanvas.onclick = (event) => {
  const box = neuralCanvas.getBoundingClientRect();
  const x = (event.clientX - box.left) * neuralCanvas.width / box.width;
  const y = (event.clientY - box.top) * neuralCanvas.height / box.height;
  const cell = cells.find((c) => x >= c.x - 1 && x <= c.x + 9 && y >= c.y - 1 && y <= c.y + 8);
  if (cell) { selectedNeuron = cell.idx; drawNeurons(); }
};
function chart(id, series, ymax, unit) {
  const el = $(id), width = 560, left = 38, right = 14, top = 18, height = 140;
  let svg = '';
  for (let i = 0; i <= 4; i++) {
    const y = top + height * (1 - i / 4);
    svg += '<line x1="38" y1="' + y + '" x2="546" y2="' + y + '" stroke="#e8ece4"/>';
    svg += '<text x="30" y="' + (y + 3) + '" text-anchor="end" fill="#7a8478">' + (ymax * i / 4).toFixed(1) + '</text>';
  }
  for (let i = 0; i <= 5; i++) {
    const x = left + (width - left - right) * i / 5;
    svg += '<text x="' + x + '" y="179" text-anchor="middle" fill="#7a8478">' + i * 2 + 's</text>';
  }
  svg += '<text x="38" y="10" fill="#667461">' + unit + '</text>';
  for (const { field, color } of series) {
    const d = state.trace.map((r, i) => (i ? 'L' : 'M') + (left + r.time / 10 * (width - left - right)).toFixed(2) +
      ',' + (top + height * (1 - r[field] / ymax)).toFixed(2)).join(' ');
    if (d) svg += '<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="1.6"/>';
  }
  el.innerHTML = svg;
}
function stimulusText() {
  const p = PROTOCOLS[state.protocol];
  if (state.protocol === 'quiet') return 'No descending stimulation. Sensory feedback disabled to measure the unpowered baseline.';
  let s = p.forwardHz ? 'DNp09: ' + p.forwardHz + ' Hz-equivalent bilateral drive. ' : '';
  if (p.backwardHz) s += 'MDN: ' + p.backwardHz + ' Hz-equivalent bilateral drive. ';
  if (p.leftHz || p.rightHz) s += 'DNa01 + DNa02: ' + (p.leftHz ? 'left' : 'right') + ' stimulation begins at 3 s. ';
  if (p.synapses === false) s += 'Synaptic transmission disabled.';
  if (p.motor === false) s += 'All 220 motor neurons silenced.';
  if (p.feedback === false) s += 'Joint/contact feedback disconnected.';
  return s + ' Stimulation sets a modeled input current, not a guaranteed firing rate.';
}
worker.onmessage = ({ data }) => {
  if (data.error) {
    running = false;
    $('stage-error').textContent = data.error; $('run-state').textContent = 'Simulation stopped';
    $('run').textContent = 'Run experiment'; return;
  }
  state = data.state; running = data.running;
  for (const id of ['run', 'reset', 'protocol', 'speed']) $(id).disabled = false;
  $('run').textContent = running ? 'Pause' : state.done ? 'Run again' : 'Run experiment';
  $('run-state').textContent = running ? 'Running in your browser' : state.done ? '10 s experiment complete' : 'Paused';
  $('clock').textContent = state.time.toFixed(3) + ' s';
  $('neural-clock').textContent = state.neuralMs + ' ms neural time';
  $('motor-spikes').textContent = state.summary.motorSpikes.toLocaleString();
  $('sensory-spikes').textContent = state.summary.sensorySpikes.toLocaleString();
  $('distance').textContent = state.summary.forward.toFixed(2);
  $('stimulus-note').textContent = stimulusText();
  view?.update(state, $('follow').checked);
  if (data.reset) view?.resetCamera();
  drawNeurons();
  const channels = ['flex', 'extend', 'lift', 'depress', 'protract', 'retract'];
  state.commands.forEach((command, i) => {
    tbody.rows[i].cells[0].querySelector('i').classList.toggle('on', state.feedback[i].contact);
    channels.forEach((key, j) => {
      const cell = tbody.rows[i].cells[j + 1];
      cell.textContent = command[key].toFixed(2);
      cell.style.setProperty('--fill', (100 * command[key]).toFixed(1) + '%');
    });
  });
  chart('knee-chart', [{ field: 'knee', color: '#5288b2' }], 1.8, 'radians · flexion-positive joint coordinate');
  chart('activation-chart', [{ field: 'flex', color: '#c27535' }, { field: 'extend', color: '#488776' }], 1, 'activation');
};
send('init');
function render() { view?.render(); requestAnimationFrame(render); }
render();
try {
  const report = await fetch('report.json').then((r) => { if (!r.ok) throw Error(); return r.json(); });
  $('validation-summary').textContent = report.passed
    ? 'Reference reproduction and causal controls passed. Sustained stepping follows descending stimulation; disconnecting synapses or silencing motor neurons abolishes propulsion. Sensory feedback changes neural activity.'
    : 'Some control checks failed; see the report before interpreting the experiment.';
  for (const row of report.trials) {
    const tr = $('result-rows').insertRow();
    const values = [row.label, row.summary.motorSpikes.toLocaleString(), row.summary.lateForward.toFixed(2),
      row.summary.contacts.join(' / ')];
    for (const value of values) tr.insertCell().textContent = value;
  }
} catch { $('validation-summary').textContent = 'Control results could not be loaded. The live experiment remains available.'; }
try {
  const report = await fetch('transfer-report.json').then((r) => { if (!r.ok) throw Error(); return r.json(); });
  $('transfer-summary').textContent = report.summary;
  if (report.videos) {
    $('transfer-view').hidden = false;
    $('transfer-active').src = report.videos.connected;
    $('transfer-passive').src = report.videos.motor_silenced;
  }
} catch { $('transfer-summary').textContent = 'The MuJoCo transfer report is unavailable. The six-leg reference above uses simplified DesktopFly mechanics.'; }
