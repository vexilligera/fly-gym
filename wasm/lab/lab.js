const $ = id => document.getElementById(id);
const frame = $('fly');
const names = ['Left front', 'Left middle', 'Left hind', 'Right front', 'Right middle', 'Right hind'];
const short = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH'];
const tripod = ['A', 'B', 'A', 'B', 'A', 'B'];
const order = [0, 3, 1, 4, 2, 5];
const drive = { walk:[1, 1], left:[.4, 1.2], right:[1.2, .4], back:[-1, -1], stop:[0, 0] };
let state = null, generation = -1, samples = [], demo = null, lastMessage = performance.now();
const lessons = {
  CPG: ['01 / THE BRAIN SETS INTENT', 'Press Walk. Equal left and right drive makes the fly walk forward. Turn left reduces drive on the left side; the six leg controllers handle the timing.'],
  tripod: ['02 / YOU TAKE OVER COORDINATION', 'Alternate Tripod A and Tripod B. Each command starts three legs together. You now supply the timing that the oscillator network supplied in the first mode.'],
  single: ['03 / YOU CONTROL EACH LEG', 'Try stepping individual legs. A command starts one full step; it does not coordinate the other five legs. This shows why distributing control across local circuits is useful.'],
};
const sequence = [
  { at:0, drive:'walk', text:'Equal drive → forward walking. The two descending values recruit all six leg oscillators.' },
  { at:.6, drive:'left', text:'Less drive on the left → a left turn. The local oscillators continue coordinating the legs.' },
  { at:1.2, drive:'right', text:'Less drive on the right → a right turn. Watch the target angle change and the physical joint follow.' },
  { at:1.8, drive:'stop', text:'Zero drive → step amplitude decays. The oscillators can keep cycling even as the joint targets settle.' },
];

$('oscillators').innerHTML = order.map(i => `<div class="oscillator" data-tripod="${tripod[i]}">
  <svg viewBox="0 0 44 44" role="img" aria-label="${names[i]} oscillator phase"><circle class="ring" cx="22" cy="22" r="18"/><circle id="core-${i}" class="core" cx="22" cy="22" r="13" opacity=".12"/><line id="arm-${i}" class="arm" x1="22" y1="22" x2="22" y2="5"/></svg>
  <div><span class="leg-label">${short[i]} <span style="color:var(--muted)">· ${names[i].split(' ')[1]}</span></span><span class="leg-state" id="leg-${i}">—</span></div>
</div>`).join('');
document.querySelectorAll('button').forEach(b => { b.disabled = true; });

function command(action, values = {}) {
  frame.contentWindow.postMessage({ type:'nmf-command', action, ...values }, location.origin);
}
function lesson(label, text) { $('lesson-label').textContent = label; $('lesson-text').textContent = text; }
function cancelDemo() { demo = null; $('demo').textContent = 'Show me the sequence'; }
function setDrive(name) {
  cancelDemo();
  command('drive', { left:drive[name][0], right:drive[name][1] });
  lesson(...lessons.CPG);
}
document.querySelectorAll('[data-drive]').forEach(b => b.onclick = () => setDrive(b.dataset.drive));
document.querySelectorAll('[data-mode]').forEach(b => b.onclick = () => {
  cancelDemo(); command('level', { level:b.dataset.mode });
});
$('pause').onclick = () => command('pause');
$('reset').onclick = () => { cancelDemo(); command('reset'); lesson(...lessons[state?.level || 'CPG']); };
$('demo').onclick = () => {
  if (demo) { cancelDemo(); command('drive', {left:0, right:0}); return; }
  demo = { waiting:true, previous:generation, index:-1 };
  command('level', { level:'CPG' });
  $('demo').textContent = 'End sequence';
};
addEventListener('keydown', e => {
  if (!state || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
  const k = e.key.toLowerCase();
  const mapping = { w:'walk', a:'left', d:'right', s:'back', q:'stop', arrowup:'walk', arrowleft:'left', arrowright:'right', arrowdown:'back' };
  if (state.level === 'CPG' && mapping[k]) { e.preventDefault(); setDrive(mapping[k]); }
});

function setModeUI(level) {
  document.querySelectorAll('[data-mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === level)));
  $('drive-controls').hidden = level !== 'CPG';
  $('step-controls').hidden = level === 'CPG';
  $('demo').hidden = level !== 'CPG';
  const labels = level === 'tripod' ? ['Step tripod A', 'Step tripod B'] : short.map(l => `Step ${l}`);
  $('step-controls').replaceChildren(...labels.map((label, index) => {
    const b = document.createElement('button'); b.textContent = label;
    b.onclick = () => command('step', { index }); return b;
  }));
  $('rhythm-title').textContent = level === 'CPG' ? 'Six coupled oscillators' : level === 'tripod' ? 'Two commanded step cycles' : 'Six independent step cycles';
  $('rhythm-detail').textContent = level === 'CPG'
    ? 'Each dial is one leg’s cycle. Opposite tripods tend to alternate. “Grip” is the adhesion command, not a contact measurement.'
    : 'These dials show commanded step phases. The coupled oscillator network is bypassed in this mode. “Grip” is the adhesion command.';
  $('drive-detail').textContent = level === 'CPG'
    ? 'Two abstract drive values. These represent your command, not measured neural spikes.'
    : 'Descending drive is bypassed. Your buttons directly trigger tripod or single-leg step cycles.';
  lesson(...lessons[level]);
}

addEventListener('message', e => {
  if (e.source !== frame.contentWindow || e.origin !== location.origin || e.data?.type !== 'nmf-state') return;
  const s = e.data;
  lastMessage = performance.now();
  if (!state) document.querySelectorAll('button').forEach(b => { b.disabled = false; });
  if (!state || s.level !== state.level) setModeUI(s.level);
  if (s.generation !== generation || (state && s.time < state.time)) {
    samples = []; generation = s.generation;
  }
  if (!Object.values(s.joint).filter(v => typeof v === 'number').every(Number.isFinite) || !s.position.every(Number.isFinite)) {
    $('connection').className = 'error'; $('connection').textContent = 'Simulation became unstable. Press Reset.'; return;
  }
  state = s;
  $('connection').className = 'ready';
  $('connection').textContent = s.paused ? '● Paused · local simulation' : '● Live · running on this computer';
  $('pause').textContent = s.paused ? 'Resume' : 'Pause';
  $('sim-time').textContent = `${s.time.toFixed(3)} s simulated`;
  $('position').textContent = `x ${s.position[0].toFixed(1)} · y ${s.position[1].toFixed(1)} mm`;
  ['left', 'right'].forEach((side, i) => {
    $('gain-' + side).textContent = s.level === 'CPG' ? s.gains[i].toFixed(2) : '—';
    $('bar-' + side).style.width = s.level === 'CPG' ? `${Math.min(1, Math.abs(s.gains[i]) / 1.2) * 100}%` : '0%';
  });
  for (let i = 0; i < 6; i++) {
    const a = s.phases[i] - Math.PI / 2;
    $('arm-' + i).setAttribute('x2', 22 + 17 * Math.cos(a));
    $('arm-' + i).setAttribute('y2', 22 + 17 * Math.sin(a));
    $('core-' + i).setAttribute('opacity', .08 + Math.min(1, s.magnitudes[i] / 1.2) * .3);
    $('leg-' + i).textContent = `${Math.round(s.phases[i] * 180 / Math.PI)}° · ${s.adhesion[i] ? 'grip' : 'release'}`;
  }
  $('joint-target').textContent = s.joint.target.toFixed(2);
  $('joint-actual').textContent = s.joint.actual.toFixed(2);
  $('contacts').textContent = s.contacts;
  if (!samples.length || s.time > samples.at(-1).time) samples.push({time:s.time, ...s.joint});
  samples = samples.filter(p => p.time >= s.time - .25);
  drawChart();
  if (demo && !s.paused) {
    if (demo.waiting && s.generation !== demo.previous) demo.waiting = false;
    if (!demo.waiting) {
      let index = sequence.findLastIndex(step => s.time >= step.at);
      if (index !== demo.index) {
        demo.index = index;
        const step = sequence[index];
        command('drive', {left:drive[step.drive][0], right:drive[step.drive][1]});
        lesson(`DEMONSTRATION / ${index + 1} OF 4`, step.text);
      }
      if (s.time >= 2.2) {
        cancelDemo(); command('pause');
        lesson('SEQUENCE COMPLETE / EXPLORE THE NEXT LEVEL', 'You just sent two commands while the local leg controllers coordinated 42 joint targets. Resume to experiment, or switch to Two tripods to take over the step timing yourself.');
      }
    }
  }
});

const chart = $('joint-chart');
function drawChart() {
  const rect = chart.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  const w = rect.width, h = rect.height;
  if (w < 70) return;
  chart.width = Math.round(w * dpr); chart.height = Math.round(h * dpr);
  const ctx = chart.getContext('2d'); ctx.scale(dpr, dpr);
  const left = 37, right = w - 6, top = 9, bottom = h - 22;
  const values = samples.flatMap(p => [p.target, p.actual]);
  const lo = values.length ? Math.min(...values) : -1, hi = values.length ? Math.max(...values) : 1;
  const pad = Math.max(.06, (hi - lo) * .12), min = lo - pad, max = hi + pad;
  const end = Math.max(.25, state?.time || 0), start = end - .25;
  const x = t => left + (t - start) / .25 * (right - left);
  const y = v => bottom - (v - min) / (max - min) * (bottom - top);
  ctx.font = '9px -apple-system, sans-serif';
  for (let i = 0; i <= 2; i++) {
    const value = min + (max - min) * i / 2;
    ctx.strokeStyle = '#2b3843'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(left, y(value)); ctx.lineTo(right, y(value)); ctx.stroke();
    ctx.fillStyle = '#a4b4c0'; ctx.textAlign = 'right'; ctx.fillText(value.toFixed(2), left - 6, y(value) + 3);
  }
  ctx.textAlign = 'left'; ctx.fillText(`${start.toFixed(2)} s`, left, h - 5);
  ctx.textAlign = 'right'; ctx.fillText(`${end.toFixed(2)} s`, right, h - 5);
  ctx.textAlign = 'center'; ctx.fillText('simulated time', (left + right) / 2, h - 5);
  for (const [field, color, dash] of [['target','#a4b3ff',[4,3]], ['actual','#79dacc',[]]]) {
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.setLineDash(dash); ctx.beginPath();
    samples.forEach((p,i) => { if (i) ctx.lineTo(x(p.time), y(p[field])); else ctx.moveTo(x(p.time), y(p[field])); });
    ctx.stroke();
  }
}
new ResizeObserver(drawChart).observe(chart);
setInterval(() => {
  if (document.visibilityState === 'visible' && performance.now() - lastMessage > 30000) {
    $('connection').className = 'error';
    $('connection').textContent = 'Simulation has not responded. Check that WebGL is enabled and reload.';
  }
}, 5000);
