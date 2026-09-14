const $ = id => document.getElementById(id);
const ns = 'http://www.w3.org/2000/svg';
function svgNode(tag, attributes, text = '') {
  const node = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  node.textContent = text;
  return node;
}
function chart(id, time, series, unit) {
  const svg = $(id);
  svg.replaceChildren();
  const values = series.flatMap(s => s.values).filter(Number.isFinite);
  const pad = Math.max((Math.max(...values) - Math.min(...values)) * .08, .05);
  const low = Math.min(...values) - pad, high = Math.max(...values) + pad;
  const start = time[0], end = time.at(-1);
  const x = t => 48 + (t - start) / (end - start) * 532;
  const y = v => 205 - (v - low) / (high - low) * 160;
  for (let i = 0; i <= 4; i++) {
    const value = low + (high - low) * i / 4, t = start + (end - start) * i / 4;
    svg.append(svgNode('line', { x1: 48, x2: 580, y1: y(value), y2: y(value), stroke: '#e9eee7' }));
    svg.append(svgNode('text', { x: 41, y: y(value) + 4, 'text-anchor': 'end', fill: '#657469' }, value.toFixed(unit === 'degrees' ? 0 : 1)));
    svg.append(svgNode('text', { x: x(t), y: 227, 'text-anchor': 'middle', fill: '#657469' }, `${t.toFixed(0)}s`));
  }
  svg.append(svgNode('text', { x: 48, y: 17, fill: '#657469' }, unit));
  series.forEach((s, i) => {
    const d = s.values.map((v, j) => `${j ? 'L' : 'M'}${x(time[j]).toFixed(2)},${y(v).toFixed(2)}`).join(' ');
    svg.append(svgNode('path', { d, fill: 'none', stroke: s.color, 'stroke-width': 1.7 }));
    svg.append(svgNode('text', { x: 48 + i * 175, y: 34, fill: s.color }, s.name));
  });
}
async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}
const splitNames = { animal_test: 'Unseen fly', protocol_test: 'Reverse-order transfer', train: 'Training', validation: 'Model selection' };
function cells(row, values) {
  for (const value of values) {
    const cell = document.createElement('td');
    cell.textContent = value;
    row.append(cell);
  }
}
try {
  const [report, view, checks] = await Promise.all(['claw-cells-report.json', 'claw-cells-traces.json', 'claw-cells-validation.json'].map(json));
  const groups = report.evaluation.groups;
  for (const [cohort, id] of [['JR209', 'cell-flex-error'], ['JR688', 'cell-ext-error']]) {
    const ratio = groups[cohort].animal_test.relative_mse;
    $(id).textContent = `${(100 * Math.abs(1 - ratio)).toFixed(1)}% ${ratio < 1 ? 'lower' : 'higher'}`;
  }
  $('cell-map-count').textContent = `${report.mapping.verified_recording_matches} / 4`;
  const failed = report.evaluation.fly_scores.filter(r => r.split === 'animal_test' && r.relative_mse >= 1);
  $('cell-verdict').textContent = `The fits beat a training-constant baseline in ${4 - failed.length} of four unseen flies, with each fly weighted equally. ${failed.map(r => `${r.animal_id} has ${r.relative_mse.toFixed(2)}× baseline error`).join('; ')}. This supports different sensory response families, but the fit does not establish the tuning of individual connectome neurons.`;
  const contrasts = ['JR209', 'JR688'].map(cohort => {
    const rows = report.evaluation.region_scores.filter(r => r.cohort === cohort && r.split === 'animal_test');
    const count = rows.filter(r => cohort === 'JR209' ? r.tuning.low_minus_high_drr > 0 : r.tuning.low_minus_high_drr < 0).length;
    return `${count}/${rows.length} ${cohort} region traces have higher measured calcium at ${cohort === 'JR209' ? 'low' : 'high'} angles`;
  });
  $('cell-tuning-observation').textContent = `Directly in the unseen-fly recordings: ${contrasts.join('; ')}. This compares angles ≤60° with ≥120° and does not depend on the fitted model. Counts include both movement orders.`;
  $('cell-check-status').textContent = checks.passed ? 'Source integrity, held-out isolation and causal prediction checks passed.' : 'Validation checks need attention.';
  for (const r of report.evaluation.fly_scores.filter(r => r.split === 'animal_test')) {
    const row = document.createElement('tr');
    cells(row, [r.animal_id, r.region_traces, `${r.relative_mse.toFixed(3)}×`, r.median_region_r.toFixed(2), r.relative_mse < 1 ? 'Better than constant' : 'Worse than constant']);
    $('cell-fly-results').append(row);
  }
  for (const r of report.mapping.neurons) {
    const row = document.createElement('tr');
    cells(row, [r.bodyId, r.type, r.mancBodyid ?? 'No cross-reference', 'Unresolved']);
    $('cell-mapping').append(row);
  }
  function draw() {
    const r = view.traces.find(r => r.id === $('cell-record').value);
    const score = report.evaluation.region_scores.find(s => s.id === r.id);
    const predicted = score.models.separate;
    const fit = report.frozen_fit.cohorts[r.cohort].selected;
    $('cell-record-score').textContent = `${splitNames[r.split]} · source column ${r.region} · ${predicted.relative_mse.toFixed(2)}× constant-baseline error · correlation ${predicted.pearson_r?.toFixed(2) ?? 'undefined'}. Measured calcium is higher at ${score.tuning.low_minus_high_drr > 0 ? 'low' : 'high'} angles. Gain and offset use training flies; relaxation uses the model-selection fly.`;
    $('cell-curve-details').textContent = `${r.cohort}: population half-response angle ${fit.parameters[2].toFixed(1)}°, width ${fit.parameters[3].toFixed(1)}°, effective relaxation ${fit.tau_s.toFixed(1)} s. ${fit.tau_s === 3 ? 'The relaxation reaches the largest candidate tested. ' : ''}These are observation-model parameters, not identified neuronal thresholds or measured GCaMP kinetics.`;
    chart('cell-calcium-chart', r.time_s, [{ name: 'Measured', values: r.calcium, color: '#596873' }, { name: 'Frozen fit', values: r.prediction, color: '#b44226' }, { name: 'Extension only', values: r.legacy_prediction, color: '#84a69a' }], 'ΔR/R · source fluorescence units');
    chart('cell-angle-chart', r.time_s, [{ name: 'Recorded angle', values: r.angle_deg, color: '#596873' }], 'degrees');
  }
  function chooseCohort() {
    $('cell-record').replaceChildren();
    const cohort = $('cell-cohort').value;
    for (const split of ['animal_test', 'protocol_test', 'validation', 'train']) {
      const group = document.createElement('optgroup');
      group.label = splitNames[split];
      for (const r of view.traces.filter(r => r.cohort === cohort && r.split === split)) {
        const option = document.createElement('option');
        option.value = r.id;
        option.textContent = `Fly ${r.fly} · region ${r.region} · starts ${r.order === 2 ? 'extended' : 'flexed'}`;
        group.append(option);
      }
      $('cell-record').append(group);
    }
    draw();
  }
  $('cell-cohort').disabled = false;
  $('cell-record').disabled = false;
  $('cell-cohort').addEventListener('change', chooseCohort);
  $('cell-record').addEventListener('change', draw);
  chooseCohort();
} catch (error) {
  $('cell-verdict').textContent = `Unable to load cell validation: ${error.message}`;
}
