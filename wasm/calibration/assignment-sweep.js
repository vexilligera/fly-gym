const $ = id => document.getElementById(id);
const ns = 'http://www.w3.org/2000/svg';
const labels = {flex_10:'Flex 10°',flex_30:'Flex 30°',flex_50:'Flex 50°',extend_10:'Extend 10°',extend_30:'Extend 30°',extend_50:'Extend 50°',flex_30_slow:'Flex 30° · slow',extend_30_slow:'Extend 30° · slow'};
const settingLabels = {nominal:'Max 150 Hz · original midpoints',gain_75:'Max 75 Hz · lower gain',gain_300:'Max 300 Hz · higher gain',midpoint_minus10:'Midpoints −10° · max 150 Hz',midpoint_plus10:'Midpoints +10° · max 150 Hz'};
function node(tag, attrs, text='') { const e=document.createElementNS(ns,tag); for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);e.textContent=text;return e; }
function chart(id,time,series,unit,release) {
  const svg=$(id);svg.replaceChildren();
  const all=series.flatMap(s=>s.values.flatMap((v,i)=>s.sd?[v-s.sd[i],v+s.sd[i]]:[v]));
  const pad=Math.max((Math.max(...all)-Math.min(...all))*.08,.05),low=Math.min(...all)-pad,high=Math.max(...all)+pad;
  const x=t=>50+(t-time[0])/(time.at(-1)-time[0])*530,y=v=>210-(v-low)/(high-low)*150;
  for(let i=0;i<=4;i++){
    const value=low+(high-low)*i/4,t=time[0]+(time.at(-1)-time[0])*i/4;
    svg.append(node('line',{x1:50,x2:580,y1:y(value),y2:y(value),stroke:'#e8ede7'}));
    svg.append(node('text',{x:43,y:y(value)+4,'text-anchor':'end',fill:'#647168'},value.toFixed(1)));
    svg.append(node('text',{x:x(t),y:231,'text-anchor':'middle',fill:'#647168'},`${t.toFixed(1)}s`));
  }
  svg.append(node('text',{x:50,y:14,fill:'#647168'},unit));
  svg.append(node('line',{x1:x(release),x2:x(release),y1:60,y2:210,stroke:'#bda46a','stroke-dasharray':'3 3'}));
  series.forEach((s,j)=>{
    if(s.sd){
      const upper=s.values.map((v,i)=>`${x(time[i])},${y(v+s.sd[i])}`);
      const lower=s.values.map((v,i)=>`${x(time[i])},${y(v-s.sd[i])}`).reverse();
      svg.append(node('polygon',{points:[...upper,...lower].join(' '),fill:s.color,opacity:.12}));
    }
    svg.append(node('path',{d:s.values.map((v,i)=>`${i?'L':'M'}${x(time[i]).toFixed(2)},${y(v).toFixed(2)}`).join(' '),fill:'none',stroke:s.color,'stroke-width':1.8,'stroke-dasharray':s.dashed?'4 3':'none'}));
    svg.append(node('text',{x:50+(j%2)*270,y:31+Math.floor(j/2)*16,fill:s.color},s.name));
  });
}
function row(table, values) {const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}$(table).append(tr);}
async function json(url){const r=await fetch(url);if(!r.ok)throw new Error(`${url}: ${r.status}`);return r.json();}
function option(select,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;select.append(o);}
try {
  const [report,traces]=await Promise.all(['assignment-report.json','assignment-traces.json'].map(json));
  const ids=report.config.sensory_body_id_order, assignments=report.config.assignments;
  $('assignment-total').textContent=assignments.length;
  $('assignment-trials').textContent=report.runtime.trial_count.toLocaleString();
  $('assignment-close').textContent=report.summary.close_pairs.length;
  $('assignment-summary').textContent=`At the assumed 0.5° RMS angle and 5 Hz RMS pooled-motor resolutions, ${report.summary.joint_separated_pairs} of 120 candidate pairs exceed those thresholds on at least one primary test; ${report.summary.close_pairs.length} pairs remain close. These are model predictions. No biological mapping has been selected or rejected.`;
  const angleMax=Math.max(...report.pairwise.map(p=>p.max_angle_rms_deg)),motorMax=Math.max(...report.pairwise.map(p=>p.max_motor_rms_hz));
  const primary=report.motor_recruitment.primary_perturbations,all=report.motor_recruitment.all_connected;
  $('assignment-observation').textContent=`The largest primary pair differences are ${angleMax.toFixed(3)}° RMS and ${motorMax.toFixed(2)} Hz RMS. Primary trials produced ${primary.flexor.total_spikes} flexor and ${primary.extensor.total_spikes} extensor spikes. Across all gain/threshold settings, extensors fired in ${all.extensor.trials_with_spikes}/${all.extensor.trials} connected trials.`;
  $('assignment-order').textContent=`Letter order: ${ids.join(' → ')}. F = provisional flexion response; E = provisional extension response.`;
  for(const name of Object.keys(report.config.settings))option($('assignment-setting'),name,settingLabels[name]);
  for(const a of assignments){option($('assignment-a'),a,a);option($('assignment-b'),a,a);}
  $('assignment-a').value='FFFF';$('assignment-b').value='EEEE';
  for(const [setting,result] of Object.entries(report.sensitivity_at_30_deg))row('assignment-sensitivity',[settingLabels[setting],`${result.angle_separated_pairs} / 120`,`${result.joint_separated_pairs} / 120`,result.close_pairs.length]);
  const names=report.protocol_comparison.map(p=>p.protocol);
  const head=document.createElement('tr');for(const text of ['Assignment',...names.map(n=>labels[n])]){const th=document.createElement('th');th.textContent=text;head.append(th);}$('assignment-matrix-head').append(head);
  for(const a of assignments)row('assignment-matrix',[a,...names.map(n=>report.groups.find(g=>g.assignment===a&&g.protocol===n&&g.setting==='nominal').max_mean_evoked_angle_deg.toFixed(2)+'°')]);
  for(const p of report.protocol_comparison)row('assignment-protocols',[labels[p.protocol],`${p.angle_separated_pairs} / 120`,`${p.joint_separated_pairs} / 120`,`${p.every_seed_joint_separated_pairs} / 120`]);
  const seen=new Set();
  const readouts=[];
  for(const r of report.suggested_downstream_readouts){if(seen.has(r.body_id)||seen.size>=5)continue;seen.add(r.body_id);readouts.push(r);row('assignment-readouts',[r.body_id,r.type??'Unspecified',labels[r.protocol],r.candidate_mean_rate_range_hz.toFixed(1)+' Hz',r.median_within_mapping_seed_sd_hz.toFixed(1)+' Hz']);option($('assignment-neuron'),String(readouts.length-1),`${r.body_id} · ${r.type??'Unspecified'} · ${labels[r.protocol]}`);}
  function neuronChart(){
    const r=readouts[Number($('assignment-neuron').value)],svg=$('assignment-neuron-chart');svg.replaceChildren();
    const values=assignments.map(a=>r.mean_evoked_hz_by_assignment[a]);
    const low=Math.min(0,...values),high=Math.max(1,...values),span=high-low;
    const y=v=>205-(v-low)/span*165;
    for(let i=0;i<=4;i++){const value=low+span*i/4;svg.append(node('line',{x1:50,x2:580,y1:y(value),y2:y(value),stroke:'#e8ede7'}));svg.append(node('text',{x:42,y:y(value)+4,'text-anchor':'end',fill:'#647168'},value.toFixed(1)));}
    svg.append(node('text',{x:50,y:18,fill:'#647168'},`${r.body_id} · post-onset response minus sham (simulated Hz)`));
    values.forEach((v,i)=>{const x=50+i*530/16;const bar=node('rect',{x:x+3,y:Math.min(y(v),y(0)),width:530/16-6,height:Math.max(.6,Math.abs(y(v)-y(0))),fill:v>=0?'#b44226':'#456f62'});bar.append(node('title',{},`${assignments[i]}: ${v.toFixed(2)} Hz`));svg.append(bar);svg.append(node('text',{x:x+530/32,y:224,'text-anchor':'middle',fill:'#647168','font-size':10},assignments[i]));});
  }
  $('assignment-neuron').disabled=false;$('assignment-neuron').addEventListener('change',neuronChart);neuronChart();
  function draw(){
    const setting=$('assignment-setting').value,protocol=$('assignment-protocol').value;
    const a=$('assignment-a').value,b=$('assignment-b').value;
    const ta=traces.find(t=>t.assignment===a&&t.setting===setting&&t.protocol===protocol);
    const tb=traces.find(t=>t.assignment===b&&t.setting===setting&&t.protocol===protocol);
    const sa=report.groups.find(t=>t.assignment===a&&t.setting===setting&&t.protocol===protocol);
    const sb=report.groups.find(t=>t.assignment===b&&t.setting===setting&&t.protocol===protocol);
    $('assignment-detail').textContent=`${a}: peak mean evoked angle effect ${sa.max_mean_evoked_angle_deg.toFixed(3)}°, ${sa.mean_evoked_motor_spikes.toFixed(1)} additional motor spikes versus sham. ${b}: ${sb.max_mean_evoked_angle_deg.toFixed(3)}°, ${sb.mean_evoked_motor_spikes.toFixed(1)} additional motor spikes. Spikes include the whole trial; the angle metric uses the post-release interval.`;
    chart('assignment-angle-chart',ta.time_s,[{name:`${a} · mean ± seed SD`,values:ta.angle_mean_deg,sd:ta.angle_sd_deg,color:'#b44226'},{name:`${b} · mean ± seed SD`,values:tb.angle_mean_deg,sd:tb.angle_sd_deg,color:'#456f62'}],'evoked circuit angle difference (degrees)',ta.release_s);
    chart('assignment-motor-chart',ta.time_s,[{name:`${a} flexor`,values:ta.flexor_mean_hz,color:'#b44226'},{name:`${b} flexor`,values:tb.flexor_mean_hz,color:'#456f62'},{name:`${a} extensor`,values:ta.extensor_mean_hz,color:'#b44226',dashed:true},{name:`${b} extensor`,values:tb.extensor_mean_hz,color:'#456f62',dashed:true}],'simulated motor activity minus sham (Hz)',ta.release_s);
    $('assignment-cell-labels').replaceChildren();
    ids.forEach((id,i)=>row('assignment-cell-labels',[id,a[i]==='F'?'Flexion':'Extension',b[i]==='F'?'Flexion':'Extension']));
  }
  function settingChanged(){
    const old=$('assignment-protocol').value; $('assignment-protocol').replaceChildren();
    const names=$('assignment-setting').value==='nominal'?report.protocol_comparison.map(p=>p.protocol):['flex_30','extend_30'];
    for(const n of names)option($('assignment-protocol'),n,labels[n]);
    $('assignment-protocol').value=names.includes(old)?old:'flex_30';draw();
  }
  for(const id of ['assignment-a','assignment-b','assignment-setting','assignment-protocol'])$(id).disabled=false;
  $('assignment-setting').addEventListener('change',settingChanged);
  for(const id of ['assignment-a','assignment-b','assignment-protocol'])$(id).addEventListener('change',draw);
  settingChanged();
  $('assignment-checks').textContent=report.preflight.passed&&report.sweep_validation.passed?'Input-adapter regression, silencing controls and complete-sweep checks passed on B300.':'Validation requires attention.';
}catch(error){$('assignment-summary').textContent=`Unable to load assignment sweep: ${error.message}`;}
