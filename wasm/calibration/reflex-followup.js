const $=id=>document.getElementById(id);
const NS='http://www.w3.org/2000/svg';
function element(tag,attrs={},text=''){const n=document.createElementNS(NS,tag);for(const [key,value] of Object.entries(attrs))n.setAttribute(key,value);n.textContent=text;return n;}
function chart(id,time,series,unit,release){
  const svg=$(id);svg.replaceChildren();
  const values=series.flatMap(s=>s.values.filter(v=>Number.isFinite(v)));
  let low=Math.min(...values),high=Math.max(...values);const pad=Math.max((high-low)*.08,.05);low-=pad;high+=pad;
  const end=Math.max(...time), x=t=>48+t/end*532, y=v=>205-(v-low)/(high-low)*160;
  for(let i=0;i<=4;i++){
    const value=low+(high-low)*i/4, t=end*i/4;
    svg.append(element('line',{x1:48,x2:580,y1:y(value),y2:y(value),stroke:'#e9eee7'}));
    svg.append(element('text',{x:41,y:y(value)+4,'text-anchor':'end',fill:'#657469'},Math.abs(high-low)>10?value.toFixed(0):value.toFixed(2)));
    svg.append(element('text',{x:x(t),y:227,'text-anchor':'middle',fill:'#657469'},`${t.toFixed(end>10?0:1)}s`));
  }
  svg.append(element('text',{x:48,y:17,fill:'#657469'},unit));
  if(release!==undefined)svg.append(element('line',{x1:x(release),x2:x(release),y1:45,y2:205,stroke:'#bda46a','stroke-dasharray':'3 3'}));
  series.forEach((s,i)=>{
    let connected=false;
    const d=s.values.map((v,j)=>{if(!Number.isFinite(v)){connected=false;return '';}const cmd=connected?'L':'M';connected=true;return `${cmd}${x(time[j]).toFixed(2)},${y(v).toFixed(2)}`;}).join(' ');
    svg.append(element('path',{d,fill:'none',stroke:s.color,'stroke-width':1.7}));
    svg.append(element('text',{x:48+i*175,y:34,fill:s.color},s.name));
  });
}
async function json(url){const r=await fetch(url);if(!r.ok)throw new Error(`${url}: HTTP ${r.status}`);return r.json();}
const names={train:'Training',validation:'Model selection',animal_test:'Unseen animal test',protocol_test:'Opposite-order test'};
try{
  const [report,traces,sweep]=await Promise.all(['claw-report.json','claw-traces.json','sweep-trials.json'].map(json));
  const test=report.evaluation.groups.animal_test, protocol=report.evaluation.groups.protocol_test;
  $('claw-summary').textContent=`On the two unseen flies, the frozen fit reduces error by ${(100*(1-test.relative_mse)).toFixed(1)}% versus the training-constant baseline (median correlation ${test.median_pearson_r.toFixed(2)}). All ${test.trials} unseen-animal trials and all ${protocol.trials} opposite-order trials beat that baseline. This supports a useful sensory-response benchmark; physiological circuit calibration remains open.`;
  for(const r of traces){const option=document.createElement('option');option.value=r.id;option.textContent=`Fly ${r.animal_id} · ${r.protocol.includes('ext_first')?'extension':'flexion'} first · ${names[r.split]}`;$('claw-record').append(option);}
  function recording(){
    const r=traces.find(r=>r.id===$('claw-record').value);
    const score=report.evaluation.scores.find(s=>s.id===r.id).models[report.frozen_fit.selected_model];
    $('claw-score').textContent=`${names[r.split]} · ${r.id} · ${(100*score.relative_mse).toFixed(1)}% of constant-baseline error · correlation ${score.pearson_r.toFixed(2)}. No per-test gain, offset or time shift was fitted.`;
    chart('claw-chart',r.time_s,[{name:'Measured',values:r.calcium,color:'#596873'},{name:'Frozen fit',values:r.prediction,color:'#b44226'},{name:'Extension only',values:r.legacy_prediction,color:'#84a69a'}],'normalized calcium');
    chart('claw-input-chart',r.time_s,[{name:'Recorded angle',values:r.angle_deg,color:'#596873'}],'degrees');
  }
  $('claw-record').value=traces.find(r=>r.split==='animal_test').id;
  $('claw-record').disabled=false;$('claw-record').addEventListener('change',recording);recording();
  for(const r of report.sweep){
    const option=document.createElement('option');option.value=r.id;option.textContent=r.name;$('sweep-case').append(option);
    const row=document.createElement('tr');
    for(const text of [r.name,`${r.motor_spikes} / ${r.disconnected_motor_spikes}`,`${r.max_angle_difference_deg.toFixed(3)}°`]){const cell=document.createElement('td');cell.textContent=text;row.append(cell);}
    $('sweep-table').append(row);
  }
  function physical(){
    const r=report.sweep.find(r=>r.id===$('sweep-case').value),pair=sweep[r.id];
    const a=pair.connected.trace,p=pair.disconnected.trace,time=a.map(x=>x.time_s);
    $('sweep-summary').textContent=`${r.name}: ${r.motor_spikes} motor spikes with the circuit connected, ${r.disconnected_motor_spikes} disconnected. Maximum added movement: ${r.max_angle_difference_deg.toFixed(3)}°. ${r.protocol.displacement_deg<0?'No neural reflex appears: the current input rule is silent during flexion.':'Positive extension responses remain dependent on provisional sensory and muscle parameters.'}`;
    chart('sweep-chart',time,[{name:'Connected',values:a.map(x=>x.angle_deg),color:'#b44226'},{name:'Disconnected',values:p.map(x=>x.angle_deg),color:'#596873'}],'degrees · dashed line = release',r.release_s);
    chart('sweep-rate-chart',time,[{name:'Sensory',values:a.map(x=>x.sensory_hz),color:'#596873'},{name:'Flexor',values:a.map(x=>x.flexor_hz),color:'#b44226'},{name:'Extensor',values:a.map(x=>x.extensor_hz),color:'#459578'}],'simulated Hz',r.release_s);
  }
  $('sweep-case').value='4';$('sweep-case').disabled=false;$('sweep-case').addEventListener('change',physical);physical();
}catch(error){$('claw-summary').textContent=`Unable to load follow-up: ${error.message}`;$('sweep-summary').textContent='Follow-up unavailable.';}
