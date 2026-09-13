import { BrainView } from './brain-view.js';

const $ = id => document.getElementById(id);
const frame = $('fly');
const brainView = new BrainView();
let ready = false, bodyReady = false, running = false, pending = null, loopTask = null, nextId = 0, history = [];
const acknowledgements = new Map();
const short = ['LF','LM','LH','RF','RM','RH'];
$('oscillators').innerHTML = [0,3,1,4,2,5].map(i => `<div class="oscillator" data-tripod="${[0,2,4].includes(i)?'A':'B'}"><svg viewBox="0 0 44 44" role="img" aria-label="${short[i]} leg cycle"><circle class="ring" cx="22" cy="22" r="18"/><line class="arm" id="arm-${i}" x1="22" y1="22" x2="22" y2="5"/></svg><div><span class="leg-label">${short[i]}</span><span class="leg-state" id="leg-${i}">—</span></div></div>`).join('');
function command(action, values={}) { frame.contentWindow.postMessage({type:'nmf-command',action,...values},location.origin); }
function status(text,error=false) { $('status').textContent=text; $('status').className=error?'error':''; }
function playbackState(text, live=false) { $('brain-live-state').textContent=text; $('brain-live-state').dataset.running=String(live); }
function enable() { $('run').disabled = $('reset').disabled = !(ready && bodyReady); }
async function request(action, data={}) {
  const response = await fetch(`/api/brain/${action}`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok || result.error) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}
async function checkReady() {
  try {
    const response=await fetch('/api/brain/status');
    if (!response.ok) throw new Error('Restart the local server to enable the brain backend.');
    const s=await response.json();
    if (s.status==='error') throw new Error(s.message);
    if (s.status!=='ready') {status(s.message);setTimeout(checkReady,1500);return;}
    status('Mapping live spikes onto brain coordinates…');
    await brainView.load();
    await request('reset');
    command('reset');
    ready=true;
    $('neurons').textContent=s.metadata.neurons.toLocaleString();
    $('edges').textContent=(s.metadata.connections/1e6).toFixed(2)+'M';
    $('synapses').textContent=(s.metadata.synapse_count_sum/1e6).toFixed(2)+'M';
    const engine=s.metadata.backend==='cuda'?`CUDA · ${s.metadata.gpu}`:'Brian2 · CPU';
    $('runtime-info').textContent=engine;
    $('compute-info').textContent=[engine,s.metadata.hostname,s.metadata.slurm_job_id&&`Slurm ${s.metadata.slurm_job_id}`].filter(Boolean).join(' · ');
    status('● Full connectome ready · paused');enable();
  } catch(error) {status(error.message,true);}
}
addEventListener('message',e=>{
  if(e.origin!==location.origin || e.source!==frame.contentWindow)return;
  if(e.data?.type==='nmf-advanced') {
    const waiter=acknowledgements.get(e.data.id);
    if(waiter){clearTimeout(waiter.timer);acknowledgements.delete(e.data.id);waiter.resolve(e.data);}
  }
  if(e.data?.type!=='nmf-state')return;
  const s=e.data;
  if(!bodyReady){bodyReady=true;command('clock',{external:true});command('reset');enable();}
  $('body-time').textContent=`Body ${s.time.toFixed(3)} s`;
  $('position').textContent=`x ${s.position[0].toFixed(1)} · y ${s.position[1].toFixed(1)} mm`;
  for(let i=0;i<6;i++){
    const a=s.phases[i]-Math.PI/2;
    $('arm-'+i).setAttribute('x2',22+17*Math.cos(a));$('arm-'+i).setAttribute('y2',22+17*Math.sin(a));
    $('leg-'+i).textContent=`${Math.round(s.phases[i]*180/Math.PI)}° · ${s.adhesion[i]?'grip':'release'}`;
  }
  $('joint-target').textContent=s.joint.target.toFixed(2);$('joint-actual').textContent=s.joint.actual.toFixed(2);$('contacts').textContent=s.contacts;
});
function advance(gains) {
  return new Promise((resolve,reject)=>{
    const id=++nextId;
    const timer=setTimeout(()=>{acknowledgements.delete(id);reject(new Error('Body simulation did not acknowledge the neural step.'));},10000);
    acknowledgements.set(id,{resolve,reject,timer});command('advance',{id,left:gains[0],right:gains[1]});
  });
}
function render(s){
  $('brain-time').textContent=`Brain ${s.time.toFixed(3)} s`;
  $('spikes').textContent=s.spikes.toLocaleString();$('active').textContent=s.active_neurons.toLocaleString();
  for(const [key,value] of Object.entries(s.filtered_rates_hz))$(key).textContent=value.toFixed(1);
  $('gain-left').textContent=s.gains[0].toFixed(2);$('gain-right').textContent=s.gains[1].toFixed(2);
  $('speed').textContent=`${s.playback_speed.toFixed(2)}× real time`;
  $('compute-speed').textContent=`Brain computation ${(s.step_seconds/s.wall_seconds).toFixed(2)}× real time · Playback includes network and body`;
  $('top-neurons').replaceChildren();
  if (!s.top_neurons.length) $('top-neurons').textContent='No spikes in this window.';
  for (const n of s.top_neurons.slice(0,4)) {
    const button=document.createElement('button');
    button.textContent=`${n.type||n.id} (${n.spikes})`;
    button.title=`Inspect neuron ${n.id}`;
    button.onclick=()=>brainView.select(n.index);
    $('top-neurons').append(button);
  }
  brainView.update(s);
  history.push({time:s.time,spikes:s.spikes});history=history.slice(-60);draw();
}
async function tick(){
  const started=performance.now();
  const s=await request('step',{stimulus:$('stimulus').value,rate_hz:Number($('rate').value),odor:Number($('odor').value),silence:$('silence').checked});
  const body=await advance(s.gains);
  if(Math.abs(body.time-s.time)>.0002)throw new Error('Brain/body clocks differ. Reset both to synchronize them.');
  s.playback_speed=s.step_seconds/((performance.now()-started)/1000);
  render(s);
}
async function loop(){
  while(running){
    try {pending=tick();await pending;pending=null;}
    catch(error){pending=null;running=false;status(error.message,true);playbackState('Stopped · last window');$('run').textContent='Run brain & body';return;}
  }
  playbackState('Paused · last window');
  status('● Full connectome ready · paused');
}
$('run').onclick=()=>{
  running=!running;$('run').textContent=running?'Pause':'Run brain & body';
  if(running){status('● Live connectome → body');playbackState('Live · 20 ms bins',true);if(!loopTask)loopTask=loop().finally(()=>{loopTask=null;});}
  else playbackState('Finishing current step…');
};
$('reset').onclick=async()=>{
  running=false;$('run').disabled=$('reset').disabled=true;$('run').textContent='Run brain & body';
  try{if(pending)await pending;await request('reset');command('reset');history=[];draw();brainView.reset();playbackState('Paused');$('brain-time').textContent='Brain 0.000 s';$('speed').textContent='paused';$('active').textContent=$('spikes').textContent='0';for(const type of ['DNp09','DNa02','MDN','ORN_DM1'])for(const side of ['left','right'])$(type+'_'+side).textContent='0';$('gain-left').textContent=$('gain-right').textContent='0.00';$('top-neurons').textContent='Network and body reset to rest.';status('● Full connectome ready · paused');}
  catch(error){status(error.message,true);}finally{enable();}
};
$('stimulus').onchange=()=>{
  const odor=$('stimulus').value==='odor';$('rate-label').hidden=odor;$('odor-label').hidden=!odor;
  $('input-description').textContent=odor?'Odor experiment: normalized exposure drives 68 ORN_DM1 neurons. Activity propagates through the actual brain wiring; the decoder may produce no locomotion. Reset both for an independent trial.':$('stimulus').value==='none'?'External stimulation is off. Residual activity may decay. From rest, this model has no spontaneous baseline activity.':'Direct neural stimulation: random input events activate the named neurons. These are interventions, not autonomous behavioral decisions. Reset both for an independent trial.';
};
$('rate').oninput=()=>{$('rate-value').textContent=$('rate').value+' Hz';};
$('odor').oninput=()=>{$('odor-value').textContent=Number($('odor').value).toFixed(2)+' · '+(150*Number($('odor').value)).toFixed(0)+' Hz';};
function draw(){
  const canvas=$('activity-chart'),rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio,2),w=rect.width,h=rect.height;
  canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);const left=40,right=w-6,top=8,bottom=h-22,max=Math.max(1,...history.map(x=>x.spikes)),start=Math.max(0,(history.at(-1)?.time||0)-1.2);
  c.font='9px -apple-system, sans-serif';c.fillStyle='#a4b4c0';c.strokeStyle='#2c3945';
  for(const v of [0,max]){const y=bottom-v/max*(bottom-top);c.beginPath();c.moveTo(left,y);c.lineTo(right,y);c.stroke();c.textAlign='right';c.fillText(String(v),left-6,y+3);}
  c.textAlign='left';c.fillText(`${start.toFixed(2)} s`,left,h-4);c.textAlign='right';c.fillText(`${(start+1.2).toFixed(2)} s`,right,h-4);c.textAlign='center';c.fillText('simulated time',w/2,h-4);
  c.strokeStyle='#79dacc';c.lineWidth=1.7;c.beginPath();history.forEach((p,i)=>{const x=left+(p.time-start)/1.2*(right-left),y=bottom-p.spikes/max*(bottom-top);if(i)c.lineTo(x,y);else c.moveTo(x,y);});c.stroke();
}
new ResizeObserver(draw).observe($('activity-chart'));
checkReady();
