import { BrainView } from '../connectome/brain-view.js';

const $ = id => document.getElementById(id);
const brainView = new BrainView();
let state = {status:'idle'}, ready = false, busy = false, lastFrame = -1;
let trialId = null, recordedId = null, trials = [];
const names = {vision:'Vision connected',blind:'Eyes disconnected',shuffled:'Mapping shuffled',readout_off:'Steering disconnected'};
const descriptions = {
  vision:'Rendered dark contrast drives R1–6 input. Actual downstream L2 spikes determine steering; a constant walking drive keeps the six-leg controller moving.',
  blind:'Control: eye cameras still render, but all visual input events are zero. The same constant walking drive remains; the brain starts from rest.',
  shuffled:'Control: the 1,442 retinal channels are permuted before neural input. The decoder keeps its original registration, breaking spatial correspondence.',
  readout_off:'Control: vision still drives the full brain, but L2 activity no longer changes steering. The same constant walking drive remains.',
};
function message(text,error=false) { $('status').textContent=text; $('status').classList.toggle('error',error); }
function controls() {
  $('start').disabled=$('reset').disabled=!ready||busy;
  $('pause').disabled=!ready||busy||!['running','paused'].includes(state.status)||!state.config;
  $('pause').textContent=state.status==='running'?'Pause':'Resume';
}
function config() { return {condition:$('condition').value,target_deg:Number($('target').value),heading_deg:Number($('heading').value),seed:Number($('seed').value),duration:3}; }
async function api(path,args) {
  const response=await fetch('/api/'+path,args===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(args)});
  const data=await response.json();
  if(!response.ok)throw new Error(data.error||'Compute service request failed');
  return data;
}
async function command(action,args={}) {
  busy=true; controls(); message(action==='pause'?'Finishing current simulation bin…':'Preparing the visual trial…');
  try {
    const next=await api('vision/'+action,args);
    render(next);
  } catch(error){message(error.message,true);} finally{busy=false;controls();}
}
$('start').onclick=()=>command('start',config());
$('reset').onclick=()=>command('reset',config());
$('pause').onclick=()=>command(state.status==='running'?'pause':'start');
$('condition').onchange=()=>{$('condition-description').textContent=descriptions[$('condition').value];};
for(const key of ['heading','target'])$(key).oninput=()=>{$(key+'-value').textContent=$(key).value+'°';};

function render(s) {
  state=s;controls();
  if(s.status==='error'){message(s.message||'Visual trial stopped with an error',true);return;}
  if(!s.config){message(s.message||'Ready to start a visual trial');return;}
  if(s.trial_id!==trialId){trialId=s.trial_id;lastFrame=-1;}
  const statusText={running:'Live visual feedback',paused:'Paused',reached:'Stripe reached',finished:'Trial finished'}[s.status]||s.status;
  message(`● ${statusText} · ${names[s.config.condition]}`);
  $('trial-state').textContent=statusText.toUpperCase();
  $('brain-live-state').textContent=s.status==='running'?'Live · latest bin':'Held · last bin';
  $('brain-live-state').dataset.running=s.status==='running';
  if(s.frame===lastFrame)return;
  lastFrame=s.frame;
  if(s.brain)brainView.update(s.brain);else brainView.reset();
  $('body-loading').hidden=Boolean(s.images);
  if(s.images) {
    $('body').src='data:image/jpeg;base64,'+s.images.body;
    for(const [index,side] of ['left','right'].entries())$('eye-'+side).src='data:image/jpeg;base64,'+s.images.eyes[index];
  }
  $('body-time').textContent=s.time.toFixed(3)+' s';
  $('distance').textContent=s.score.distance_mm.toFixed(1)+' mm to stripe';
  $('speed').textContent=s.playback_speed.toFixed(2)+'× real time';
  $('sensory-time').textContent=`Retinal input at ${s.sensory_time.toFixed(3)} s → brain/body at ${s.time.toFixed(3)} s.`;
  const mean=values=>values.reduce((a,b)=>a+b,0)/values.length;
  $('contrast-left').textContent=(100*mean(s.contrast[0])).toFixed(1)+'%';
  $('contrast-right').textContent=(100*mean(s.contrast[1])).toFixed(1)+'%';
  $('l2-spikes').textContent=s.decoder.L2_spikes.toLocaleString();
  $('active').textContent=(s.brain?.active_neurons||0).toLocaleString();
  $('bearing').textContent=s.decoder.bearing_deg.toFixed(1)+'°';
  $('coherence').textContent=s.decoder.coherence.toFixed(2);
  $('gain-left').textContent=s.decoder.gains[0].toFixed(2);$('gain-right').textContent=s.decoder.gains[1].toFixed(2);
  $('ground-truth').textContent=`Evaluation only: heading error ${s.score.heading_error_deg.toFixed(1)}°. These world coordinates are not steering inputs.`;
  $('result').textContent=s.done?(s.status==='reached'?`Entered the 3 mm arrival zone after ${s.time.toFixed(2)} simulated seconds.`:`Time limit reached, ${s.score.distance_mm.toFixed(1)} mm from the stripe.`):'Trial in progress. Arrival is scored within 3 mm of the stripe.';
  path(s);
  if(s.done&&recordedId!==trialId){recordedId=trialId;trials.unshift(s);trials=trials.slice(0,10);renderTrials();}
}
function path(s) {
  const points=[...s.path,s.target,[0,0]];
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  const minX=Math.min(-3,...xs)-3,maxX=Math.max(22,...xs)+3,minY=Math.min(-12,...ys)-3,maxY=Math.max(12,...ys)+3;
  const scale=Math.min(280/(maxX-minX),195/(maxY-minY));
  const xy=p=>[20+(p[0]-minX)*scale,210-(p[1]-minY)*scale];
  $('path').setAttribute('d',s.path.map((p,i)=>(i?'L':'M')+xy(p).join(' ')).join(' '));
  const [tx,ty]=xy(s.target);for(const id of ['target-zone','target-dot']){$(id).setAttribute('cx',tx);$(id).setAttribute('cy',ty);}
  $('target-zone').setAttribute('r',3*scale);
  const [x,y]=xy(s.position);$('fly-marker').setAttribute('transform',`translate(${x} ${y}) rotate(${-s.heading_deg})`);
}
function renderTrials() {
  $('trials').replaceChildren();
  for(const s of trials){const row=document.createElement('tr');for(const value of [names[s.config.condition],`${s.config.target_deg}° / ${s.config.heading_deg}°`,s.config.seed,s.time.toFixed(2)+' s',s.score.distance_mm.toFixed(1)+' mm',s.status==='reached'?'Reached':'Time limit']){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}$('trials').append(row);}
}
async function initialize() {
  while(!ready){
    try {
      const s=await api('brain/status');
      if(s.status==='error')throw new Error(s.message);
      if(s.status==='ready'){
        const m=s.metadata;$('neurons').textContent=m.neurons.toLocaleString();$('edges').textContent=(m.connections/1e6).toFixed(2)+'M';
        $('compute-info').textContent=`${m.gpu||m.backend} · ${m.hostname}${m.slurm_job_id?' · Slurm '+m.slurm_job_id:''}`;
        await brainView.load();ready=true;controls();break;
      }
      message(s.message||'Brain loading…');
    }catch(error){message(error.message,true);}
    await new Promise(resolve=>setTimeout(resolve,1500));
  }
  try {
    const response=await fetch('validation.json');
    if(response.ok){const v=await response.json();$('validation').textContent=v.summary;}
    else $('validation').textContent='Use repeated matched trials to assess this experimental controller.';
  }catch{$('validation').textContent='Deployment validation is unavailable.';}
  while(true){
    try{if(!busy){const next=await api('vision/status');if(!busy)render(next);}}catch(error){message('Connection lost: '+error.message,true);}
    await new Promise(resolve=>setTimeout(resolve,150));
  }
}
initialize();
