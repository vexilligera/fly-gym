import { BrainView } from '../connectome/brain-view.js';

const $ = id => document.getElementById(id);
const brainView = new BrainView();
const names = {combined:'Vision + smell',vision_only:'Vision only',odor_only:'Smell only',neither:'Both disconnected'};
const descriptions = {
  combined:'Eye images drive R1–6 cells and local antenna samples drive DM1 odor neurons in the same brain step. A hand-designed controller combines L2 and odor-neuron spikes to steer the legs.',
  vision_only:'Control: eyes remain connected; odor input is zero. The fly can respond to walls but has no encoded food-odor cue.',
  odor_only:'Control: antenna samples remain connected; eye input is zero. The fly follows the odor signal without visual wall avoidance.',
  neither:'Control: both sensory input streams are zero. The same constant walking drive remains. Any arrival is incidental.',
};
let state = {status:'idle'}, ready = false, busy = false, lastFrame = -1;
let trialId = null, recordedId = null, trials = [], world = null, loadingWorld = false;

function message(text, error=false) {
  $('status').textContent=text;
  $('status').classList.toggle('error',error);
}
function controls() {
  $('start').disabled=$('reset').disabled=!ready||busy;
  $('pause').disabled=!ready||busy||!['running','paused'].includes(state.status)||!state.config;
  $('pause').textContent=state.status==='running'?'Pause':'Resume';
}
function config() {
  return {condition:$('condition').value,heading_deg:Number($('heading').value),
    seed:Number($('seed').value),duration:Number($('duration').value),food_odor:$('food-odor').checked};
}
async function api(path, args) {
  const response=await fetch('/api/'+path,args===undefined?{}:{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(args)});
  const data=await response.json();
  if(!response.ok)throw new Error(data.error||'Compute service request failed');
  return data;
}
async function command(action,args={}) {
  if(!$('seed').reportValidity())return;
  busy=true;controls();
  message(action==='pause'?'Finishing the current simulation bin…':'Preparing the maze and both senses…');
  try { render(await api('maze/'+action,args)); }
  catch(error) { message(error.message,true); }
  finally { busy=false;controls(); }
}
$('start').onclick=()=>command('start',config());
$('reset').onclick=()=>command('reset',config());
$('pause').onclick=()=>command(state.status==='running'?'pause':'start');
$('condition').onchange=()=>{$('condition-description').textContent=descriptions[$('condition').value];};
$('heading').oninput=()=>{$('heading-value').textContent=$('heading').value+'°';};
$('show-field').onchange=drawField;

async function loadWorld() {
  if(world||loadingWorld)return;
  loadingWorld=true;
  try {
    world=await api('maze/world');
    for(const [x,y,hx,hy] of world.walls) {
      const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
      for(const [key,value] of Object.entries({x:x-hx,y:y-hy,width:2*hx,height:2*hy,fill:'#526574',stroke:'#91a4b3','stroke-width':.12}))rect.setAttribute(key,value);
      $('walls').append(rect);
    }
    drawField();
  } catch(error) { message('Maze map unavailable: '+error.message,true); }
  finally { loadingWorld=false; }
}
function drawField() {
  const canvas=$('odor-map'), ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!world||!$('show-field').checked||!state.config?.food_odor)return;
  const field=world.odor_field, scale=canvas.width/44, cell=field.metadata.spacing_mm;
  for(let y=0;y<field.axis_mm.length;y++)for(let x=0;x<field.axis_mm.length;x++) {
    if(field.blocked[y][x])continue;
    const strength=Math.sqrt(field.values[y][x]);
    ctx.fillStyle=`rgba(240,171,69,${strength*.8})`;
    ctx.fillRect((field.axis_mm[x]-cell/2+22)*scale,(22-field.axis_mm[y]-cell/2)*scale,cell*scale+.5,cell*scale+.5);
  }
}
function render(s) {
  state=s;controls();
  if(s.status==='error'){message(s.message||'Maze simulation stopped with an error',true);return;}
  if(!s.config){message(s.message||'Ready to run the sugar maze');return;}
  if(s.trial_id!==trialId){trialId=s.trial_id;lastFrame=-1;drawField();}
  loadWorld();
  const statusText={running:'Live sensory feedback',paused:'Paused',reached:'Sugar zone reached',finished:'Time limit reached',fallen:'Fly lost balance'}[s.status]||s.status;
  message(`● ${statusText} · ${names[s.config.condition]}${s.config.food_odor?'':' · Food odor off'}`);
  $('trial-state').textContent=s.status.toUpperCase();
  $('brain-live-state').textContent=s.status==='running'?'Live · latest bin':'Held · last bin';
  $('brain-live-state').dataset.running=s.status==='running';
  if(s.frame===lastFrame)return;
  lastFrame=s.frame;
  if(s.brain)brainView.update(s.brain);else brainView.reset();
  $('body-loading').hidden=Boolean(s.images);
  if(s.images) {
    $('body').src='data:image/jpeg;base64,'+s.images.body;
    for(const [i,side] of ['left','right'].entries())$('eye-'+side).src='data:image/jpeg;base64,'+s.images.eyes[i];
  }
  $('body-time').textContent=s.time.toFixed(3)+' s';
  $('distance').textContent=s.score.distance_mm.toFixed(1)+' mm to sugar';
  $('speed').textContent=s.playback_speed.toFixed(2)+'× real time';
  const vision=['combined','vision_only'].includes(s.config.condition);
  const smell=['combined','odor_only'].includes(s.config.condition);
  $('eyes-state').textContent=vision?'721 samples / eye':'Input disconnected';
  $('sensory-time').textContent=`Sensors sampled at ${s.sensory_time.toFixed(3)} s → brain/body at ${s.time.toFixed(3)} s.`;
  $('odor-state').textContent=!s.config.food_odor?'Food odor off: both antenna samples are zero.':smell?'Local food-odor concentration, normalized 0–1.':'Odor is present here; neural odor input is disconnected.';
  const d=s.decoder||{}, o=s.brain?.olfaction;
  for(const [i,side] of ['left','right'].entries()) {
    $('odor-'+side).textContent=s.odor[i].toFixed(3);
    $('odor-meter-'+side).value=s.odor[i];
    $('orn-'+side).textContent=(d['odor_'+side+'_hz']||0).toFixed(0)+' Hz';
    $('pn-'+side).textContent=o?.PN_spikes[i]||0;
    $('gain-'+side).textContent=(d.gains?.[i]||0).toFixed(2);
  }
  $('l2-spikes').textContent=(d.L2_spikes||0).toLocaleString();
  $('wall-front').textContent=(d.wall_front||0).toFixed(2);
  $('active').textContent=(s.brain?.active_neurons||0).toLocaleString();
  $('steering').textContent=d.escape?'Turning away from a strong front-wall response.':`Odor steering ${(d.odor_turn||0).toFixed(2)} · Visual avoidance ${(d.visual_turn||0).toFixed(2)}. Positive turns left.`;
  $('ground-truth').textContent=`Evaluation: ${s.score.distance_mm.toFixed(1)} mm from center. The controller does not receive this distance.`;
  $('result').textContent=s.status==='reached'?`Entered the 2.5 mm food zone after ${s.time.toFixed(2)} simulated seconds. Feeding is not modeled.`:s.status==='fallen'?'The fly lost balance; the trial stopped.':s.done?'Time limit reached without entering the food zone.':'A trial ends at the food zone, the time limit, or loss of balance.';
  $('path').setAttribute('d',s.path.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' '));
  $('fly-marker').setAttribute('transform',`translate(${s.position[0]} ${s.position[1]}) rotate(${s.heading_deg})`);
  if(s.done&&recordedId!==trialId) {
    recordedId=trialId;
    trials.unshift({config:s.config,time:s.time,distance:s.score.distance_mm,status:s.status});
    trials=trials.slice(0,12);renderTrials();
  }
}
function renderTrials() {
  $('trials').replaceChildren();
  for(const s of trials) {
    const row=document.createElement('tr');
    const values=[names[s.config.condition],s.config.food_odor?'On':'Off',`${s.config.heading_deg}° / ${s.config.seed}`,s.time.toFixed(2)+' s',s.distance.toFixed(1)+' mm',{reached:'Reached',finished:'Time limit',fallen:'Lost balance'}[s.status]];
    for(const value of values){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}
    $('trials').append(row);
  }
}
async function initialize() {
  while(!ready) {
    try {
      const s=await api('brain/status');
      if(s.status==='error')throw new Error(s.message);
      if(s.status==='ready') {
        const m=s.metadata;
        $('neurons').textContent=m.neurons.toLocaleString();$('edges').textContent=(m.connections/1e6).toFixed(2)+'M';
        $('compute-info').textContent=`${m.gpu||m.backend} · ${m.hostname}${m.slurm_job_id?' · Slurm '+m.slurm_job_id:''}`;
        await brainView.load();ready=true;controls();break;
      }
      message(s.message||'Brain loading…');
    } catch(error) { message(error.message,true); }
    await new Promise(resolve=>setTimeout(resolve,1500));
  }
  try {
    const response=await fetch('validation.json');
    $('validation').textContent=response.ok?(await response.json()).summary:'Try matched trials to assess each sensory contribution.';
  } catch { $('validation').textContent='Measured validation results are unavailable.'; }
  while(true) {
    try { if(!busy){const next=await api('maze/status');if(!busy)render(next);} }
    catch(error) { message('Connection lost: '+error.message,true); }
    await new Promise(resolve=>setTimeout(resolve,150));
  }
}
initialize();
