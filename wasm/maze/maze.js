import { BrainView } from '../connectome/brain-view.js';

const $ = id => document.getElementById(id);
const brainView = new BrainView();
const names = {combined:'Vision + smell',vision_only:'Vision only',odor_only:'Smell only',neither:'Both disconnected'};
const layouts = {simple:'Simple maze',complex:'Branching maze'};
const controllers = {descending:'Neuron readouts',sensory_policy:'Sensory policy'};
const descriptions = {
  combined:'Eye images drive R1–6 cells and local antenna samples drive DM1 odor neurons in the same brain step.',
  vision_only:'Control: eyes remain connected; odor input is zero.',
  odor_only:'Control: antenna samples remain connected; eye input is zero.',
  neither:'Control: both sensory input streams are zero. From reset, the brain has no spontaneous drive.',
};
const controllerDescriptions = {
  descending:'DNp09, DNa02, and MDN activity sets the leg gains. No constant walking drive, odor-following rule, or wall-avoidance rule. The neuron-to-leg mapping and leg CPG remain engineered. Useful navigation is not guaranteed.',
  sensory_policy:'Comparison: the previous controller combines L2 and ORN activity with constant walking drive, odor following, wall avoidance, and held escape turns.',
};
let state = {status:'idle'}, ready = false, busy = false, lastFrame = -1;
let trialId = null, recordedId = null, trials = [], world = null, loadingWorld = false;
let cameraStreaming = false, cameraRetryAt = 0, lastSnapshot = null;
let cameraView = null, commandEpoch = 0, pendingPoll = null;
let validationReport = null, readoutReport = null;

function renderValidation(layout,controller) {
  const report=controller==='descending'?readoutReport:layout==='complex'?validationReport?.complex:validationReport;
  $('validation').textContent=report?.summary||'Measured validation results are unavailable for this layout.';
}

function message(text, error=false) {
  $('status').textContent=text;
  $('status').classList.toggle('error',error);
}
function controls() {
  $('start').disabled=$('reset').disabled=!ready||busy;
  $('pause').disabled=!ready||busy||!['running','tasting','paused'].includes(state.status)||!state.config;
  $('pause').textContent=['running','tasting'].includes(state.status)?'Pause':'Resume';
  $('taste-start').disabled=$('taste-control').disabled=!ready||busy||state.status!=='reached'||!state.done;
}
function config() {
  return {layout:$('layout').value,condition:$('condition').value,heading_deg:Number($('heading').value),
    seed:Number($('seed').value),duration:Number($('duration').value),food_odor:$('food-odor').checked,
    controller:$('controller').value,silence_descending:$('controller').value==='descending'&&$('silence-descending').checked};
}
async function api(path, args, signal) {
  const response=await fetch('/api/'+path,args===undefined?{signal}:{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(args),signal});
  const data=await response.json().catch(()=>{throw new Error('Compute service is temporarily unavailable.');});
  if(!response.ok)throw new Error(data.error||'Compute service request failed');
  return data;
}
async function command(action,args={}) {
  if(!$('seed').reportValidity())return;
  ++commandEpoch;
  pendingPoll?.abort();
  busy=true;controls();
  stopCamera();
  if(action==='reset'||(action==='start'&&Object.keys(args).length)) {
    $('body').hidden=true;
    $('body-loading').hidden=false;
    $('body-loading').textContent='Resetting MuJoCo view…';
    $('body-time').textContent='Resetting…';
  }
  message(action==='pause'?'Finishing the current simulation bin…':action==='taste'?'Preparing the sugar-taste assay…':'Preparing the maze and both senses…');
  try { render(await api('maze/'+action,args),true); }
  catch(error) { message(error.message,true); }
  finally { busy=false;controls(); }
}
$('start').onclick=()=>command('start',config());
$('reset').onclick=()=>command('reset',config());
$('pause').onclick=()=>command(['running','tasting'].includes(state.status)?'pause':'start');
async function watchTaste(rate_hz) {
  await command('taste',{rate_hz});
  if(state.status==='tasting')$('brain-canvas').scrollIntoView({behavior:'smooth',block:'center'});
}
$('taste-start').onclick=()=>watchTaste(Number($('taste-rate').value));
$('taste-control').onclick=()=>watchTaste(0);
$('enlarge-fly').onclick=()=>{
  const expanded=document.querySelector('.vision-workspace').classList.toggle('camera-enlarged');
  $('enlarge-fly').textContent=expanded?'Restore view':'Enlarge fly';
  $('enlarge-fly').setAttribute('aria-expanded',String(expanded));
  $('body-heading').scrollIntoView({behavior:'smooth',block:'start'});
};
$('condition').onchange=()=>{$('condition-description').textContent=descriptions[$('condition').value];};
$('controller').onchange=()=>{
  $('controller-description').textContent=controllerDescriptions[$('controller').value];
  $('silence-descending').disabled=$('controller').value!=='descending';
  if($('silence-descending').disabled)$('silence-descending').checked=false;
};
$('heading').oninput=()=>{$('heading-value').textContent=$('heading').value+'°';};
$('show-field').onchange=drawField;
$('layout').onchange=()=>{
  $('layout-description').textContent=$('layout').value==='complex'
    ?'25 cells · 5 dead ends · 8 turns on the route to sugar. Changes apply to the next trial.'
    :'The original two-baffle arena. Changes apply to the next trial.';
};
$('body').onerror=()=>{
  if(!cameraStreaming)return;
  stopCamera();cameraRetryAt=Date.now()+5000;
  if(lastSnapshot)$('body').src='data:image/jpeg;base64,'+lastSnapshot.body;
  $('body-time').textContent='Snapshot fallback';
};

function stopCamera() {
  if(cameraStreaming)$('body').removeAttribute('src');
  cameraStreaming=false;
}
function renderCamera(s,newTrial,forceSnapshot) {
  const view=s.taste?'proboscis':'maze';
  if(newTrial||view!==cameraView) {
    stopCamera();lastSnapshot=null;cameraRetryAt=0;cameraView=view;
    $('body').removeAttribute('src');
  }
  if(s.images?.body)lastSnapshot={body:s.images.body,time:s.taste?.time??s.time,frame:s.frame};
  // Reset/pause must replace the buffered stream with the command's snapshot.
  // A new trial first shows its snapshot; a later running poll opens a new stream.
  // Feeding snapshots keep mouth and brain timestamps paired.
  if(newTrial||forceSnapshot||s.status!=='running'||s.taste) {
    stopCamera();
    if(lastSnapshot?.frame!==s.frame)lastSnapshot=null;
  } else if(!cameraStreaming&&Date.now()>=cameraRetryAt) {
    cameraStreaming=true;
    $('body').src='/api/maze/camera.mjpg';
  }
  if(!cameraStreaming&&lastSnapshot) {
    const src='data:image/jpeg;base64,'+lastSnapshot.body;
    if($('body').getAttribute('src')!==src)$('body').src=src;
  }
  const visible=cameraStreaming||Boolean(lastSnapshot);
  $('body').hidden=!visible;
  $('body-loading').hidden=visible;
  $('body-loading').textContent='Waiting for the current camera frame…';
  $('body-time').textContent=cameraStreaming?'Streaming camera':lastSnapshot
    ?s.taste?`Mouth + brain ${lastSnapshot.time.toFixed(2)} s`:`Camera ${lastSnapshot.time.toFixed(3)} s`
    :'Waiting for camera';
}

async function loadWorld(layout) {
  if(world?.layout===layout||loadingWorld)return;
  loadingWorld=true;
  try {
    const next=await api('maze/world');
    if(next.layout!==state.config?.layout)return;
    world=next;
    $('walls').replaceChildren();
    for(const [x,y,hx,hy] of world.walls) {
      const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
      for(const [key,value] of Object.entries({x:x-hx,y:y-hy,width:2*hx,height:2*hy,fill:'#526574',stroke:'#91a4b3','stroke-width':.12}))rect.setAttribute(key,value);
      $('walls').append(rect);
    }
    $('arena-description').textContent=`${world.name} · 40 × 40 mm · ${world.description}`;
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
function render(s,forceSnapshot=false) {
  state=s;controls();
  if(s.status==='error'){message(s.message||'Maze simulation stopped with an error',true);return;}
  if(!s.config){message(s.message||'Ready to run the sugar maze');return;}
  const newTrial=s.trial_id!==trialId;
  if(newTrial){
    trialId=s.trial_id;lastFrame=-1;
    renderValidation(s.config.layout,s.config.controller);
    if(world?.layout!==s.config.layout){world=null;$('walls').replaceChildren();}
    drawField();
  }
  loadWorld(s.config.layout);
  const statusText={running:'Live sensory feedback',tasting:'Live sugar taste → proboscis',paused:'Paused',reached:s.taste?'Sugar / proboscis assay complete':'Sugar zone reached',finished:'Time limit reached',fallen:'Fly lost balance'}[s.status]||s.status;
  const isReadout=s.config.controller==='descending';
  message(`● ${statusText} · ${controllers[s.config.controller]||'Sensory policy'}${s.config.silence_descending?' · DNs silenced':''} · ${layouts[s.config.layout]||'Simple maze'} · ${names[s.config.condition]}${s.config.food_odor?'':' · Food odor off'}`);
  $('trial-state').textContent=s.status.toUpperCase();
  $('brain-live-state').textContent=['running','tasting'].includes(s.status)?'Live · latest bin':'Held · last bin';
  $('brain-live-state').dataset.running=['running','tasting'].includes(s.status);
  // Status and camera changes can arrive without a new simulation frame.
  renderCamera(s,newTrial,forceSnapshot);
  if(s.frame===lastFrame)return;
  lastFrame=s.frame;
  if(s.brain)brainView.update(s.brain);else brainView.reset();
  if(s.images) {
    for(const [i,side] of ['left','right'].entries())$('eye-'+side).src='data:image/jpeg;base64,'+s.images.eyes[i];
  }
  $('distance').textContent=s.taste?(s.taste.proboscis?.contact?.touching?'Labellum touching sugar':'No mouth contact'):'At brain readout: '+s.score.distance_mm.toFixed(1)+' mm';
  $('body-heading').textContent=s.taste?'Proboscis close-up':'The sugar maze';
  $('body').alt=s.taste?'Live MuJoCo proboscis extension and turning driven by measured MN9 activity':'Live overhead MuJoCo view of the sugar maze';
  $('camera-description').textContent=s.taste?'Feeding pose facing sugar · Taste requires mouth contact · Legs held.':'Camera streams independently. Brain readouts may update more slowly.';
  $('speed').textContent=s.taste?'Mouth + brain · 0.2× playback':s.playback_speed.toFixed(2)+'× real time';
  const vision=['combined','vision_only'].includes(s.config.condition);
  const smell=['combined','odor_only'].includes(s.config.condition);
  $('eyes-state').textContent=s.taste?'Held arrival image':vision?'721 samples / eye':'Input disconnected';
  $('sensory-time').textContent=s.taste?`Assay brain time ${s.taste.time.toFixed(3)} s; feeding pose shown; navigation arrival ${s.time.toFixed(3)} s.`:`Sensors sampled at ${s.sensory_time.toFixed(3)} s → brain/body at ${s.time.toFixed(3)} s.`;
  $('odor-state').textContent=s.taste?'Arrival samples held for reference. Vision and odor inputs are off during the taste assay.':!s.config.food_odor?'Food odor off: both antenna samples are zero.':smell?'Local food-odor concentration, normalized 0–1.':'Odor is present here; neural odor input is disconnected.';
  const d=s.taste?{}:s.decoder||{}, o=s.brain?.olfaction;
  for(const [i,side] of ['left','right'].entries()) {
    $('odor-'+side).textContent=s.odor[i].toFixed(3);
    $('odor-meter-'+side).value=s.odor[i];
    $('orn-'+side).textContent=(o?.ORN_rates_hz[i]||0).toFixed(0)+' Hz';
    $('pn-'+side).textContent=o?.PN_spikes[i]||0;
    $('gain-'+side).textContent=s.taste?'held':(d.gains?.[i]||0).toFixed(2);
    for(const cell of ['DNp09','DNa02','MDN']) {
      const key=cell+'_'+side;
      $('dn-'+cell+'-'+side).textContent=(s.brain?.filtered_rates_hz[key]||0).toFixed(1)+' Hz';
      $('dn-'+cell+'-'+side).title=`Latest 20 ms: ${s.brain?.rates_hz[key]||0} Hz`;
    }
  }
  $('l2-spikes').textContent=(d.L2_spikes||0).toLocaleString();
  $('wall-front').textContent=isReadout?'unused':(d.wall_front||0).toFixed(2);
  $('vision-detail').textContent=isReadout?'Dark wall contrast enters R1–6 cells. L2 counts are displayed as a sensory response; movement reads descending neurons.':'Dark wall contrast enters R1–6 cells. The comparison policy uses downstream L2 activity for wall avoidance.';
  $('motor-heading').textContent=isReadout?'3. Descending neurons → legs':'3. Sensory policy → legs';
  $('motor-detail').textContent=controllerDescriptions[s.config.controller]||controllerDescriptions.sensory_policy;
  $('active').textContent=(s.brain?.active_neurons||0).toLocaleString();
  $('steering').textContent=s.taste?'Legs held; MN9 drives the mouth servos. Navigation gains are inactive.':isReadout
    ?s.config.silence_descending?'Control: DNp09, DNa02, and MDN spikes are suppressed throughout this trial. Sensory inputs remain connected.'
      :`Forward ${(d.forward||0).toFixed(3)} · Reverse ${(d.reverse||0).toFixed(3)} · Turn ${(d.turn||0).toFixed(3)}. Zero readout produces zero leg gain.`
    :d.escape?'Turning away from a strong front-wall response.':`Odor steering ${(d.odor_turn||0).toFixed(2)} · Visual avoidance ${(d.visual_turn||0).toFixed(2)}. Positive turns left.`;
  $('ground-truth').textContent=`Evaluation: ${s.score.distance_mm.toFixed(1)} mm from center. The controller does not receive this distance.`;
  $('result').textContent=(s.status==='reached'||s.taste)?`Entered the 2.5 mm food zone after ${s.time.toFixed(2)} simulated seconds. Watch the proboscis and brain response below; ingestion is not modeled.`:s.status==='fallen'?'The fly lost balance; the trial stopped.':s.done?'Time limit reached without entering the food zone.':'A trial ends at the food zone, the time limit, or loss of balance.';
  renderTaste(s.taste,s.brain?.sugar);
  $('path').setAttribute('d',s.path.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' '));
  $('fly-marker').setAttribute('transform',`translate(${s.position[0]} ${s.position[1]}) rotate(${s.heading_deg})`);
  if(s.done&&recordedId!==trialId) {
    recordedId=trialId;
    trials.unshift({config:s.config,time:s.time,distance:s.score.distance_mm,status:s.status});
    trials=trials.slice(0,12);renderTrials();
  }
}
function renderTaste(taste,readout) {
  const trace=taste?.trace||[];
  $('taste-phase').textContent=!taste?'Available after arrival':`${taste.done?'Complete':taste.phase} · ${taste.time.toFixed(2)} / 8.00 s`;
  $('taste-detail').textContent=!taste?'21 released sugar GRNs are stimulated. Both MN9 motor neurons are observed downstream, without direct stimulation.':`${taste.rate_hz===0?'No-taste control':taste.rate_hz+' Hz sugar input'} · 21 GRNs · reset neural baseline · staged feeding pose. Contact gates taste; MN9 drives approximate mouth servos. Liquid intake is not modeled.`;
  $('taste-grn').textContent=readout?readout.GRN_hz.toFixed(1)+' Hz':'—';
  for(const side of ['left','right'])$('taste-mn-'+side).textContent=readout?readout.MN9_hz[side].toFixed(1)+' Hz':'—';
  $('mouth-contact').textContent=!taste?.proboscis?.contact?'—':taste.proboscis.contact.touching?'Touching sugar':'No contact';
  const mouth=taste?.proboscis?.angles_deg;
  $('mouth-extension').textContent=mouth?(-mouth.rostrum_pitch).toFixed(1)+'°':'—';
  $('mouth-turn').textContent=mouth?`${Math.abs(mouth.mouth_yaw).toFixed(1)}° ${mouth.mouth_yaw>=0?'left':'right'}`:'—';
  const maximum=Math.max(200,...trace.flatMap(p=>[p.GRN_hz,p.MN9_left_hz,p.MN9_right_hz]));
  $('taste-chart-max').textContent=Math.ceil(maximum);
  for(const [id,key] of [['grn','GRN_hz'],['left','MN9_left_hz'],['right','MN9_right_hz']]) {
    $('taste-line-'+id).setAttribute('d',trace.map((p,i)=>`${i?'L':'M'}${40+p.time*46} ${120-p[key]/maximum*105}`).join(' '));
  }
}
function renderTrials() {
  $('trials').replaceChildren();
  for(const s of trials) {
    const row=document.createElement('tr');
    const values=[layouts[s.config.layout]||'Simple maze',(controllers[s.config.controller]||'Sensory policy')+(s.config.silence_descending?' · silenced':''),names[s.config.condition],s.config.food_odor?'On':'Off',`${s.config.heading_deg}° / ${s.config.seed}`,s.time.toFixed(2)+' s',s.distance.toFixed(1)+' mm',{reached:'Reached',finished:'Time limit',fallen:'Lost balance'}[s.status]];
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
    if(response.ok)validationReport=await response.json();
    const readoutResponse=await fetch('readout-validation.json');
    if(readoutResponse.ok)readoutReport=await readoutResponse.json();
    renderValidation(state.config?.layout||$('layout').value,state.config?.controller||$('controller').value);
  } catch { $('validation').textContent='Measured validation results are unavailable.'; }
  while(true) {
    await pollStatus();
    await new Promise(resolve=>setTimeout(resolve,150));
  }
}
async function pollStatus() {
  if(busy)return;
  const epoch=commandEpoch, controller=new AbortController();
  pendingPoll=controller;
  try {
    const next=await api('maze/status'+(cameraStreaming?'?body=0':''),undefined,controller.signal);
    // A pre-command response may finish after Reset, even after busy clears.
    if(!busy&&epoch===commandEpoch)render(next);
  } catch(error) {
    if(!controller.signal.aborted&&epoch===commandEpoch)message('Connection lost: '+error.message,true);
  } finally {
    if(pendingPoll===controller)pendingPoll=null;
  }
}
initialize();
