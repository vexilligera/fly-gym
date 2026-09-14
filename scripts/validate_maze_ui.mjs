// Exercise the real maze UI with a minimal DOM and delayed HTTP responses.
// Run with: node scripts/validate_maze_ui.mjs
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source=readFileSync(new URL('../wasm/maze/maze.js',import.meta.url),'utf8')
  .replace(/^import .*;\n/,'').replace(/initialize\(\);\s*$/,'');
const html=readFileSync(new URL('../wasm/maze/index.html',import.meta.url),'utf8');
const stream='/api/maze/camera.mjpg';
const picture=body=>'data:image/jpeg;base64,'+body;

function element() {
  const attrs=new Map();
  return {
    hidden:false, dataset:{}, textContent:'', changes:[],
    classList:{toggle(){}}, replaceChildren(){}, append(){},
    scrollIntoView(){}, reportValidity(){return true;},
    getContext(){return {clearRect(){}};},
    getAttribute(key){return attrs.get(key)??null;},
    setAttribute(key,value){attrs.set(key,value);},
    removeAttribute(key){attrs.delete(key);this.changes.push({remove:key});},
    get src(){return attrs.get('src');},
    set src(value){attrs.set('src',value);this.changes.push({src:value});},
  };
}
function harness() {
  const elements=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(([,id])=>[id,element()]));
  const requests=[];
  const context=vm.createContext({
    document:{getElementById:id=>{assert.ok(elements[id],id);return elements[id];},
      createElement:element,createElementNS:element},
    BrainView:class {update(){} reset(){}}, AbortController, Date,
    fetch:(url,options)=>new Promise((resolve,reject)=>requests.push({url,options,reject,
      reply:data=>resolve({ok:true,json:async()=>data})})),
  });
  vm.runInContext(source,context);
  vm.runInContext("ready=true;world={layout:'simple'};",context);
  return {elements,requests,run:code=>vm.runInContext(code,context),
    render(s){context.input=s;vm.runInContext('render(input)',context);}};
}
function sample(trial,frame,status='running',body=`${trial}-${frame}`) {
  return {trial_id:trial,frame,status,time:frame*.02,sensory_time:frame*.02,
    config:{layout:'simple',condition:'combined',food_odor:true},
    brain:null,images:{eyes:['left','right'],...(body===null?{}:{body})},
    score:{distance_mm:5},playback_speed:.4,odor:[.1,.2],path:[[0,0]],
    position:[0,0],heading_deg:75,done:false};
}

{
  const h=harness(),img=h.elements.body;
  h.render(sample('old',0,'paused'));
  h.render(sample('old',20));
  assert.equal(img.src,stream);
  const reset=h.run("command('reset',{})");
  assert.equal(img.getAttribute('src'),null,'Reset disconnects MJPEG immediately');
  assert.equal(img.hidden,true,'Reset does not present the previous run as current');
  assert.equal(h.elements['body-loading'].hidden,false);
  h.requests.shift().reply(sample('new',0,'paused','reset-image'));
  await reset;
  assert.equal(img.src,picture('reset-image'),'Reset uses its authoritative frame');
  assert.equal(img.hidden,false);
  assert.equal(h.elements['body-time'].textContent,'Camera 0.000 s');
  h.render(sample('new',0,'paused','reset-image'));
  assert.equal(img.src,picture('reset-image'),'Paused polls keep the reset frame');

  // Resume/pause can change status without advancing the numeric frame.
  h.render(sample('new',0,'running','reset-image'));
  assert.equal(img.src,stream,'Resume opens a fresh stream at the same frame');
  h.render(sample('new',0,'paused','paused-image'));
  assert.equal(img.src,picture('paused-image'),'Pause drops buffered frames');
  h.render(sample('new',1));
  const changes=img.changes.length;
  h.render(sample('other',0,'running','other-start'));
  assert.ok(img.changes.slice(changes).some(c=>c.remove==='src'));
  assert.equal(img.src,picture('other-start'),'A new running trial also flushes the stream');

  // A body=0 poll can observe another client's reset or a completed trial.
  h.render(sample('other',1));
  h.render(sample('external',0,'paused',null));
  assert.equal(img.hidden,true,'Never reuse an image from a different trial');
  h.render(sample('external',0,'paused','external-reset'));
  assert.equal(img.src,picture('external-reset'),'Same-frame full poll fills the missing snapshot');
  h.render(sample('external',1));
  h.render(sample('external',2,'reached',null));
  assert.equal(img.hidden,true,'Wait for the arrival image instead of showing an old start image');
  h.render(sample('external',2,'reached','arrival'));
  assert.equal(img.src,picture('arrival'));

  const feeding=sample('external',3,'tasting','mouth');
  feeding.taste={time:.02,rate_hz:200,phase:'baseline',trace:[]};
  h.render(feeding);
  assert.equal(img.src,picture('mouth'));
  h.render(sample('after-taste',0,'paused','maze-again'));
  assert.equal(img.src,picture('maze-again'),'Reset leaves the feeding close-up');
  assert.equal(h.elements['body-heading'].textContent,'The sugar maze');

  h.render(sample('after-taste',1));
  img.onerror();
  assert.notEqual(img.src,stream,'Stream failure falls back to the last snapshot');
  h.render(sample('after-taste',1,'running','fallback'));
  assert.equal(img.src,picture('fallback'),'Fallback refreshes even at the same frame');
}

for(const failOldPoll of [false,true]) {
  const h=harness();
  h.render(sample('old',0,'paused'));
  h.render(sample('old',10));
  const pending=h.run('pollStatus()');
  const oldRequest=h.requests.shift();
  assert.equal(oldRequest.url,'/api/maze/status?body=0');
  const reset=h.run("command('reset',{})");
  assert.equal(oldRequest.options.signal.aborted,true,'Reset cancels the pending poll');
  h.requests.shift().reply(sample('new',0,'paused','reset-image'));
  await reset;
  // Model a response already being decoded when aborted, or a late network error.
  if(failOldPoll)oldRequest.reject(new Error('Old connection failed'));
  else oldRequest.reply(sample('old',11));
  await pending;
  assert.equal(h.run('state.trial_id'),'new','Late pre-reset responses cannot restore the old trial');
  assert.equal(h.elements.body.src,picture('reset-image'));
  assert.match(h.elements.status.textContent,/Paused/,'A late error cannot replace reset status');
  const refresh=h.run('pollStatus()');
  const request=h.requests.shift();
  assert.equal(request.url,'/api/maze/status','Paused polling requests the full snapshot');
  request.reply(sample('new',0,'paused','reset-image'));
  await refresh;
}
console.log('PASS: reset/resume/pause, trial and feeding transitions, snapshot fallback, and delayed poll races');

{
  const h=harness(),s=sample('neuron',2);
  s.config.controller='descending';
  s.brain={filtered_rates_hz:{DNp09_left:12.345},rates_hz:{DNp09_left:50},olfaction:{ORN_rates_hz:[10,20],PN_spikes:[1,2]}};
  s.decoder={mode:'descending',forward:.2,reverse:.1,turn:.3,gains:[-.08,.28]};
  h.render(s);
  assert.match(h.elements.status.textContent,/Neuron readouts/);
  assert.equal(h.elements['dn-DNp09-left'].textContent,'12.3 Hz');
  assert.equal(h.elements['gain-left'].textContent,'-0.08');
  assert.equal(h.elements['wall-front'].textContent,'unused');
  assert.match(h.elements.steering.textContent,/Zero readout/);
  const silenced={...s,trial_id:'silenced',config:{...s.config,silence_descending:true}};
  h.render(silenced);
  assert.match(h.elements.steering.textContent,/suppressed/);
  h.elements.controller.value='sensory_policy';
  h.elements['silence-descending'].checked=true;
  h.elements.controller.onchange();
  assert.equal(h.elements['silence-descending'].disabled,true);
  assert.equal(h.elements['silence-descending'].checked,false);
  assert.equal(h.run('config().controller'),'sensory_policy');
  assert.equal(h.run('config().silence_descending'),false);
}
console.log('PASS: DN readouts, active controller labels, silencing control and comparison config');
