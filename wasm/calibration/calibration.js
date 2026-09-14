const $=id=>document.getElementById(id);
const active=$('active-video'), passive=$('passive-video'), slider=$('time');
let trials, playing=false, seeking=false;
const NS='http://www.w3.org/2000/svg';
function node(tag,attrs={},text=''){const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);n.textContent=text;return n;}
function chart(id,series,ymin,ymax,unit){
  const svg=$(id), x=t=>46+t/1.6*535, y=v=>194-(v-ymin)/(ymax-ymin)*150;
  svg.append(node('rect',{x:x(.3),y:35,width:x(.6)-x(.3),height:159,fill:'#faf0d9'}));
  for(let i=0;i<4;i++){const v=ymin+(ymax-ymin)*i/3;svg.append(node('line',{x1:46,y1:y(v),x2:581,y2:y(v),stroke:'#e9eee7'}));svg.append(node('text',{x:39,y:y(v)+4,'text-anchor':'end',fill:'#6d7870'},String(Math.round(v))));}
  for(const t of [0,.4,.8,1.2,1.6])svg.append(node('text',{x:x(t),y:215,'text-anchor':'middle',fill:'#6d7870'},`${t}s`));
  svg.append(node('text',{x:46,y:25,fill:'#6d7870'},unit));
  series.forEach((s,i)=>{svg.append(node('path',{d:s.data.map((r,j)=>`${j?'L':'M'}${x(r.time_s).toFixed(2)},${y(r[s.key]).toFixed(2)}`).join(' '),fill:'none',stroke:s.color,'stroke-width':2}));svg.append(node('text',{x:140+i*140,y:25,fill:s.color},s.name));});
  svg.append(node('line',{class:'cursor',x1:46,y1:35,x2:46,y2:194,stroke:'#2b4537','stroke-dasharray':'3 3'}));
}
function update(t){
  slider.value=t; $('clock').textContent=`${t.toFixed(3)} s`;
  $('phase').textContent=t<.3?'Resting pose · external fixture holds the tibia':t<.4?'External fixture extends the tibia':t<.6?'Extension held · proprioceptors respond':'Released · muscle and passive forces move the tibia';
  const row=trials.connected.trace.reduce((a,b)=>Math.abs(b.time_s-t)<Math.abs(a.time_s-t)?b:a);
  for(const name of ['sensory','flexor','extensor'])$(name).textContent=`${row[name+'_hz'].toFixed(1)} Hz`;
  document.querySelectorAll('.cursor').forEach(n=>{const x=46+t/1.6*535;n.setAttribute('x1',x);n.setAttribute('x2',x);});
}
function pause(){playing=false;active.pause();passive.pause();$('play').textContent='Play comparison';}
$('play').addEventListener('click',async()=>{
  if(playing){pause();return;}
  if(active.ended||active.currentTime>=active.duration-.05){active.currentTime=0;passive.currentTime=0;}
  passive.currentTime=active.currentTime;
  try{await Promise.all([active.play(),passive.play()]);playing=true;$('play').textContent='Pause comparison';}catch(e){pause();$('phase').textContent=`Replay could not start: ${e.message}`;}
});
$('reset').addEventListener('click',()=>{pause();active.currentTime=0;passive.currentTime=0;update(0);});
slider.addEventListener('input',()=>{pause();seeking=true;const t=Number(slider.value);active.currentTime=t*5;passive.currentTime=t*5;update(t);seeking=false;});
active.addEventListener('ended',pause);
function tick(){if(playing&&!seeking){const t=active.currentTime/5;update(t);if(Math.abs(passive.currentTime-active.currentTime)>.08)passive.currentTime=active.currentTime;}requestAnimationFrame(tick);}
try{
  const responses=await Promise.all(['report.json','trials.json'].map(async url=>{const r=await fetch(url);if(!r.ok)throw new Error(`${url}: HTTP ${r.status}`);return r.json();}));
  const [report,data]=responses;trials=data;
  $('cells').textContent=report.connectome.neurons.toLocaleString();$('edges').textContent=report.connectome.edges.toLocaleString();
  $('effect').textContent=`${report.causal_effect.max_additional_flexion_deg.toFixed(2)}°`;
  $('spikes').textContent=`${report.causal_effect.motor_spikes} / ${report.causal_effect.disconnected_motor_spikes}`;
  const score=report.calcium_fits[0].scores.find(s=>s.split==='test');
  $('fit-result').textContent=`The held-out ramp has ${(100*score.relative_mse).toFixed(1)}% of the zero-response baseline error. Several held-out swing recordings are worse than that baseline. The fit is not accepted for neural calibration.`;
  for(const reason of report.promotion.reasons){const li=document.createElement('li');li.textContent=reason.replace('The frozen observation fit','The earlier 13Bα observation fit');$('limits').append(li);}
  chart('angle-chart',[{name:'Connected',key:'angle_deg',data:trials.connected.trace,color:'#b44226'},{name:'Disconnected',key:'angle_deg',data:trials.disconnected.trace,color:'#64748b'}],90,125,'degrees');
  chart('rate-chart',[{name:'Sensory',key:'sensory_hz',data:trials.connected.trace,color:'#64748b'},{name:'Flexor',key:'flexor_hz',data:trials.connected.trace,color:'#b44226'},{name:'Extensor',key:'extensor_hz',data:trials.connected.trace,color:'#459578'}],0,180,'Hz');
  for(const id of ['play','reset','time'])$(id).disabled=false;
  update(0);requestAnimationFrame(tick);
}catch(e){$('phase').textContent=`Unable to load the calibration results: ${e.message}`;}
