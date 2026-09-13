// Optional inspector for the local lab. Values come from the running game;
// no second simulation or prerecorded motion is used by the lab UI.
export function attachInspector(game) {
  const style = document.createElement('style');
  style.textContent = '#levels, #hud, #help { display:none } #stats { top:16px; right:16px }';
  document.head.appendChild(style);
  const overlay = document.getElementById('overlay');
  const knee = game.meta.actuators.find(a => a.joint === 'lf_trochanterfemur-lf_tibia-pitch');
  if (!knee) throw new Error('The inspector requires the left-front tibia actuator.');
  let lastSample = -Infinity;
  let generation = 0;
  const heldTimers = new Set();
  // In connectome mode the parent owns the clock and controls. Do not let the
  // game's keyboard shortcuts switch controllers or reset only the body.
  window.addEventListener('keydown', e => {
    if (game.externalClock && !e.metaKey && !e.ctrlKey && !e.altKey) e.stopImmediatePropagation();
  }, true);
  function run() {
    game.phase = 'running';
    game.inspectorPaused = false;
    overlay.classList.add('hidden');
  }
  function clearSteps() {
    for (const timer of heldTimers) clearTimeout(timer);
    heldTimers.clear();
    game.input.held.clear();
  }
  function reset() {
    clearSteps();
    game.restart();
    game.externalSteps = 0;
    generation++;
    run();
  }
  function step(key) {
    game.input.held.add(key);
    const timer = setTimeout(() => {
      game.input.held.delete(key);
      heldTimers.delete(timer);
    }, 80);
    heldTimers.add(timer);
  }
  window.addEventListener('message', e => {
    if (e.source !== parent || e.origin !== location.origin || e.data?.type !== 'nmf-command') return;
    const command = e.data;
    if (command.action === 'clock') {
      game.externalClock = !!command.external;
    } else if (command.action === 'advance' && game.externalClock) {
      if (![command.left, command.right].every(Number.isFinite)) return;
      game.input.gainL = Math.max(-1.2, Math.min(1.2, command.left));
      game.input.gainR = Math.max(-1.2, Math.min(1.2, command.right));
      game._padState = null;
      // One brain batch = 20 ms. Physics cannot outrun the neural simulation.
      const steps = Math.round(.02 / game.dt);
      for (let i = 0; i < steps; i++) game._physicsStep();
      game.externalSteps = (game.externalSteps || 0) + steps;
      parent.postMessage({type:'nmf-advanced', id:command.id, time:game.simTime}, location.origin);
    } else if (command.action === 'drive' && game.level === 'CPG') {
      if (![command.left, command.right].every(Number.isFinite)) return;
      game.input.gainL = Math.max(-1.2, Math.min(1.2, command.left));
      game.input.gainR = Math.max(-1.2, Math.min(1.2, command.right));
      if (game.phase === 'ready') run();
    } else if (command.action === 'pause') {
      game.inspectorPaused = !game.inspectorPaused;
    } else if (command.action === 'reset') {
      reset();
    } else if (command.action === 'level' && ['CPG', 'tripod', 'single'].includes(command.level)) {
      clearSteps();
      game.setLevel(command.level);
      generation++;
      run();
    } else if (command.action === 'step') {
      const keys = game.level === 'tripod' ? ['g', 'h'] : game.level === 'single' ? [...'tgbzhn'] : [];
      if (Number.isInteger(command.index) && keys[command.index]) step(keys[command.index]);
    }
  });
  game.inspector = {
    sample(now) {
      if (now - lastSample < 50) return;
      lastSample = now;
      const { controller: c, data: d, meta, level, bodyId: b } = game;
      const pad = game._padState;
      const phases = level === 'CPG' ? c.phases : level === 'single' ? c.legPhases : c.tripodMap.map(g => c.tripodPhases[g]);
      const position = Array.from(d.xpos.slice(3 * b, 3 * b + 3));
      parent.postMessage({
        type: 'nmf-state', generation, time: game.simTime, level,
        paused: !!game.inspectorPaused, phase: game.phase,
        gains: pad?.active ? [pad.gainL, pad.gainR] : [game.input.gainL, game.input.gainR],
        phases: Array.from(phases, p => ((p % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)),
        magnitudes: level === 'CPG' ? Array.from(c.mags) : Array(6).fill(1),
        adhesion: meta.adhesion.map(id => d.ctrl[id] > 0.5),
        contacts: d.ncon, position,
        heading: Math.atan2(d.xmat[9 * b + 3], d.xmat[9 * b]) * 180 / Math.PI,
        joint: { name: knee.joint, target: d.ctrl[knee.id], actual: d.qpos[knee.qposadr] },
        frequency: meta.control.cpg.intrinsic_freqs[0],
      }, location.origin);
    },
  };
  reset();
}
