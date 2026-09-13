import * as THREE from 'three';
import { OrbitControls } from '../shared/vendor/three/OrbitControls.js';

const $ = id => document.getElementById(id);

// One measured annotation anchor per neuron. All spike counts come from the
// backend's completed bin; rendering never synthesizes activity between bins.
export class BrainView {
  async load() {
    const response = await fetch('/api/brain/geometry');
    if (!response.ok) throw new Error('Could not load the brain coordinates.');
    const data = await response.json();
    if (data.schema !== 'flywire-v783-anchors-v1' || data.positions_um.length !== data.indices.length * 3) {
      throw new Error('Brain coordinate format does not match this viewer.');
    }
    this.indices = data.indices;
    this.counts = new Uint32Array(data.neuron_count);
    this.inputIndices = new Set();
    this.pointForNeuron = new Int32Array(data.neuron_count).fill(-1);
    this.positions = new Float32Array(data.positions_um.length);
    const bounds = new THREE.Box3();
    for (let p = 0; p < data.indices.length; p++) {
      this.pointForNeuron[data.indices[p]] = p;
      // Convert the source volume into an upright X / -Y display, with -Z
      // depth. This is an annotation-volume view, not a claimed anatomical view.
      const xyz = new THREE.Vector3(data.positions_um[3*p], -data.positions_um[3*p+1], -data.positions_um[3*p+2]);
      bounds.expandByPoint(xyz);
      xyz.toArray(this.positions, p * 3);
    }
    const center = bounds.getCenter(new THREE.Vector3());
    for (let p = 0; p < data.indices.length; p++) {
      this.positions[p*3] -= center.x;
      this.positions[p*3+1] -= center.y;
      this.positions[p*3+2] -= center.z;
    }
    this.extent = bounds.getSize(new THREE.Vector3());
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 1, 10000);
    this.renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    const canvas = this.renderer.domElement;
    canvas.setAttribute('aria-label', 'Interactive 3D brain: gray annotation anchors, teal responding neurons, amber directly stimulated neurons. Drag to rotate, scroll to zoom, click a point to inspect.');
    canvas.setAttribute('role', 'img');
    $('brain-canvas').append(canvas);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.minDistance = 80;
    this.controls.maxDistance = 4500;
    this.controls.enablePan = true;
    this.controls.addEventListener('change', () => this.draw());

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    geometry.computeBoundingSphere();
    this.context = new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0x8295ad, size: 1.2, sizeAttenuation: false,
      transparent: true, opacity: .15, depthWrite: false,
    }));
    this.scene.add(this.context);
    this.activeGeometry = new THREE.BufferGeometry();
    this.activeGeometry.boundingSphere = geometry.boundingSphere.clone();
    this.activePositions = new Float32Array(this.positions.length);
    this.strengths = new Float32Array(data.indices.length);
    this.directInputs = new Float32Array(data.indices.length);
    this.activeNeuronIndices = [];
    for (const [name, array, size] of [
      ['position', this.activePositions, 3], ['strength', this.strengths, 1], ['directInput', this.directInputs, 1],
    ]) this.activeGeometry.setAttribute(name, new THREE.BufferAttribute(array, size).setUsage(THREE.DynamicDrawUsage));
    this.activeGeometry.setDrawRange(0, 0);
    this.active = new THREE.Points(this.activeGeometry, new THREE.ShaderMaterial({
      uniforms: {pixelRatio: {value: this.renderer.getPixelRatio()}},
      vertexShader: `
        attribute float strength;
        attribute float directInput;
        uniform float pixelRatio;
        varying float inputColor;
        void main() {
          inputColor = directInput;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = (3.0 + sqrt(min(strength, 10.0)) * 1.5) * pixelRatio;
        }`,
      fragmentShader: `
        varying float inputColor;
        void main() {
          float r = length(gl_PointCoord - vec2(0.5));
          if (r > 0.5) discard;
          vec3 color = mix(vec3(0.29, 0.91, 0.79), vec3(1.0, 0.62, 0.28), inputColor);
          gl_FragColor = vec4(color, 0.95 * (1.0 - smoothstep(0.18, 0.5, r)));
        }`,
      transparent: true, depthTest: false, depthWrite: false,
    }));
    this.active.frustumCulled = false;
    this.active.renderOrder = 1;
    this.scene.add(this.active);

    const ring = document.createElement('canvas');
    ring.width = ring.height = 64;
    const c = ring.getContext('2d');
    c.strokeStyle = '#ffffff'; c.lineWidth = 3;
    c.beginPath(); c.arc(32, 32, 23, 0, Math.PI * 2); c.stroke();
    this.marker = new THREE.Sprite(new THREE.SpriteMaterial({map: new THREE.CanvasTexture(ring), depthTest: false, transparent: true}));
    this.marker.visible = false;
    this.marker.renderOrder = 2;
    this.scene.add(this.marker);
    this.selected = null;
    this.raycaster = new THREE.Raycaster();
    let pointerStart = null;
    canvas.addEventListener('pointerdown', e => {pointerStart = [e.clientX, e.clientY];});
    canvas.addEventListener('pointerup', e => {
      if (pointerStart && Math.hypot(e.clientX-pointerStart[0], e.clientY-pointerStart[1]) < 5) this.pick(e);
      pointerStart = null;
    });
    $('brain-home').onclick = () => this.home();
    $('brain-context').onchange = e => {this.context.visible = e.target.checked; this.draw();};
    $('clear-neuron').onclick = () => this.select(null);
    this.observer = new ResizeObserver(() => {
      const {width, height} = $('brain-canvas').getBoundingClientRect();
      if (!width || !height) return;
      const previousFit = this.fitDistance(this.camera.aspect);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height);
      if (!this.initialized) {this.initialized = true; this.home();}
      else {
        this.camera.position.sub(this.controls.target).multiplyScalar(this.fitDistance(this.camera.aspect) / previousFit).add(this.controls.target);
        this.controls.update();
      }
      this.draw();
    });
    this.observer.observe($('brain-canvas'));
    $('brain-geometry').textContent = `${data.positioned_count.toLocaleString()} located · ${data.unpositioned_count} without coordinates, still simulated`;
    $('brain-loading').hidden = true;
    this.reset();
  }

  fitDistance(aspect) {
    const fov = THREE.MathUtils.degToRad(this.camera.fov / 2);
    return Math.max(this.extent.y / 2 / Math.tan(fov), this.extent.x / 2 / Math.tan(fov) / aspect) * 1.14 + this.extent.z / 2;
  }

  home() {
    this.controls.target.set(0, 0, 0);
    this.camera.position.set(0, 0, this.fitDistance(this.camera.aspect));
    this.camera.up.set(0, 1, 0);
    this.controls.update();
    this.draw();
  }

  draw() {
    if (this.marker?.visible) {
      const scale = this.camera.position.distanceTo(this.marker.position) * 2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 28 / this.renderer.domElement.clientHeight;
      this.marker.scale.setScalar(scale);
    }
    this.renderer.render(this.scene, this.camera);
  }

  update(s) {
    this.counts.fill(0);
    this.inputIndices = new Set(s.activity.input_indices);
    this.activeNeuronIndices = [];
    let p = 0, missing = 0;
    const responseIndices = [], directIndices = [];
    s.activity.indices.forEach((index, i) => {
      this.counts[index] = s.activity.spike_counts[i];
      (this.inputIndices.has(index) ? directIndices : responseIndices).push(index);
    });
    // Draw directly stimulated cells last so their amber color remains legible
    // in dense activity, without changing any counts or spatial positions.
    for (const index of responseIndices.concat(directIndices)) {
      const point = this.pointForNeuron[index];
      if (point < 0) {missing++; continue;}
      this.activePositions.set(this.positions.subarray(point*3, point*3+3), p*3);
      this.strengths[p] = this.counts[index];
      this.directInputs[p] = this.inputIndices.has(index) ? 1 : 0;
      this.activeNeuronIndices.push(index);
      p++;
    }
    for (const attr of Object.values(this.activeGeometry.attributes)) {
      attr.clearUpdateRanges();
      if (p) attr.addUpdateRange(0, p * attr.itemSize);
      attr.needsUpdate = true;
    }
    this.activeGeometry.setDrawRange(0, p);
    $('brain-window').textContent = `${Math.max(0, s.time-s.step_seconds).toFixed(3)}–${s.time.toFixed(3)} s`;
    $('brain-visible').textContent = `${p.toLocaleString()} firing${missing ? ` · ${missing} unlocated` : ''}`;
    $('brain-canvas').dataset.activeNeurons = String(p);
    $('brain-canvas').dataset.time = String(s.time);
    this.updateSelection();
    this.draw();
  }

  reset() {
    this.update({time: 0, step_seconds: 0, activity: {indices: [], spike_counts: [], input_indices: []}});
  }

  pick(e) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1, -(e.clientY-rect.top)/rect.height*2+1);
    this.raycaster.params.Points.threshold = this.camera.position.distanceTo(this.controls.target) * .007;
    this.raycaster.setFromCamera(mouse, this.camera);
    // Prefer a firing neuron. A stationary anchor is selectable if none fires
    // near the click. Never treat the selection ring as activity.
    let hits = this.raycaster.intersectObject(this.active);
    let indices = this.activeNeuronIndices;
    if (!hits.length && this.context.visible) {hits = this.raycaster.intersectObject(this.context); indices = this.indices;}
    hits.sort((a,b) => a.distanceToRay-b.distanceToRay);
    if (hits.length) this.select(indices[hits[0].index]);
  }

  async select(index) {
    this.selected = index;
    this.marker.visible = false;
    $('selected-neuron').hidden = index === null;
    $('brain-pick-hint').hidden = index !== null;
    if (index === null) {this.draw(); return;}
    const point = this.pointForNeuron[index];
    if (point >= 0) {
      this.marker.position.fromArray(this.positions, point*3);
      this.marker.visible = true;
    }
    $('neuron-name').textContent = 'Loading neuron…';
    $('neuron-id').textContent = '';
    $('neuron-detail').textContent = '';
    this.updateSelection(); this.draw();
    try {
      const response = await fetch(`/api/brain/neuron/${index}`);
      if (!response.ok) throw new Error('Could not load neuron annotation.');
      const n = await response.json();
      if (this.selected !== index) return;
      $('neuron-name').textContent = n.cell_type || n.super_class || 'Unclassified neuron';
      $('neuron-id').textContent = n.id;
      $('neuron-detail').textContent = [n.side, n.top_nt && `${n.top_nt} annotation`].filter(Boolean).join(' · ') || 'No public cell annotation';
    } catch (error) {if (this.selected === index) $('neuron-name').textContent = error.message;}
  }

  updateSelection() {
    if (this.selected === null) return;
    const count = this.counts[this.selected];
    $('neuron-spikes').textContent = `${count} spike${count === 1 ? '' : 's'} / 20 ms${this.inputIndices.has(this.selected) ? ' · direct input' : ''}`;
  }
}
