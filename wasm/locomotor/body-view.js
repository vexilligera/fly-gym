import * as THREE from '../shared/vendor/three/three.module.js';
import { OrbitControls } from '../shared/vendor/three/OrbitControls.js';
import { GEOMETRIES, legPoints } from './loop.js';

export class BodyView {
  constructor(element) {
    this.element = element;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    element.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#eef0e8');
    this.scene.fog = new THREE.Fog('#eef0e8', 140, 330);
    this.camera = new THREE.PerspectiveCamera(38, 1, .1, 600);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 28;
    this.controls.maxDistance = 180;
    this.controls.maxPolarAngle = Math.PI * .49;
    this.scene.add(new THREE.HemisphereLight('#ffffff', '#89907b', 2.4));
    const light = new THREE.DirectionalLight('#fff4da', 3.5);
    light.position.set(-25, 15, 80);
    light.castShadow = true;
    light.shadow.mapSize.set(2048, 2048);
    Object.assign(light.shadow.camera, { left: -100, right: 100, top: 100, bottom: -100, near: 1, far: 200 });
    light.shadow.bias = -.001;
    this.scene.add(light);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(600, 600),
      new THREE.MeshStandardMaterial({ color: '#e9eddf', roughness: 1 }));
    ground.position.z = -.05; ground.receiveShadow = true; this.scene.add(ground);
    const grid = new THREE.GridHelper(600, 120, '#bcc6b3', '#d1d9c9');
    grid.rotation.x = Math.PI / 2; grid.position.z = 0; this.scene.add(grid);
    this.fly = new THREE.Group(); this.scene.add(this.fly);
    const body = new THREE.MeshStandardMaterial({ color: '#7a572e', roughness: .8 });
    const dark = new THREE.MeshStandardMaterial({ color: '#493e29', roughness: .8 });
    const eye = new THREE.MeshStandardMaterial({ color: '#9e3e25', roughness: .5 });
    const sphere = (pos, scale, material) => {
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 20), material);
      mesh.position.set(...pos); mesh.scale.set(...scale); mesh.castShadow = true;
      this.fly.add(mesh); return mesh;
    };
    sphere([0, 2.5, 6.8], [4.3, 5.8, 4], body);
    sphere([0, -6.2, 5.8], [3.9, 7.8, 3.2], dark);
    sphere([0, 9, 6.7], [3, 2.8, 2.7], body);
    sphere([-2.25, 9.7, 7], [1.5, 2, 2.15], eye);
    sphere([2.25, 9.7, 7], [1.5, 2, 2.15], eye);
    const wingMaterial = new THREE.MeshPhysicalMaterial({ color: '#d4dcca', transparent: true,
      opacity: .44, roughness: .35, side: THREE.DoubleSide, depthWrite: false });
    for (const side of [-1, 1]) {
      const wing = sphere([side * 1.9, -6, 10.35], [2.5, 8.7, .12], wingMaterial);
      wing.rotation.z = side * .15; wing.castShadow = false;
      this.segment([side * .8, 11, 6.8], [side * 1.4, 13.4, 7.7], .13, dark, this.fly);
    }
    this.segments = GEOMETRIES.map(() => [0, 1, 2].map((i) =>
      this.segment([0, 0, 0], [1, 0, 0], [.36, .25, .15][i], body, this.fly)));
    this.feet = GEOMETRIES.map(() => {
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(.48, 10, 8),
        new THREE.MeshStandardMaterial({ color: '#3b8867', transparent: true, opacity: .85 }));
      this.fly.add(mesh); return mesh;
    });
    const lineGeometry = new THREE.BufferGeometry();
    this.trail = new THREE.Line(lineGeometry, new THREE.LineBasicMaterial({ color: '#ad7441', transparent: true, opacity: .65 }));
    this.scene.add(this.trail);
    this.lastPosition = new THREE.Vector3();
    this.resetCamera();
    new ResizeObserver(() => this.resize()).observe(element);
    this.resize();
  }
  segment(a, b, radius, material, parent) {
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 1, 10), material);
    mesh.castShadow = true; parent.add(mesh); this.placeSegment(mesh, a, b); return mesh;
  }
  placeSegment(mesh, a, b) {
    const start = new THREE.Vector3(...a), end = new THREE.Vector3(...b);
    const delta = end.clone().sub(start);
    mesh.position.copy(start.add(end).multiplyScalar(.5));
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.clone().normalize());
    mesh.scale.y = delta.length();
  }
  resetCamera() {
    const p = this.fly.position;
    this.camera.position.set(p.x + 45, p.y + 58, 58);
    this.controls.target.set(p.x, p.y, 4);
    this.controls.update();
  }
  resize() {
    const w = this.element.clientWidth, h = this.element.clientHeight;
    this.renderer.setSize(w, h); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }
  update(state, follow = true) {
    const next = new THREE.Vector3(...state.position, 0);
    if (follow) {
      const delta = next.clone().sub(this.lastPosition);
      this.camera.position.add(delta); this.controls.target.add(delta);
    }
    this.lastPosition.copy(next);
    this.fly.position.copy(next); this.fly.rotation.z = state.yaw;
    state.feedback.forEach((f, i) => {
      const points = legPoints(GEOMETRIES[i], f);
      this.segments[i].forEach((mesh, j) => this.placeSegment(mesh, points[j], points[j + 1]));
      this.feet[i].position.set(...points[3]); this.feet[i].visible = f.contact;
    });
    this.trail.geometry.dispose();
    this.trail.geometry = new THREE.BufferGeometry().setFromPoints(
      state.path.map((p) => new THREE.Vector3(p[0], p[1], .08)));
  }
  render() { this.controls.update(); this.renderer.render(this.scene, this.camera); }
}
