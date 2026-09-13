"""Server-side NeuroMechFly physics, official compound eyes, and a dark stripe.

Uses the same exported MJCF and recorded-step CPG as the existing browser fly.
Coordinates of the stripe are used for rendering and scoring only. Motor
commands come from the neural decoder, never a world-coordinate bearing.
"""
from pathlib import Path
import base64
import io
import json
import xml.etree.ElementTree as ET

import mujoco as mj
import numba as nb
import numpy as np
from PIL import Image
import yaml
from scipy.spatial.transform import Rotation
from flygym import assets_dir
from flygym.vision.retina import Retina

ROOT = Path(__file__).resolve().parents[1]


@nb.njit(cache=True)
def cpg_controls(phases, mags, gains, freqs, coupling, biases, conv, tables, neutral, swing, cmap, adhesion, dt, steps):
    """Batch the existing Euler CPG and periodic recorded-step interpolation."""
    controls = np.zeros((steps, 48))
    for step in range(steps):
        dph = np.zeros(6)
        for i in range(6):
            sign = 1 if gains[i // 3] > 0 else -1
            dph[i] = 2 * np.pi * freqs[i] * sign
            for j in range(6):
                dph[i] += mags[j] * coupling[i, j] * np.sin(phases[j] - phases[i] - biases[i, j])
        phases += dph * dt
        for i in range(6):
            mags[i] += conv[i] * (abs(gains[i // 3]) - mags[i]) * dt
            phase = phases[i] % (2 * np.pi)
            x = phase / (2 * np.pi) * tables.shape[1]
            i0, fraction = int(x), x - int(x)
            for d in range(7):
                a = tables[i, i0, d] * (1 - fraction) + tables[i, (i0 + 1) % tables.shape[1], d] * fraction
                controls[step, cmap[i, d]] = neutral[i, d] + mags[i] * (a - neutral[i, d])
            controls[step, adhesion[i]] = not (swing[i, 0] < phase < swing[i, 1])
    return controls


class VisualWorld:
    def __init__(self):
        model_dir = ROOT / 'wasm/game/assets/model'
        tree = ET.parse(model_dir / 'fly.xml')
        root, wb = tree.getroot(), tree.find('worldbody')
        removed = set()
        for geom in list(wb.findall('geom')):
            if geom.get('name', '').startswith(('start_pole', 'gate')):
                removed.add(geom.get('name'))
                wb.remove(geom)
        contact = root.find('contact')
        target_contacts = []
        for pair in list(contact):
            if pair.get('geom2') == 'start_pole_left':
                attrs = dict(pair.attrib)
                attrs.update(name='stripe_' + attrs['name'], geom2='visual_stripe')
                target_contacts.append(attrs)
            if pair.get('geom1') in removed or pair.get('geom2') in removed:
                contact.remove(pair)
        for attrs in target_contacts:
            ET.SubElement(contact, 'pair', attrs)
        self.stripe_name = 'visual_stripe'
        ET.SubElement(wb, 'geom', name=self.stripe_name, type='cylinder', size='.65 4',
                      pos='18 0 4', rgba='.015 .015 .015 1', contype='0', conaffinity='0')
        # A uniformly light arena makes the isolated vertical-target task
        # unambiguous. The controller still receives only rendered pixels.
        for geom in wb.findall('geom'):
            if geom.get('type') == 'plane':
                geom.set('rgba', '1 1 1 1')
        floor_texture = root.find("asset/texture[@name='checker']")
        floor_texture.set('rgb1', '.82 .85 .86')
        floor_texture.set('rgb2', '.9 .93 .94')
        root.find("asset/material[@name='grid']").set('texrepeat', '50 50')
        for mesh in root.findall('asset/mesh'):
            mesh.set('file', str(model_dir / mesh.get('file')))
        config = yaml.safe_load((assets_dir / 'model/neuromechfly/vision.yaml').read_text())
        for name, sensor in config['sensors'].items():
            parent = root.find(f".//body[@name='nmf/{sensor['parent']}']")
            # The flattened asset omits the source fly's extrinsic XYZ Euler
            # convention. Bake its orientation as a quaternion explicitly.
            quat = Rotation.from_euler('xyz', sensor['orientation']).as_quat(scalar_first=True)
            ET.SubElement(parent, 'camera', name=name, mode='fixed',
                          pos=' '.join(map(str, sensor['rel_pos'])),
                          quat=' '.join(map(str, quat)),
                          fovy=str(config['fovy_per_eye']))
        self.model = mj.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mj.MjData(self.model)
        self.stripe_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, self.stripe_name)
        self.thorax = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'nmf/c_thorax')
        self.cameras = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, name) for name in config['sensors']]
        self.retina = Retina()
        self.eye_renderer = mj.Renderer(self.model, height=512, width=450)
        self.body_renderer = mj.Renderer(self.model, height=480, width=640)
        self.eye_options = mj.MjvOption()
        self.eye_options.geomgroup[1:3] = 0
        # The entire fly is hidden from the eyes for this first stripe task,
        # avoiding self-leg contrast. This does not change contact physics.
        fly_geoms = self.model.geom_bodyid != 0
        self.model.geom_group[fly_geoms & (self.model.geom_group != 3)] = 2
        self.body_camera = mj.MjvCamera()
        self.body_camera.elevation = -38
        self.body_camera.azimuth = 130
        self.body_camera.distance = 24
        self.meta = json.loads((ROOT / 'wasm/game/assets/model_meta.json').read_text())
        cpg = self.meta['control']['cpg']
        pp = [self.meta['preprogrammed']['legs'][leg] for leg in self.meta['control']['leg_order']]
        self.cpg_args = [np.array(cpg[key]) for key in ('intrinsic_freqs', 'coupling_weights', 'phase_biases', 'convergence_coefs')]
        self.cpg_args += [np.array([p[key] for p in pp]) for key in ('angles', 'neutral', 'swing')]
        self.cpg_args += [np.array(self.meta['ctrl_index_by_leg_dof']), np.array(self.meta['adhesion']), self.meta['timestep']]
        self.reset()
        self._retinal_rays(config['fovy_per_eye'])

    def _retinal_rays(self, fovy):
        """Derive pixel bearings from CAMERA geometry, never neuron anchors.

        Apply the official fisheye inverse sampling equation at each hex center.
        Coordinates are fly-body relative and remain fixed as the animal turns.
        """
        ids = self.retina.ommatidia_id_map
        row, col = np.indices(ids.shape)
        centers = np.array([[row[ids == i].mean(), col[ids == i].mean()] for i in range(1, 722)])
        y = (2 * centers[:, 0] - 512) / 512 / self.retina.zoom
        x = (2 * centers[:, 1] - 450) / 450 / self.retina.zoom
        denom = 1 - self.retina.distortion_coefficient * (x*x + y*y) + 1e-6
        xx, yy = x / denom, y / denom
        tan = np.tan(np.deg2rad(fovy) / 2)
        rays = np.stack([xx * tan * 450/512, -yy * tan, -np.ones(721)], axis=1)
        body_rot = self.data.xmat[self.thorax].reshape(3, 3)
        self.bearings, self.elevations = [], []
        for cam in self.cameras:
            relative_rotation = body_rot.T @ self.data.cam_xmat[cam].reshape(3, 3)
            transformed = rays @ relative_rotation.T
            self.bearings.append(np.arctan2(transformed[:, 1], transformed[:, 0]))
            self.elevations.append(np.arctan2(transformed[:, 2], np.linalg.norm(transformed[:, :2], axis=1)))
        self.bearings, self.elevations = np.array(self.bearings), np.array(self.elevations)
        valid = (np.abs(xx) < 1) & (np.abs(yy) < 1) & (denom > 0)
        # Look above the ground and below the top of the tall stripe. The
        # selection is fixed in retinal coordinates, not target coordinates.
        self.visual_mask = valid[None, :] & (self.elevations > .03) & (self.elevations < .65) & (np.abs(self.bearings) < 1.9)

    def reset(self, heading_deg=0, target_deg=30, seed=1):
        mj.mj_resetDataKeyframe(self.model, self.data, 0)
        yaw = np.deg2rad(heading_deg)
        self.data.qpos[:2] = 0, 0
        self.data.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        angle = np.deg2rad(target_deg)
        self.target = np.array([18*np.cos(angle), 18*np.sin(angle)])
        self.model.geom_pos[self.stripe_id, :2] = self.target
        self.phases = np.array([0, np.pi, 0, np.pi, 0, np.pi]) + np.random.default_rng(seed).normal(0, .05, 6)
        self.mags = np.zeros(6)
        mj.mj_forward(self.model, self.data)
        self.step([0, 0], steps=1500)
        self.data.time = 0.0
        self.path = [self.position[:2].tolist()]

    @property
    def position(self):
        return self.data.xpos[self.thorax].copy()

    @property
    def heading(self):
        mat = self.data.xmat[self.thorax].reshape(3, 3)
        return float(np.arctan2(mat[1, 0], mat[0, 0]))

    def step(self, gains, steps=200):
        actions = cpg_controls(self.phases, self.mags, np.asarray(gains), *self.cpg_args, steps)
        for action in actions:
            self.data.ctrl[:] = action
            mj.mj_step(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError('Non-finite body state')

    def vision(self):
        frames, values = [], []
        for camera in self.cameras:
            self.eye_renderer.update_scene(self.data, camera, scene_option=self.eye_options)
            frame = self.retina.correct_fisheye(self.eye_renderer.render())
            frames.append(frame)
            values.append(self.retina.raw_image_to_hex_pxls(frame).sum(axis=1))
        self.eye_frames, self.readings = np.array(frames), np.array(values)
        # Dark-contrast event encoding; not a phototransduction model.
        self.contrast = np.clip((.65 - self.readings) / .65, 0, 1) * self.visual_mask
        return self.contrast

    @staticmethod
    def jpeg(frame, size=None):
        im = Image.fromarray(frame)
        if size:
            im = im.resize(size)
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=78)
        return base64.b64encode(buf.getvalue()).decode('ascii')

    def images(self):
        center = self.position
        self.body_camera.lookat[:] = [(center[0] + self.target[0])/2, (center[1] + self.target[1])/2, 1.5]
        self.body_camera.distance = max(20.0, np.linalg.norm(center[:2] - self.target) * 1.35)
        self.body_renderer.update_scene(self.data, self.body_camera)
        eyes = []
        for values in self.readings:
            # Faithful 721-channel mosaic; zero outside the retinal lattice.
            mosaic = np.zeros((512, 450), dtype=np.uint8)
            ids = self.retina.ommatidia_id_map
            mask = ids > 0
            mosaic[mask] = np.clip(values[ids[mask]-1] * 255, 0, 255).astype(np.uint8)
            eyes.append(self.jpeg(mosaic, (225, 256)))
        return {'body': self.jpeg(self.body_renderer.render()), 'eyes': eyes}

    def score(self):
        offset = self.target - self.position[:2]
        error = np.arctan2(offset[1], offset[0]) - self.heading
        return {'distance_mm': float(np.linalg.norm(offset)),
                'heading_error_deg': float(np.rad2deg(np.arctan2(np.sin(error), np.cos(error))))}

    def close(self):
        self.eye_renderer.close()
        self.body_renderer.close()
