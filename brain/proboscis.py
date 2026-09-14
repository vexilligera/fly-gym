"""MN9-driven, engineered proboscis servos in a posed copy of the maze.

Native rostrum/haustellum meshes and masses; chosen hinges and transfer gains.
Only the mouth is dynamic. This is not a calibrated muscle or ingestion model.
"""
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation


class ProboscisDrive:
    def __init__(self):
        self.rates = np.zeros(2)
        self.targets = np.zeros(3)

    def step(self, mn9_hz, dt=.02):
        rates = np.array([mn9_hz['left'], mn9_hz['right']], dtype=float)
        if not np.isfinite(rates).all() or (rates < 0).any():
            raise ValueError('MN9 rates must be finite and nonnegative')
        self.rates += (1-np.exp(-dt/.12)) * (rates-self.rates)
        extension = np.clip(self.rates.mean()/100, 0, 1)
        # Stronger contralateral MN9 produces a leftward illustration for the
        # released left-GRN cohort. This is a declared, uncalibrated yaw rule.
        turn = (self.rates[1]-self.rates[0])/(self.rates.sum()+20)
        self.targets = np.deg2rad([25*turn*extension, -100*extension, 70*extension])
        return self.targets.copy()


class ProboscisWorld:
    def __init__(self, world):
        root = ET.fromstring(world.model_xml)
        # Freeze the *measured arrival pose*, including every leg articulation.
        # A separate model leaves walking dynamics and their random stream intact.
        for tag in ('actuator', 'sensor', 'tendon', 'equality', 'contact', 'keyframe', 'size'):
            for item in list(root.findall(tag)):
                root.remove(item)
        for body in root.findall('.//body'):
            idx = mj.mj_name2id(world.model, mj.mjtObj.mjOBJ_BODY, body.get('name'))
            parent = world.model.body_parentid[idx]
            rotation = world.data.xmat[parent].reshape(3, 3)
            pos = rotation.T @ (world.data.xpos[idx]-world.data.xpos[parent])
            relative = rotation.T @ world.data.xmat[idx].reshape(3, 3)
            body.set('pos', ' '.join(map(str, pos)))
            body.set('quat', ' '.join(map(str, Rotation.from_matrix(relative).as_quat(scalar_first=True))))
            for joint in list(body.findall('joint')) + list(body.findall('freejoint')):
                body.remove(joint)
        root.find('option').set('integrator', 'implicitfast')
        actuator = ET.SubElement(root, 'actuator')
        self.joint_names = ('mouth_yaw', 'rostrum_pitch', 'haustellum_pitch')
        specs = [('c_rostrum', '0 0 1', '-.45 .45'),
                 ('c_rostrum', '0 1 0', '-1.8 .02'),
                 ('c_haustellum', '0 1 0', '-.02 1.3')]
        for name, (segment, axis, limits) in zip(self.joint_names, specs):
            body = root.find(f".//body[@name='nmf/{segment}']")
            body.set('gravcomp', '1')
            ET.SubElement(body, 'joint', name=name, type='hinge', axis=axis,
                          range=limits, limited='true', armature='1e-6', damping='.0002')
            ET.SubElement(actuator, 'position', name=name, joint=name, kp='.04', kv='.001',
                          ctrlrange=limits, ctrllimited='true')
        self.model = mj.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mj.MjData(self.model)
        self.drive = ProboscisDrive()
        mj.mj_forward(self.model, self.data)
        self.renderer = mj.Renderer(self.model, height=600, width=800)
        self.options = mj.MjvOption()
        # Observer cutaway: an adjacent tall maze wall must not hide the mouth.
        for i in range(self.model.ngeom):
            if (mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i) or '').startswith('maze_wall_'):
                self.model.geom_group[i] = 4
        self.options.geomgroup[4] = 0
        self.camera = mj.MjvCamera()
        self.camera.lookat[:] = world.position + world.data.xmat[world.thorax].reshape(3, 3) @ np.array([.45, 0, -.2])
        self.camera.distance = 4.0
        self.camera.elevation = -22
        self.camera.azimuth = np.rad2deg(world.heading)+115
        self.jpeg = world.jpeg

    def step(self, mn9_hz):
        self.data.ctrl[:] = self.drive.step(mn9_hz)
        mj.mj_step(self.model, self.data, nstep=200)
        mj.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError('Non-finite proboscis state')
        return self.snapshot()

    def snapshot(self):
        return {'time': float(self.data.time),
                'angles_deg': dict(zip(self.joint_names, np.rad2deg(self.data.qpos).tolist())),
                'targets_deg': dict(zip(self.joint_names, np.rad2deg(self.drive.targets).tolist())),
                'filtered_MN9_hz': self.drive.rates.tolist(),
                'mapping': 'Engineered MN9-to-servo mapping; yaw and haustellum coupling are uncalibrated',
                'ingestion': False}

    def image(self):
        self.renderer.update_scene(self.data, self.camera, scene_option=self.options)
        return self.jpeg(self.renderer.render(), caption=f'Mouth assay {self.data.time:.2f} s')

    def close(self):
        self.renderer.close()
