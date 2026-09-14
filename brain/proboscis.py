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
    # Approximate distal labellum location in the native haustellum mesh (mm).
    LABELLUM = np.array([.3184, 0, -.1411])

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
        # A declared feeding preparation, separate from the recorded arrival:
        # face the source and place the labellum at the edge of sugar solution.
        # Keep height and all relative leg articulations, so feet stay grounded.
        old_haustellum = mj.mj_name2id(world.model, mj.mjtObj.mjOBJ_BODY, 'nmf/c_haustellum')
        old_tip = world.data.xpos[old_haustellum] + world.data.xmat[old_haustellum].reshape(3, 3) @ self.LABELLUM
        yaw_rotation = Rotation.from_euler('z', -world.heading).as_matrix()
        target_tip = np.array([-1.35, 0, old_tip[2]])
        posed_position = target_tip - yaw_rotation @ (old_tip-world.position)
        posed_rotation = yaw_rotation @ world.data.xmat[world.thorax].reshape(3, 3)
        thorax = root.find(".//body[@name='nmf/c_thorax']")
        thorax.set('pos', ' '.join(map(str, posed_position)))
        thorax.set('quat', ' '.join(map(str, Rotation.from_matrix(posed_rotation).as_quat(scalar_first=True))))
        self.posed_position = posed_position
        self.food_radii = np.array([1.6, 1.6, .45])
        self.food_center = np.array([0, 0, target_tip[2]-.45*np.sqrt(1-(1.35/1.6)**2)+.015])
        wb = root.find('worldbody')
        for geom in list(wb.findall('geom')):
            if geom.get('name', '').startswith('sugar_cube_'):
                wb.remove(geom)
        ET.SubElement(wb, 'geom', name='sugar_solution', type='ellipsoid',
                      pos=' '.join(map(str, self.food_center)), size='1.6 1.6 .45',
                      rgba='.95 .66 .15 .65', contype='0', conaffinity='0')
        haustellum = root.find(".//body[@name='nmf/c_haustellum']")
        ET.SubElement(haustellum, 'site', name='labellum_contact',
                      pos=' '.join(map(str, self.LABELLUM)), size='.015', rgba='0 0 0 0')
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
        self.tip_site = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, 'labellum_contact')
        self.food_geom = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, 'sugar_solution')
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
        self.camera.lookat[:] = posed_position + posed_rotation @ np.array([.65, 0, -.22])
        self.camera.distance = 4.6
        self.camera.elevation = -20
        self.camera.azimuth = 110
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
                'contact': self.contact(),
                'placement': 'Staged feeding pose facing the sugar solution; navigation arrival is retained separately',
                'position_mm': self.posed_position.tolist(),
                'mapping': 'Engineered MN9-to-servo mapping; yaw and haustellum coupling are uncalibrated',
                'ingestion': False}

    def contact(self):
        tip = self.data.site_xpos[self.tip_site]
        center = self.data.geom_xpos[self.food_geom]
        inside = float(np.sum(((tip-center)/self.model.geom_size[self.food_geom])**2))
        return {'touching': inside <= 1, 'labellum_mm': tip.tolist(),
                'normalized_surface_offset': inside-1,
                'sensor': 'Approximate labellum point within the sugar-solution ellipsoid'}

    def image(self):
        self.renderer.update_scene(self.data, self.camera, scene_option=self.options)
        return self.jpeg(self.renderer.render(), caption=f'Mouth assay {self.data.time:.2f} s')

    def close(self):
        self.renderer.close()
