"""A small, physical two-baffle maze with sugar and an associated food odor."""
import xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np
from brain.visual_world import VisualWorld
from brain.odor_field import OdorField


class MazeWorld(VisualWorld):
    # x/y center and half-extents, in mm. Corridors comfortably fit the fly.
    WALLS = [(-20,0,.5,20.5),(20,0,.5,20.5),(0,-20,20,.5),(0,20,20,.5),
             (-6,-8.5,.5,11.5),(6,8.5,.5,11.5)]

    def __init__(self):
        self.field = OdorField(self.WALLS)
        self.food_odor = True
        super().__init__()

    def configure_arena(self, root):
        wb, contact = root.find('worldbody'), root.find('contact')
        stripe = wb.find("geom[@name='visual_stripe']")
        stripe.set('rgba', '0 0 0 0')
        templates = []
        for pair in list(contact):
            if 'visual_stripe' in (pair.get('geom1'),pair.get('geom2')):
                templates.append(dict(pair.attrib)); contact.remove(pair)
        if not templates:
            raise ValueError('Maze needs physical fly/wall contact templates')
        for i, (x,y,hx,hy) in enumerate(self.WALLS):
            name = f'maze_wall_{i}'
            ET.SubElement(wb,'geom',name=name,type='box',pos=f'{x} {y} 2',size=f'{hx} {hy} 2',
                          rgba='.10 .15 .21 1',contype='0',conaffinity='0')
            for j, template in enumerate(templates):
                attrs = dict(template); attrs['name'] = f'maze_contact_{i}_{j}'
                key = 'geom1' if attrs['geom1']=='visual_stripe' else 'geom2'
                attrs[key] = name
                ET.SubElement(contact,'pair',attrs)
        # A shallow food patch and pale sugar cubes. Its height stays below
        # the vision task's horizon mask; food is located through the odor cue.
        ET.SubElement(wb,'geom',name='sugar_patch',type='cylinder',pos='0 0 .03',size='1.6 .03',
                      rgba='.92 .63 .19 1',contype='0',conaffinity='0')
        for i,(x,y) in enumerate([(-.45,-.4),(.4,-.1),(0,.5)]):
            ET.SubElement(wb,'geom',name=f'sugar_cube_{i}',type='box',pos=f'{x} {y} .3',size='.38 .38 .28',
                          rgba='.97 .93 .79 1',contype='0',conaffinity='0')

    def reset(self, heading_deg=75, target_deg=30, seed=1):
        # target_deg is accepted for the inherited constructor, but never used
        # as a maze controller input. Sugar always stays at the maze center.
        mj.mj_resetDataKeyframe(self.model,self.data,0)
        yaw = np.deg2rad(heading_deg)
        self.data.qpos[:2] = [-13,-12]
        self.data.qpos[3:7] = [np.cos(yaw/2),0,0,np.sin(yaw/2)]
        self.target = np.array([0.0,0.0])
        self.model.geom_pos[self.stripe_id] = [1000,1000,1000]
        self.phases = np.array([0,np.pi,0,np.pi,0,np.pi])+np.random.default_rng(seed).normal(0,.05,6)
        self.mags = np.zeros(6)
        mj.mj_forward(self.model,self.data)
        self.step([0,0],steps=1500)
        self.data.time = 0.0
        self.path = [self.position[:2].tolist()]

    def smell(self):
        # Virtual antenna sensors fixed to the head in this legs-only body.
        local = np.array([[.65,.38,.08],[.65,-.38,.08]])
        rotation = self.data.xmat[self.thorax].reshape(3,3)
        self.antennae = local @ rotation.T + self.position
        self.odor = self.field.sample(self.antennae) if self.food_odor else np.zeros(2)
        return self.odor.copy()

    def configure_body_camera(self, center):
        self.body_camera.lookat[:] = [0,0,0]
        self.body_camera.distance = 64
        self.body_camera.elevation = -73
        self.body_camera.azimuth = 90

    def map_geometry(self):
        return {'walls': self.WALLS, 'extent_mm': 20.5, 'sugar': [0,0],
                'arrival_radius_mm': 2.5, 'odor_field': self.field.display()}

    def score(self):
        score = super().score()
        wall_ids = [mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_GEOM,f'maze_wall_{i}') for i in range(len(self.WALLS))]
        score['wall_contacts'] = sum(int(c.geom1 in wall_ids or c.geom2 in wall_ids) for c in self.data.contact)
        score['upright'] = bool(self.data.xmat[self.thorax].reshape(3,3)[2,2] > .4)
        return score
