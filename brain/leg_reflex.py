"""A reduced MaleCNS circuit driving a restrained FlyMimic tibia.

This is a causal calibration fixture, not a validated biological reflex model.
No CPG, recorded joint trajectory, or maze policy supplies motor commands.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class ReflexParameters:
    synapse_mv: float = .275
    membrane_tau_s: float = .02
    synaptic_tau_s: float = .005
    delay_s: float = .0018
    refractory_s: float = .0022
    sensory_max_hz: float = 150.
    sensory_span_deg: float = 20.
    motor_filter_s: float = .05
    motor_rate_scale_hz: float = 100.
    muscle_gain: float = .2
    glutamate_sign: int = -1


@dataclass(frozen=True)
class ReflexProtocol:
    rest_angle_deg: float = 100.
    displacement_deg: float = 20.
    onset_s: float = .3
    ramp_s: float = .1
    hold_s: float = .2
    distal_mass_scale: float = 1.

    @property
    def release_s(self):
        return self.onset_s + self.ramp_s + self.hold_s

    def held_angle(self, time_s):
        if time_s >= self.release_s - 1e-12:
            return None
        return self.rest_angle_deg + self.displacement_deg*np.clip(
            (time_s-self.onset_s)/self.ramp_s, 0, 1)


class LegCircuit:
    def __init__(self, graph, parameters=ReflexParameters(), seed=1):
        self.graph, self.parameters = graph, parameters
        self.ids = [n['bodyId'] for n in graph['neurons']]
        self.index = {body: i for i, body in enumerate(self.ids)}
        self.sensory = np.array([self.index[i] for i in graph['sensory_ids']])
        self.flexor = np.array([self.index[i] for i in graph['flexor_ids']])
        self.extensor = np.array([self.index[i] for i in graph['extensor_ids']])
        signs = {'acetylcholine': 1, 'gaba': -1, 'glutamate': parameters.glutamate_sign}
        polarity = {n['bodyId']: signs.get(n['transmitter'], 0) for n in graph['neurons']}
        edges = graph['edges']
        self.weights = csr_matrix((
            [e['weight'] * polarity[e['body_pre']] * parameters.synapse_mv for e in edges],
            ([self.index[e['body_post']] for e in edges],
             [self.index[e['body_pre']] for e in edges])), shape=(len(self.ids), len(self.ids)))
        self.unknown_sign_edges = sum(polarity[e['body_pre']] == 0 for e in edges)
        self.dt = .0001
        self.rng = np.random.default_rng(seed)
        self.voltage = np.full(len(self.ids), -52.)
        self.current = np.zeros(len(self.ids))
        self.refractory = np.zeros(len(self.ids))
        self.queue = np.zeros((round(parameters.delay_s/self.dt), len(self.ids)))
        self.pointer = 0
        self.counts = np.zeros(len(self.ids), dtype=int)
        self.rates = np.zeros(len(self.ids))

    def step(self, angle_deg, rest_angle, *, connected=True, sensory_on=True, motor_on=True):
        """Advance one millisecond using the current physical joint angle."""
        p = self.parameters
        drive = p.sensory_max_hz * np.clip((angle_deg-rest_angle)/p.sensory_span_deg, 0, 1)
        if not sensory_on:
            drive = 0.
        counts = np.zeros(len(self.ids), dtype=int)
        motor = np.concatenate((self.flexor, self.extensor))
        for _ in range(10):
            self.current += self.queue[self.pointer]
            self.queue[self.pointer] = 0
            self.current *= np.exp(-self.dt/p.synaptic_tau_s)
            self.refractory = np.maximum(0, self.refractory-self.dt)
            free = self.refractory <= 0
            self.voltage[free] += self.dt/p.membrane_tau_s * (-52-self.voltage[free]+self.current[free])
            self.voltage[~free] = -52
            # Sensory spikes are the only externally imposed neural events.
            self.voltage[self.sensory] = -52
            fired = (self.voltage >= -45) & free
            fired[self.sensory] = self.rng.random(len(self.sensory)) < drive*self.dt
            if not motor_on:
                fired[motor] = False
                self.voltage[motor] = -52
                self.current[motor] = 0
            self.voltage[fired] = -52
            self.refractory[fired] = p.refractory_s
            if connected:
                self.queue[self.pointer] = self.weights @ fired.astype(float)
            self.pointer = (self.pointer + 1) % len(self.queue)
            counts += fired
        self.counts += counts
        decay = np.exp(-.001/p.motor_filter_s)
        self.rates = decay*self.rates + (1-decay)*counts/.001
        flexor = float(self.rates[self.flexor].mean())
        extensor = float(self.rates[self.extensor].mean())
        activation = np.clip(p.muscle_gain*np.array([flexor, extensor])/p.motor_rate_scale_hz, 0, 1)
        return activation, {'sensory_drive_hz': float(drive), 'sensory_hz': float(self.rates[self.sensory].mean()),
                            'flexor_hz': flexor, 'extensor_hz': extensor,
                            'network_spikes': int(counts.sum())}


class LegFixture:
    def __init__(self, rest_angle=100., distal_mass_scale=1.):
        import mujoco as mj
        from flygym.compose import MusculoskeletalFly
        self.mj = mj
        fly = MusculoskeletalFly()
        fly.mjcf_root.visual.global_.offwidth = 760
        fly.mjcf_root.visual.global_.offheight = 540
        initial, _ = fly.compile()
        pose = initial.key_qpos[0].copy()
        for j in range(initial.njnt):
            name = mj.mj_id2name(initial, mj.mjtObj.mjOBJ_JOINT, j)
            if name != 'joint_LFTibia_pitch':
                q = initial.jnt_qposadr[j]
                fly.mjcf_root.add_equality(name='fixture_'+name, type=mj.mjtEq.mjEQ_JOINT,
                    name1=name, data=[float(pose[q]-initial.qpos0[q])] + [0.]*10,
                    solref=[.002, 1.])
        self.model, self.data = fly.compile()
        m, d = self.model, self.data
        if distal_mass_scale <= 0:
            raise ValueError('Distal mass scale must be positive')
        if distal_mass_scale != 1:
            for body in range(m.nbody):
                name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, body) or ''
                if name.startswith(('LFTibia', 'LFTarsus')):
                    m.body_mass[body] *= distal_mass_scale
                    m.body_inertia[body] *= distal_mass_scale
            mj.mj_setConst(m, d)
        mj.mj_resetDataKeyframe(m, d, 0)
        m.actuator_ctrlrange[:, 0] = 0  # permit genuinely zero excitation controls
        joint = m.joint('joint_LFTibia_pitch').id
        self.q = int(m.jnt_qposadr[joint])
        self.v = int(m.jnt_dofadr[joint])
        self.joint = joint
        self.muscles = [m.actuator(n).id for n in ('LFTibia_flex_93434', 'LFTibia_extensor_93932')]
        self.landmarks = [m.body(n).id for n in ('LFFemur', 'LFTibia', 'LFTarsus1')]
        # Use physical landmark angle, not the differently oriented MJCF qpos.
        grid = np.linspace(*m.jnt_range[joint], 400)
        angles = []
        for qpos in grid:
            d.qpos[self.q] = qpos
            mj.mj_forward(m, d)
            angles.append(self.angle())
        self.q_grid = grid
        self.angle_grid = np.asarray(angles)
        self.rest_angle = rest_angle
        self.hold(rest_angle)
        mj.mj_forward(m, d)
        # A constant external fixture counter-torque balances the resting pose.
        # It is identical in every paired trial; it is not a neural command.
        self.counter_torque = float(d.qfrc_bias[self.v] - d.qfrc_passive[self.v] - d.qfrc_actuator[self.v])
        self.pose = pose
        self.renderer = None

    def angle(self):
        a, b, c = self.data.xpos[self.landmarks]
        u, v = a-b, c-b
        return float(np.degrees(np.arccos(np.clip(np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v)), -1, 1))))

    def hold(self, angle):
        if not self.angle_grid.min() <= angle <= self.angle_grid.max():
            raise ValueError('Requested angle lies outside this body model range')
        order = np.argsort(self.angle_grid)
        self.data.qpos[self.q] = np.interp(angle, self.angle_grid[order], self.q_grid[order])
        self.data.qvel[self.v] = 0
        self.mj.mj_forward(self.model, self.data)

    def step(self, activation, held_angle=None):
        d, m = self.data, self.model
        d.ctrl[:] = 0
        d.ctrl[self.muscles] = activation
        d.qfrc_applied[self.v] = self.counter_torque
        for _ in range(10):
            if held_angle is not None:
                self.hold(held_angle)
            self.mj.mj_step(m, d)
        if held_angle is not None:
            self.hold(held_angle)
        self.mj.mj_forward(m, d)

    def image(self):
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, 540, 760)
        cam = self.mj.MjvCamera()
        cam.lookat[:] = self.data.xpos[self.model.body('LFTibia').id]
        cam.distance = 2.4
        cam.azimuth = 110
        cam.elevation = -25
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def run_reflex(graph, condition='connected', *, seed=1, duration=1.6, parameters=ReflexParameters(), video=None, protocol=ReflexProtocol()):
    if condition not in ('connected', 'disconnected', 'sensory_off', 'motor_silenced'):
        raise ValueError('Unknown reflex condition')
    if protocol.ramp_s <= 0 or protocol.onset_s < 0 or protocol.hold_s < 0 or duration <= protocol.release_s:
        raise ValueError('Protocol needs a positive ramp and a post-release interval')
    fixture = LegFixture(protocol.rest_angle_deg, protocol.distal_mass_scale)
    brain = LegCircuit(graph, parameters, seed)
    trace, images = [], []
    try:
        for step in range(round(duration/.001)):
            t = step*.001
            # A physical perturbation, explicitly separate from the neural loop.
            held = protocol.held_angle(t)
            if held is not None:
                fixture.hold(held)
            activation, neural = brain.step(fixture.angle(), fixture.rest_angle,
                connected=condition != 'disconnected', sensory_on=condition != 'sensory_off',
                motor_on=condition != 'motor_silenced')
            fixture.step(activation, held)
            if step % 5 == 0:
                trace.append({'time_s': float(fixture.data.time), 'angle_deg': fixture.angle(),
                              'held': held is not None, 'muscle_activation': activation.tolist(),
                              'muscle_force_model_units': fixture.data.actuator_force[fixture.muscles].tolist(),
                              **neural})
            if video and step % 20 == 0:
                images.append(fixture.image().copy())
        if video:
            import imageio.v2 as imageio
            # 50 simulated frames/s played at 10 fps for a fivefold slowed replay.
            imageio.mimsave(str(video), images, fps=10, macro_block_size=2)
        return {'condition': condition, 'seed': seed, 'duration_s': duration,
                'parameters': asdict(parameters), 'protocol': asdict(protocol), 'trace': trace,
                'spikes_by_body_id': dict(zip(map(str, brain.ids), map(int, brain.counts))),
                'unknown_sign_edges_disabled': brain.unknown_sign_edges,
                'fixture_counter_torque': fixture.counter_torque,
                'final_angle_deg': fixture.angle(),
                'total_spikes': int(brain.counts.sum()),
                'motor_spikes': int(brain.counts[np.r_[brain.flexor, brain.extensor]].sum())}
    finally:
        fixture.close()
