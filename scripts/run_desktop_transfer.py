"""Test the DesktopFly nerve-cord model on our restrained MuJoCo tibia.

The six-leg browser reference is unchanged. This is a separately measured
transfer with two existing Hill-type muscles, not a six-leg MuJoCo walker.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from brain.desktop_locomotor import DesktopLocomotor, GRAPH_PATH
from brain.leg_reflex import LegFixture


def check_port(reference_path):
    reference = json.loads(reference_path.read_text())
    brain = DesktopLocomotor()
    for side in ('left', 'right'):
        brain.set_descending('DNp09', side, 40)
    leg = dict(hipAngle=0., hipVelocity=0., kneeAngle=.95, kneeVelocity=4.,
               contact=False, load=0., elevationVelocity=0.)
    brain.feedback({1: leg})
    brain.step(2000)
    errors = {}
    for source, target in [('voltage', 'voltage'), ('rates', 'rates'), ('adaptation', 'adaptation'),
                           ('excitatory', 'exc'), ('inhibitory', 'inh'), ('refractory', 'refractory')]:
        actual = getattr(brain, target)
        errors[source] = float(np.max(np.abs(actual - reference[source])))
        np.testing.assert_allclose(actual, reference[source], rtol=0, atol=1e-10)
    assert int(brain.counts.sum()) == reference['total_spikes']
    assert int(brain.counts[brain.motor].sum()) == reference['motor_spikes']
    assert int(brain.counts[brain.sensory].sum()) == reference['sensory_spikes']
    return {'passed': True, 'max_errors': errors, 'counts_match': True,
            'reference': 'Unmodified upstream JavaScript, 2 s bilateral DNp09 and fixed LF knee velocity.'}


def trial(condition, directory, duration=4., muscle_gain=.2, video=False):
    fixture = LegFixture()
    brain = DesktopLocomotor()
    for side in ('left', 'right'):
        brain.set_descending('DNp09', side, 40)
    brain.synapses_enabled = condition != 'synapses_cut'
    brain.feedback_enabled = condition != 'feedback_off'
    if condition == 'motor_silenced':
        brain.silenced[brain.motor] = True
    trace = []
    previous_angle = fixture.angle()
    writer = None
    started = time.monotonic()
    if video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(directory / (condition + '.mp4')), fps=10, macro_block_size=2)
    try:
        for step in range(round(duration * 1000)):
            angle = fixture.angle()
            # DesktopFly knee flexion is positive. The fixture reports the
            # physical femur–tibia interior angle, which decreases in flexion.
            # Hip and elevation are restrained. No contact/load is invented.
            feedback = dict(hipAngle=0., hipVelocity=0.,
                kneeAngle=.95 + np.deg2rad(fixture.rest_angle - angle),
                kneeVelocity=np.deg2rad(previous_angle - angle) / .001,
                contact=False, load=0., elevationVelocity=0.)
            brain.feedback({1: feedback})
            brain.step()
            activation = muscle_gain * brain.tibia_activation()
            fixture.step(activation)
            previous_angle = angle
            if step % 5 == 0:
                trace.append({'time_s': float(fixture.data.time), 'angle_deg': fixture.angle(),
                    'activation': activation.tolist(),
                    'flexor_hz': brain.channel_rate(1, 'tibia_flexor'),
                    'extensor_hz': brain.channel_rate(1, 'tibia_extensor'),
                    'muscle_force_model_units': fixture.data.actuator_force[fixture.muscles].tolist()})
            if writer and step % 20 == 0:
                writer.append_data(fixture.image())
        assert np.isfinite(fixture.data.qpos).all() and np.isfinite(fixture.data.qvel).all()
        assert abs(brain.sim_ms / 1000 - fixture.data.time) < 1e-8
        lf_flex = brain.motor_groups[1, 'tibia_flexor']
        lf_ext = brain.motor_groups[1, 'tibia_extensor']
        result = {'condition': condition, 'duration_s': duration, 'muscle_gain': muscle_gain,
            'descending_drive_hz_equivalent': 40, 'runtime_seconds': time.monotonic() - started,
            'total_spikes': int(brain.counts.sum()), 'motor_spikes': int(brain.counts[brain.motor].sum()),
            'sensory_spikes': int(brain.counts[brain.sensory].sum()),
            'lf_flexor_spikes': int(brain.counts[lf_flex].sum()),
            'lf_extensor_spikes': int(brain.counts[lf_ext].sum()),
            'trace': trace, 'final_angle_deg': fixture.angle(),
            'spikes_by_body_id': {n['id']: int(brain.counts[i]) for i, n in enumerate(brain.neurons)}}
        (directory / (condition + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        return result
    finally:
        if writer:
            writer.close()
        fixture.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/desktop-locomotor/transfer')
    parser.add_argument('--reference', type=Path, default=ROOT / 'outputs/desktop-locomotor/neural-reference.json')
    parser.add_argument('--videos', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    port = check_port(args.reference)
    print(json.dumps({'port': port}), flush=True)
    trials = {}
    for condition in ('connected', 'motor_silenced', 'synapses_cut', 'feedback_off'):
        trials[condition] = trial(condition, args.output,
            video=args.videos and condition in ('connected', 'motor_silenced'))
        print(json.dumps({k: v for k, v in trials[condition].items()
                          if k not in ('trace', 'spikes_by_body_id')}), flush=True)
    active = trials['connected']
    assert active['motor_spikes'] > 0
    passive = np.array([r['angle_deg'] for r in trials['motor_silenced']['trace']])
    for condition in ('motor_silenced', 'synapses_cut'):
        assert trials[condition]['motor_spikes'] == 0
        np.testing.assert_allclose([r['angle_deg'] for r in trials[condition]['trace']], passive, atol=1e-9, rtol=0)
    assert trials['feedback_off']['sensory_spikes'] == 0
    angle = np.array([r['angle_deg'] for r in active['trace']])
    delta = angle - passive
    effect = float(np.max(np.abs(delta)))
    assert effect > .01, 'No measurable mechanical effect under the fixed transfer assumptions'
    report = {'passed': True, 'host': platform.node(), 'python': platform.python_version(),
        'graph_sha256': hashlib.sha256(GRAPH_PATH.read_bytes()).hexdigest(), 'port_validation': port,
        'duration_s': 4, 'muscle_gain': .2, 'max_added_angle_deg': effect,
        'max_added_flexion_deg': float(np.max(passive - angle)),
        'max_added_extension_deg': float(np.max(angle - passive)),
        'lf_flexor_spikes': active['lf_flexor_spikes'], 'lf_extensor_spikes': active['lf_extensor_spikes'],
        'trials': {k: {field: value for field, value in v.items() if field not in ('trace', 'spikes_by_body_id')}
                   for k, v in trials.items()},
        'six_leg_mujoco_locomotion_validated': False, 'biological_calibration': False,
        'summary': 'The reduced circuit drives our restrained MuJoCo left-front tibia. Over 4 s, the connected leg differs from the motor-silenced control by up to '
            + f'{effect:.2f}°. Its left-front motor pools emit {active["lf_flexor_spikes"]} flexor and {active["lf_extensor_spikes"]} extensor spikes. '
            + 'Synaptic disconnection and motor silencing remove the neural movement. This establishes a one-joint transfer under a fixed 0.2 muscle gain; it does not establish six-leg MuJoCo walking or physiological calibration.'}
    if args.videos:
        report['videos'] = {'connected': 'transfer-connected.mp4', 'motor_silenced': 'transfer-motor-silenced.mp4'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
