"""Check actuated mouth dynamics and isolation from the walking world."""
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.proboscis import ProboscisDrive
from brain.maze_world import MazeWorld


def main():
    left, right = ProboscisDrive(), ProboscisDrive()
    for _ in range(100):
        a = left.step({'left': 20, 'right': 100})
        b = right.step({'left': 100, 'right': 20})
    np.testing.assert_allclose(a, b*[-1, 1, 1], atol=1e-12)
    assert a[0] > 0
    symmetric = ProboscisDrive().step({'left': 100, 'right': 100})
    assert symmetric[0] == 0
    saturated = ProboscisDrive()
    for _ in range(100):
        targets = saturated.step({'left': 0, 'right': 1e6})
    assert np.abs(targets[0]) <= np.deg2rad(25)
    assert abs(targets[1]) <= np.deg2rad(100)

    world = MazeWorld('simple')
    try:
        before = (world.data.qpos.copy(), world.data.qvel.copy(), world.data.time,
                  world.data.xpos.copy(), world.data.xquat.copy(), world.path.copy())
        world.start_feeding()
        mouth = world.feeding
        assert mouth.model.nq == mouth.model.nu == 3
        np.testing.assert_allclose(mouth.data.xpos, before[3], atol=1e-10)
        np.testing.assert_allclose(np.abs(mouth.data.xquat), np.abs(before[4]), atol=1e-10)
        moving = [mj.mj_name2id(mouth.model, mj.mjtObj.mjOBJ_BODY, 'nmf/'+s)
                  for s in ('c_rostrum', 'c_haustellum')]
        fixed = np.ones(mouth.model.nbody, dtype=bool)
        fixed[moving] = False
        images, states = {}, {}
        for label, rates in [('rest', {'left': 0, 'right': 0}),
                             ('active', {'left': 63, 'right': 95}),
                             ('washout', {'left': 0, 'right': 0})]:
            for _ in range(100):
                mouth.step(rates)
                assert np.all(mouth.data.qpos >= mouth.model.jnt_range[:, 0]-.001)
                assert np.all(mouth.data.qpos <= mouth.model.jnt_range[:, 1]+.001)
                np.testing.assert_allclose(mouth.data.xpos[fixed], before[3][fixed], atol=1e-10)
            states[label] = mouth.snapshot()
            images[label] = mouth.image()
        assert abs(states['active']['angles_deg']['rostrum_pitch']) > 70
        assert states['active']['angles_deg']['mouth_yaw'] > 3
        for label in ('rest', 'washout'):
            assert max(abs(a) for a in states[label]['angles_deg'].values()) < .001
        assert images['rest'] != images['active']
        np.testing.assert_array_equal(world.data.qpos, before[0])
        np.testing.assert_array_equal(world.data.qvel, before[1])
        assert world.data.time == before[2] and world.path == before[5]
        world.start_feeding()
        assert world.feeding.data.time == 0
        world.reset()
        assert world.feeding is None
        result = {'passed': True, 'checks': ['bilateral symmetry', 'zero symmetric yaw',
                  'bounded targets', 'preserved arrival pose', 'fixed torso and legs',
                  'dynamic joint limits', 'extension and turning', 'washout retraction',
                  'rendered image changes', 'navigation state unchanged', 'replay and reset'],
                  'states': states}
        (ROOT/'outputs/proboscis-validation.json').write_text(json.dumps(result, indent=2)+'\n')
        print('Proboscis dynamics and navigation isolation passed', flush=True)
    finally:
        world.close()


if __name__ == '__main__':
    main()
