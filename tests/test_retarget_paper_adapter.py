"""Optional integration checks for the isolated upstream-retargeting environment."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('pinocchio')
pytest.importorskip('pyrender')
pytest.importorskip('retargeting')

path = Path(__file__).resolve().parents[1] / 'scripts/compare_egodex_retargeting.py'
spec = importlib.util.spec_from_file_location('paper_pilot', path)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_rigid_fit_recovers_pose_without_scaling():
    rng = np.random.default_rng(19)
    a = rng.normal(size=(8, 3))
    r = pilot.Rotation.from_rotvec([.4, -.7, .2]).as_matrix()
    b = a @ r.T + [.5, .2, -.3]
    fit = pilot.rigid_fit(a, b)
    np.testing.assert_allclose(fit[:3, :3], r, atol=1e-12)
    np.testing.assert_allclose(a @ fit[:3, :3].T + fit[:3, 3], b, atol=1e-12)
    assert np.linalg.det(fit[:3, :3]) == pytest.approx(1.)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_exported_robot_matches_renderer_and_tip_order(side, tmp_path):
    from retargeting.core.kinematics.pinocchio_model import RobotPinocchio
    from retargeting.core.kinematics.adaptor import RobotAdaptor
    hand = pilot.SharpaHand(side, pilot.pyrender.Scene())
    model = pilot.ConnectedHand(side, hand)
    base = np.eye(4)
    base[:3, :3] = pilot.Rotation.from_rotvec([.2, -.1, .7]).as_matrix()
    base[:3, 3] = [.2, .3, 1.]
    urdf = tmp_path / 'robot.urdf'
    pilot.write_robot(model, base, urdf)
    robot = RobotPinocchio(str(urdf), 'urdf')
    adaptor = RobotAdaptor(robot, model.names + hand.names)
    assert pilot.validate_fk(model, robot, adaptor, base) < 1e-8
    expected = ['Thumb', 'IndexFinger', 'MiddleFinger', 'RingFinger', 'LittleFinger']
    assert [hand.labels[i] for i in pilot.TIPS] == [side + finger + 'Tip' for finger in expected]
    assert [hand.labels[i] for i in pilot.BASES] == [side + finger + 'IntermediateTip' for finger in expected]


def test_identical_points_have_zero_geometric_error():
    hand = pilot.SharpaHand('right', pilot.pyrender.Scene())
    model = pilot.ConnectedHand('right', hand)
    q = np.tile((model.lower+model.upper)/2, (3, 1))
    p, _, _ = model.forward(pilot.tensor(np.eye(4)), pilot.tensor([1., 1.]), pilot.tensor(q))
    result = pilot.metrics(p.numpy(), p.numpy(), q, model, 15.)
    assert result['tip_mm'] == 0
    assert result['thumb_relative_mm'] == 0
    assert result['fingertip_direction_deg'] < 1e-5
    assert result['limit_excess_rad'] == 0
