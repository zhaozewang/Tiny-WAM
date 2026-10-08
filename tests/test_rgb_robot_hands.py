"""Check the virtual-camera adapter without image-model inference."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('pinocchio')
pytest.importorskip('pyrender')
pytest.importorskip('retargeting')
scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec = importlib.util.spec_from_file_location('rgb_robot_pilot',scripts/'render_rgb_robot_hands.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
sys.path.remove(str(scripts))


@pytest.mark.parametrize('side',['left','right'])
def test_floating_fk_matches_native_geometry(side,tmp_path):
    hand = pilot.SharpaHand(side,pilot.pyrender.Scene())
    model = pilot.floating_robot(hand,tmp_path/'hand.urdf')
    robot = pilot.RobotPinocchio(str(tmp_path/'hand.urdf'),'urdf')
    adaptor = pilot.RobotAdaptor(robot,model.names+hand.names)
    q = np.r_[[.2,-.1,14.,.3,-.4,.2], (hand.lower+hand.upper)/2]
    robot.compute_forward_kinematics(adaptor.forward_qpos(q))
    root = robot.get_frame_pose(hand.root)
    params = np.r_[pilot.Rotation.from_matrix(root[:3,:3]).as_rotvec(), root[:3,3], q[6:]]
    points,links = hand.forward(params)
    for name in hand.links:
        np.testing.assert_allclose(robot.get_frame_pose(name),links[name],atol=1e-10)
    k = np.array([[37500.,0,960],[0,37500.,540],[0,0,1.]])
    uv = points @ k.T
    uv = uv[:,:2]/uv[:,2:]
    fitted,stats = pilot.refine_pixels(robot,adaptor,model,q,uv,points,k,None)
    assert stats['tip_mean_px'] < 1e-6
    np.testing.assert_allclose(fitted,q,atol=1e-6)


def test_wilor_to_sharpa_joint_mapping():
    assert sorted(pilot.ORDER) == list(range(21))
    assert np.array(pilot.ORDER)[pilot.TIPS].tolist() == [4,8,12,16,20]
    assert np.array(pilot.ORDER)[pilot.BASES].tolist() == [3,7,11,15,19]


@pytest.mark.parametrize('side',['left','right'])
def test_silhouette_adapter_preserves_frozen_pose(side,tmp_path):
    sys.path.insert(0,str(scripts))
    try:
        import refine_rgb_hand_alignment as alignment
    finally:
        sys.path.remove(str(scripts))
    hand = pilot.SharpaHand(side,pilot.pyrender.Scene())
    model = pilot.floating_robot(hand,tmp_path/'hand.urdf')
    robot = pilot.RobotPinocchio(str(tmp_path/'hand.urdf'),'urdf')
    adaptor = pilot.RobotAdaptor(robot,model.names+hand.names)
    q = np.r_[[.3,-.2,15.,.4,-.2,.6],(hand.lower+hand.upper)/2]
    robot.compute_forward_kinematics(adaptor.forward_qpos(q))
    _,transforms = hand.forward(alignment.native_parameters(q))
    for name in hand.nodes:
        np.testing.assert_allclose(transforms[name],robot.get_frame_pose(name),atol=1e-10)
