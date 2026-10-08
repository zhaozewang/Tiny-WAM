import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from swam.retargeting.anchored_arm import WristAnchoredArm


@pytest.mark.parametrize('side',['left','right'])
def test_any_arm_joint_values_preserve_wrist_exactly(side):
    arm = WristAnchoredArm(side)
    rng = np.random.default_rng(5)
    for _ in range(30):
        wrist = np.eye(4)
        wrist[:3,:3] = Rotation.random(random_state=rng).as_matrix()
        wrist[:3,3] = rng.uniform([-1,-1,10],[1,1,25])
        original = wrist.copy()
        q = rng.uniform(arm.lower,arm.upper)
        links = arm.backward(wrist,q)
        np.testing.assert_array_equal(wrist,original)
        np.testing.assert_allclose(links[side+'_arm_link6']@arm.mount,wrist,atol=1e-12)
        np.testing.assert_allclose(arm.forward(links[side+'_arm_base_link'],q),wrist,atol=1e-12)


def test_wrong_joint_count_rejected():
    with pytest.raises(ValueError):
        WristAnchoredArm('left').backward(np.eye(4),np.zeros(7))
