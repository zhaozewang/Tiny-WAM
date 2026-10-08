"""Coordinate and rendering checks independent of downloaded model weights."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('cv2')
pytest.importorskip('h5py')
pytest.importorskip('imageio')
pytest.importorskip('torch')
spec = importlib.util.spec_from_file_location('rgb_pilot', Path(__file__).resolve().parents[1]/'scripts/reestimate_egodex_rgb.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_original_annotation_projection_uses_camera_inverse():
    camera = np.eye(4)
    camera[:3, 3] = [1., 2., 3.]
    data = {'transforms/camera': camera[None], 'camera/intrinsic': np.array([[100.,0,50],[0,100.,30],[0,0,1.]])}
    for side in ('left', 'right'):
        names = [side+'Hand']
        for finger in ['Thumb', 'IndexFinger', 'MiddleFinger', 'RingFinger', 'LittleFinger']:
            names += [side+finger+p for p in ['Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip']]
        for i, name in enumerate(names):
            pose = np.eye(4)
            pose[:3, 3] = [1.+i*.01, 2., 5.]
            data['transforms/'+name] = pose[None]
    for points in pilot.egodex_points(data, 0):
        np.testing.assert_allclose(points[:,0], 50.+np.arange(21)*.5)
        np.testing.assert_allclose(points[:,1], 30.)


def test_render_missing_or_invalid_hands_does_not_invent_points():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    np.testing.assert_array_equal(pilot.draw_joints(frame, []), frame)
    np.testing.assert_array_equal(pilot.draw_joints(frame, [np.full((21,2), np.nan)]), frame)
    out = pilot.draw_joints(frame, [np.full((21,2), 50.)])
    assert out.sum() > 0
    assert frame.sum() == 0


def test_prediction_serialization_preserves_arrays_and_handedness():
    result = pilot.serializable({'side': np.float32(1), 'points': np.ones((21,3), dtype=np.float32)})
    assert result['side'] == 1
    assert np.array(result['points']).shape == (21,3)
