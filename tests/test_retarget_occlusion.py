import numpy as np
import pytest

from tinywam.retargeting.occlusion import composite
from tinywam.retargeting.segmentation import assign_labels


def test_visible_object_only_occludes_intersecting_robot_pixels():
    source = np.full((4, 5, 3), 30, np.uint8)
    robot = np.full_like(source, 200)
    robot_mask = np.zeros((4, 5), bool)
    robot_mask[:, 1:4] = True
    objects = np.zeros_like(robot_mask)
    objects[1:3, 2] = True
    image, visible = composite(source, robot, robot_mask, objects)
    assert np.array_equal(image[objects], source[objects])
    assert (image[visible] == 200).all()
    assert visible.sum() == robot_mask.sum() - 2
    assert (image[~robot_mask] == 30).all()


def test_joint_labels_preserve_foreground_hand_over_object():
    logits = np.array([[[3, 2, -1]], [[1, 4, -2]]], dtype=float)
    labels = assign_labels(logits, [1, 2])
    assert labels.tolist() == [[1, 2, 0]]
    frame = np.zeros((1, 3, 3), np.uint8)
    robot = np.full_like(frame, 255)
    image, visible = composite(frame, robot, np.ones((1, 3), bool), labels == 1)
    assert not visible[0, 0]
    assert visible[0, 1]


def test_mismatched_shapes_fail_instead_of_broadcasting():
    with pytest.raises(ValueError):
        composite(np.zeros((4, 5, 3)), np.zeros((4, 5, 3)), np.zeros((4, 5)), np.zeros((4, 1)))


def test_empty_object_mask_reproduces_original_overlay():
    rng = np.random.default_rng(42)
    source = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    robot = rng.integers(0, 255, source.shape, dtype=np.uint8)
    mask = rng.random((8, 8)) > .5
    result, _ = composite(source, robot, mask, np.zeros_like(mask))
    assert np.array_equal(result, np.where(mask[..., None], robot, source))
