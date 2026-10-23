"""Use observed instance masks to request pose inference, never to invent poses."""
import numpy as np
from scipy.ndimage import distance_transform_edt


def mask_box(mask, image_size, minimum_area=64):
    y, x = np.nonzero(mask)
    if len(x) < minimum_area:
        return None
    height, width = mask.shape
    scale = np.array([image_size[0] / width, image_size[1] / height])
    return np.r_[np.array([x.min(), y.min()]) * scale,
                 np.array([x.max() + 1, y.max() + 1]) * scale]


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(overlap / max(np.maximum(0, a[2:] - a[:2]).prod()
                               + np.maximum(0, b[2:] - b[:2]).prod() - overlap, 1e-8))


def select_tracked_detection(detections, box):
    """Resolve duplicate handedness only with a clear spatial match to its track."""
    if box is None or not detections:
        return None
    scores = np.array([box_iou(p['hand_bbox'], box) for p in detections])
    order = np.argsort(scores)[::-1]
    if scores[order[0]] < .2 or (len(order) > 1 and scores[order[0]] - scores[order[1]] < .15):
        return None
    return detections[order[0]]


def reconcile_tracked_identities(detections, labels, image_size, previous_boxes=None,
                                 require_observed_track=False):
    """Reject a handedness claim contradicted by a persistent instance track.

    Do not mirror or relabel an already regressed pose: handedness conditions
    WiLoR inference. The caller must re-estimate a rejected pose with the track's
    identity. Weak/ambiguous masks leave the original detection unchanged.
    """
    distances = {side: distance_transform_edt(labels != identity)
                 for side, identity in ((0, 2), (1, 3))
                 if np.count_nonzero(labels == identity) >= 64}
    scale = np.array([labels.shape[1], labels.shape[0]]) / image_size
    retained, rejected = [], []
    for index, prediction in enumerate(detections):
        points = np.asarray(prediction['wilor_preds']['pred_keypoints_2d']).reshape(-1, 2) * scale
        inside = (np.isfinite(points).all(axis=1) & (points[:, 0] >= 0)
                  & (points[:, 0] < labels.shape[1]) & (points[:, 1] >= 0)
                  & (points[:, 1] < labels.shape[0]))
        xy = points[inside].astype(int)
        support = {side: float(np.mean(distance[xy[:, 1], xy[:, 0]] <= 8))
                   if len(xy) >= 5 else 0. for side, distance in distances.items()}
        claimed = int(prediction['is_right'])
        own, other = support.get(claimed, 0.), support.get(1-claimed, 0.)
        continuity = (previous_boxes is not None and 1-claimed in previous_boxes
                      and box_iou(prediction['hand_bbox'],previous_boxes[1-claimed]) >= .3)
        if own <= .15 and (other >= .6 or (other >= .25 and continuity)):
            rejected.append(dict(detection=index, claimed_side=claimed,
                                 tracked_side=1-claimed, support=support,
                                 temporal_box_match=bool(continuity),
                                 reason='persistent_track_contradicts_handedness'))
        elif require_observed_track and own < .25:
            # A first-frame track cannot certify a newly entering hand. Keep this
            # hypothesis in the audit, but do not render it as an accepted hand.
            rejected.append(dict(detection=index, claimed_side=claimed,
                                 tracked_side=None, support=support,
                                 reason='unverified_detection_requires_track_initialization'))
        else:
            retained.append(prediction)
    return retained, rejected
