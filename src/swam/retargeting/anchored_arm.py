"""R1 arm kinematics solved backward from an immutable hand mount pose.

The arm base is floating and inferred, not a calibrated shared robot torso.
Native link dimensions and joint limits are retained.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import label

from swam.r1 import SOURCE, origin_matrix


class WristAnchoredArm:
    def __init__(self, side, spacer=.012):
        self.side = side
        self.robot = ET.parse(SOURCE).getroot()
        self.joints, self.lower, self.upper = [], [], []
        for i in range(1,7):
            joint = self.robot.find(f"joint[@name='{side}_arm_joint{i}']")
            self.joints.append((joint.find('parent').get('link'),joint.find('child').get('link'),
                                origin_matrix(joint.find('origin')),
                                np.fromstring(joint.find('axis').get('xyz'),sep=' ')))
            self.lower.append(float(joint.find('limit').get('lower')))
            self.upper.append(float(joint.find('limit').get('upper')))
        self.lower,self.upper = np.array(self.lower),np.array(self.upper)
        self.mount = np.eye(4)
        self.mount[:3,:3] = Rotation.from_euler('y',np.pi/2).as_matrix()
        self.mount[0,3] = .03425+spacer
        self.spacer = spacer
        self.nodes = {}

    def backward(self, wrist, q):
        if np.asarray(q).shape != (6,):
            raise ValueError('Expected six arm joints')
        links = {self.side+'_arm_link6':wrist@np.linalg.inv(self.mount)}
        for (parent,child,origin,axis),angle in reversed(list(zip(self.joints,q))):
            local = origin.copy()
            local[:3,:3] @= Rotation.from_rotvec(axis*angle).as_matrix()
            links[parent] = links[child]@np.linalg.inv(local)
        return links

    def forward(self, base, q):
        links = {self.side+'_arm_base_link':base}
        for (parent,child,origin,axis),angle in zip(self.joints,q):
            local = origin.copy()
            local[:3,:3] @= Rotation.from_rotvec(axis*angle).as_matrix()
            links[child] = links[parent]@local
        return links[self.side+'_arm_link6']@self.mount

    def add_visuals(self, scene):
        import pyrender
        import trimesh
        # Retain the vendor's actual articulated geometry; match the accepted hand finish.
        for i in range(1,7):
            name = self.side+f'_arm_link{i}'
            visual = self.robot.find(f"link[@name='{name}']/visual")
            path = SOURCE.parent.parent/'meshes'/Path(visual.find('geometry/mesh').get('filename')).name
            mesh = trimesh.load(path,force='mesh',process=False)
            mesh.apply_transform(origin_matrix(visual.find('origin')))
            color = [.74,.77,.8,1.] if i in (2,3) else [.28,.31,.34,1.]
            material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color,metallicFactor=.3,roughnessFactor=.6)
            self.nodes[name] = scene.add(pyrender.Mesh.from_trimesh(mesh,material=material))
        adapter = trimesh.creation.cylinder(radius=.024,height=self.spacer+.001,sections=48)
        transform = np.eye(4)
        transform[:3,:3] = self.mount[:3,:3]
        transform[0,3] = .03425+self.spacer/2
        adapter.apply_transform(transform)
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[.18,.2,.22,1],metallicFactor=.4,roughnessFactor=.5)
        self.nodes[self.side+'_adapter'] = scene.add(pyrender.Mesh.from_trimesh(adapter,material=material))

    def show(self, scene, wrist, q):
        links = self.backward(wrist,q)
        for name,node in self.nodes.items():
            scene.set_pose(node,links[self.side+'_arm_link6'] if name.endswith('_adapter') else links[name])
        return links


def proximal_forearm_mask(mask, wrist_uv, palm_uv):
    """Exclude the distal hand even if SAM returned hand and arm as one instance."""
    yy, xx = np.indices(mask.shape)
    direction = np.asarray(wrist_uv) - np.asarray(palm_uv)
    if not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
        return np.zeros_like(mask, dtype=bool)
    return np.asarray(mask, dtype=bool) & ((np.stack([xx, yy], -1) - wrist_uv) @ direction > 0)


def observed_forearm_mask(sleeve, hand_track, wrist_uv, palm_uv):
    """Use proximal hand-track pixels only when the sleeve has no usable evidence.

    Require a connected component near the wrist with nontrivial proximal extent;
    never extrapolate an invisible arm or borrow another instance's pixels.
    """
    proximal = proximal_forearm_mask(sleeve, wrist_uv, palm_uv)
    if proximal.sum() >= 12:
        return proximal, 'sleeve'
    fallback = proximal_forearm_mask(hand_track, wrist_uv, palm_uv)
    components, count = label(fallback)
    palm_length = np.linalg.norm(np.asarray(wrist_uv) - palm_uv)
    if not count or not np.isfinite(palm_length) or palm_length < 1e-8:
        return proximal, 'missing'
    best = None
    for identity in range(1, count+1):
        y, x = np.nonzero(components == identity)
        if len(x) < 64:
            continue
        offsets = np.stack([x, y], -1) - wrist_uv
        distance = np.linalg.norm(offsets, axis=1).min()
        extent = np.quantile(offsets @ ((np.asarray(wrist_uv)-palm_uv)/palm_length), .9)
        if distance <= max(12., .75*palm_length) and extent >= max(12., .5*palm_length):
            if best is None or distance < best[0]:
                best = (distance, identity)
    if best is None:
        return proximal, 'missing'
    return components == best[1], 'proximal_hand_track'


def supported_forearm_mask(mask, wrist_uv, palm_uv):
    """Reject a hand sliver or disconnected mask as evidence of a forearm.

    Thresholds are in units of projected wrist-to-palm length. No unseen pixels
    are created, and a missing result is not a successful arm reconstruction.
    """
    wrist = np.asarray(wrist_uv); direction = wrist - palm_uv
    length = np.linalg.norm(direction)
    empty = np.zeros_like(mask, dtype=bool)
    if not np.isfinite(length) or length < 1e-8:
        return empty
    direction /= length
    proximal = proximal_forearm_mask(mask, wrist, palm_uv)
    components, count = label(proximal)
    best = None
    for identity in range(1, count + 1):
        y, x = np.nonzero(components == identity)
        if len(x) < 64:
            continue
        offsets = np.stack([x, y], -1) - wrist
        distances = np.linalg.norm(offsets, axis=1)
        extent = np.quantile(offsets @ direction, .9)
        far = offsets[distances >= np.quantile(distances, .55)]
        axis = np.median(far, axis=0)
        agreement = np.dot(axis, direction) / max(np.linalg.norm(axis), 1e-8)
        if (distances.min() <= max(12., .75 * length)
                and extent >= max(12., .35 * length) and agreement >= .5):
            if best is None or len(x) > best[0]:
                best = (len(x), identity)
    return empty if best is None else components == best[1]


def projected_chain_error(arm, wrist, q, intrinsic, direction, image_size):
    """Penalize inward folding and a floating proximal end in the observed view.

    This is a 2D sleeve-continuation prior, not a recovered anatomical shoulder.
    The wrist remains exactly anchored by backward kinematics.
    """
    links = arm.backward(wrist, q)
    xyz = np.array([wrist[:3, 3]] + [links[arm.side + name][:3, 3] for name in
                                   ('_arm_link5', '_arm_link4', '_arm_link2', '_arm_link1')])
    pixels = xyz @ intrinsic.T
    uv = pixels[:, :2] / np.maximum(pixels[:, 2:], .01)
    direction = np.asarray(direction)
    direction = direction / max(np.linalg.norm(direction), 1e-8)
    normal = np.array([-direction[1], direction[0]])
    progress = (uv - uv[0]) @ direction
    lateral = (uv - uv[0]) @ normal
    distances = []
    for coordinate, delta, limit in zip(uv[0], direction, image_size):
        if abs(delta) > 1e-8:
            distance = ((limit if delta > 0 else 0) - coordinate) / delta
            if distance >= 0:
                distances.append(distance)
    exit_distance = min(distances, default=0.) + 40.
    return np.r_[lateral[1:] / 50.,
                 np.maximum(0., -np.diff(progress)) / 30.,
                 max(0., exit_distance - progress[-1]) / 35.]
