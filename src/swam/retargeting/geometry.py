"""Build camera-aligned ISyHand conditioning from a paired EgoDex clip."""
import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import pyrender
from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from swam.r1 import HAND, origin_matrix, poses, mirror_hand

FINGERS = [('IndexFinger', '1'), ('MiddleFinger', '3'), ('RingFinger', '4'), ('Thumb', '2')]
INDICES = np.arange(20, 86, 2)
CROP = (640, 360, 1920, 1080)
SIZE = (512, 288)


class SharpaHand:
    """Native vendor geometry and bounded kinematics; no mesh deformation."""

    def __init__(self, side, scene, scale=1.):
        folder = ROOT / 'assets/sharpa/wave_01' / (side + '_sharpa_wave')
        self.robot = ET.parse(folder / (side + '_sharpa_wave.urdf')).getroot()
        self.names, lower, upper, self.chain = [], [], [], []
        for joint in self.robot.findall('joint'):
            index = None
            if joint.get('type') == 'revolute':
                index = len(self.names)
                self.names.append(joint.get('name'))
                lower.append(float(joint.find('limit').get('lower')))
                upper.append(float(joint.find('limit').get('upper')))
            axis = joint.find('axis')
            origin = origin_matrix(joint.find('origin'))
            origin[:3, 3] *= scale
            self.chain.append((joint.find('parent').get('link'), joint.find('child').get('link'),
                origin, index,
                np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else None))
        self.lower, self.upper = np.array(lower), np.array(upper)
        self._joint_rotations = {}
        self._wrist_rotation = None
        self.root = side + '_hand_C_MC'
        self.links = [self.root]
        self.labels = [side + 'Hand']
        for human, robot in [('IndexFinger', 'index'), ('MiddleFinger', 'middle'),
                             ('RingFinger', 'ring'), ('LittleFinger', 'pinky'), ('Thumb', 'thumb')]:
            parts = ['MC', 'PP', 'DP', 'fingertip'] if robot == 'thumb' else ['PP', 'MP', 'DP', 'fingertip']
            self.links.extend(side + '_' + robot + '_' + part for part in parts)
            self.labels.extend(side + human + part for part in ['Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip'])
        self.nodes = {}
        for link in self.robot.findall('link'):
            visual = link.find('visual')
            if visual is None:
                continue
            mesh = visual.find('geometry/mesh')
            path = folder / 'meshes' / Path(mesh.get('filename')).name
            geometry = trimesh.load(path, force='mesh', process=False)
            geometry.apply_scale(np.fromstring(mesh.get('scale', '1 1 1'), sep=' '))
            geometry.apply_transform(origin_matrix(visual.find('origin')))
            geometry.apply_scale(scale)
            rgba = np.fromstring(visual.find('material/color').get('rgba'), sep=' ')
            material = pyrender.MetallicRoughnessMaterial(baseColorFactor=rgba, metallicFactor=.15, roughnessFactor=.65)
            self.nodes[link.get('name')] = scene.add(pyrender.Mesh.from_trimesh(geometry, material=material, smooth=False))

    def forward(self, parameters):
        wrist = np.eye(4)
        rotation_key = tuple(parameters[:3])
        if self._wrist_rotation is None or self._wrist_rotation[0] != rotation_key:
            self._wrist_rotation = (rotation_key, Rotation.from_rotvec(parameters[:3]).as_matrix())
        wrist[:3, :3] = self._wrist_rotation[1]
        wrist[:3, 3] = parameters[3:6]
        transforms = {self.root: wrist}
        for parent, child, origin, index, axis in self.chain:
            transform = origin.copy()
            if index is not None:
                angle = parameters[6 + index]
                cached = self._joint_rotations.get(index)
                if cached is None or cached[0] != angle:
                    cached = (angle, Rotation.from_rotvec(axis * angle).as_matrix())
                    self._joint_rotations[index] = cached
                transform[:3, :3] @= cached[1]
            transforms[child] = transforms[parent] @ transform
        return np.array([transforms[name][:3, 3] for name in self.links]), transforms

    def show(self, scene, parameters):
        points, transforms = self.forward(parameters)
        for name, node in self.nodes.items():
            scene.set_pose(node, transforms[name])
        return points, transforms


def project(points, intrinsic):
    pixels = points @ intrinsic.T
    return pixels[:, :2] / np.maximum(pixels[:, 2:], .01)


def fit_sharpa(hand, target, confidence, intrinsic, previous):
    zero = np.r_[np.zeros(6), np.clip(np.zeros(len(hand.names)), hand.lower, hand.upper)]
    neutral, _ = hand.forward(zero)
    anchors = [0, 1, 5, 9, 13]
    a, b = neutral[anchors], target[anchors]
    u, _, vt = np.linalg.svd((a - a.mean(0)).T @ (b - b.mean(0)))
    rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    initial = zero.copy()
    initial[:3] = Rotation.from_matrix(rotation).as_rotvec()
    initial[3:6] = b.mean(0) - rotation @ a.mean(0)
    if previous is not None:
        initial = previous.copy()
    pixels = project(target, intrinsic)
    weights = np.sqrt(np.clip(confidence, 0, 1))[:, None]

    def residual(parameters):
        points, _ = hand.forward(parameters)
        terms = [((project(points, intrinsic) - pixels) * weights).ravel(),
                 ((points - target) * weights * 80).ravel()]
        if previous is not None:
            terms.append((parameters[6:] - previous[6:]) * .5)
        return np.concatenate(terms)

    bounds = (np.r_[np.full(6, -np.inf), hand.lower], np.r_[np.full(6, np.inf), hand.upper])
    result = least_squares(residual, initial, bounds=bounds, max_nfev=100, ftol=1e-5)
    return result.x


def refine_silhouette(hand, scene, renderer, parameters, target, confidence, intrinsic, mask, region):
    """Coordinate descent against actual rasterized mask pixels, with a joint prior."""
    target_mask = mask & region
    outside = distance_transform_edt(~target_mask)
    target_pixels = project(target, intrinsic)
    lower = np.r_[np.full(6, -np.inf), hand.lower]
    upper = np.r_[np.full(6, np.inf), hand.upper]
    steps = np.r_[np.full(3, .035), np.full(3, .002), np.full(len(hand.names), .07)]
    anchor = parameters.copy()

    def evaluate(p):
        hand.show(scene, p)
        depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        predicted = (depth > 0) & region
        intersection = (predicted & target_mask).sum()
        union = (predicted | target_mask).sum()
        iou = float(intersection / max(union, 1))
        points, _ = hand.forward(p)
        error = np.linalg.norm(project(points, intrinsic) - target_pixels, axis=1)
        joint_loss = np.average(error, weights=np.maximum(confidence, .01))
        outside_loss = np.minimum(outside[predicted], 12).mean() if predicted.any() else 12
        # Keep the frame's landmark solution as a weak pose prior during refinement.
        prior = np.mean((p[6:] - anchor[6:]) ** 2)
        loss = joint_loss + 12 * (1 - iou) + outside_loss + .5 * prior
        return loss, dict(iou=iou, mean_error_px=float(error.mean()), objective=float(loss))

    best, before = evaluate(parameters)
    result = parameters.copy()
    after = before
    for factor in (2., 2., 1., 1., .5, .5, .25, .25):
        for dimension, step in enumerate(steps * factor):
            candidate_best, candidate_metrics = result, after
            for sign in (-1, 1):
                trial = result.copy()
                trial[dimension] = np.clip(trial[dimension] + sign * step, lower[dimension], upper[dimension])
                loss, metrics = evaluate(trial)
                if loss < best:
                    best, candidate_best, candidate_metrics = loss, trial, metrics
            result, after = candidate_best, candidate_metrics
    hand.show(scene, result)
    return result, dict(before=before, after=after)


def prepare_paired(output, indices, crop, member):
    """Prepare paired RGB/annotations without fitting a particular robot."""
    reader = imageio.get_reader(output / 'source.mp4')
    frames = [np.asarray(Image.fromarray(reader.get_data(i)).crop(crop).resize(SIZE)) for i in indices]
    fps = reader.get_meta_data()['fps'] / (indices[1] - indices[0])
    reader.close()
    prompts = {}
    with h5py.File(output / 'source.hdf5') as data:
        intrinsic = data['camera/intrinsic'][:].copy()
        intrinsic[0, 2] -= crop[0]
        intrinsic[1, 2] -= crop[1]
        intrinsic[0] *= SIZE[0] / (crop[2] - crop[0])
        intrinsic[1] *= SIZE[1] / (crop[3] - crop[1])
        for side in ('left', 'right'):
            # Match the pilot's wrist and four landmarks on each of four fingers.
            points, _ = targets(data, side, indices[0])
            prompts[side] = project(points, intrinsic).tolist()
    np.save(output / 'source_frames.npy', frames)
    (output / 'prompts.json').write_text(json.dumps(prompts))
    (output / 'prepare.json').write_text(json.dumps(dict(source_member=member, indices=indices,
        crop=list(crop), size=list(SIZE), fps=fps, intrinsic=intrinsic.tolist(), physics=False), indent=2))
    imageio.mimwrite(output / 'original.mp4', frames, fps=fps, macro_block_size=1)


def prepare_sharpa(source, output, selected, scale=1., silhouette=False):
    """Compare native five-finger registration with the preserved ISyHand pilot."""
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / 'prepare.json').read_text())
    intrinsic = np.array(config['intrinsic'])
    frames = np.load(source / 'source_frames.npy')
    masks = np.load(source / 'human_masks.npy') > 0
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[.65, .65, .65])
    camera_pose = np.diag([1., -1., -1., 1.])
    scene.add(pyrender.IntrinsicsCamera(intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]), pose=camera_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2), pose=camera_pose)
    hands = {side: SharpaHand(side, scene, scale) for side in ('left', 'right')}
    renderer = pyrender.OffscreenRenderer(*SIZE)
    previous = dict(left=None, right=None)
    records, rendered, robot_masks, comparisons = [], [], [], []
    with h5py.File(source / 'source.hdf5') as data:
        for index in selected:
            frame_index = config['indices'][index]
            camera = np.linalg.inv(data['transforms/camera'][frame_index])
            record = dict(index=int(index), source_frame=frame_index)
            targets = {side: np.array([(camera @ data['transforms/' + name][frame_index])[:3, 3]
                       for name in hand.labels]) for side, hand in hands.items()}
            yy, xx = np.indices(masks.shape[1:])
            grid = np.stack([xx, yy], axis=-1)
            distances = {side: cKDTree(project(target, intrinsic)).query(grid.reshape(-1, 2))[0].reshape(xx.shape)
                         for side, target in targets.items()}
            for side, hand in hands.items():
                target = targets[side]
                confidence = np.array([data['confidences/' + name][frame_index] for name in hand.labels])
                parameters = fit_sharpa(hand, target, confidence, intrinsic, previous[side])
                metrics = None
                if silhouette:
                    other = 'right' if side == 'left' else 'left'
                    for node in hands[other].nodes.values():
                        node.mesh.is_visible = False
                    pixels = project(target, intrinsic)
                    direction = pixels[[1, 5, 9, 13]].mean(0) - pixels[0]
                    direction /= np.linalg.norm(direction)
                    # SAM includes bare forearms. Score only the palm/finger side of the wrist.
                    region = ((grid - pixels[0]) @ direction > -4) & (distances[side] < distances[other])
                    parameters, metrics = refine_silhouette(hand, scene, renderer, parameters, target,
                        confidence, intrinsic, masks[index], region)
                    for node in hands[other].nodes.values():
                        node.mesh.is_visible = True
                previous[side] = parameters
                points, _ = hand.forward(parameters)
                hand.show(scene, parameters)
                error = np.linalg.norm(project(points, intrinsic) - project(target, intrinsic), axis=1)
                record[side] = dict(parameters=parameters.tolist(), joint_names=hand.names, error_px=error.tolist(),
                                    error_m=np.linalg.norm(points-target, axis=1).tolist(), confidence=confidence.tolist(), silhouette=metrics)
            rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            rgb, mask = rgb[:, :, :3], depth > 0
            rendered.append(rgb)
            robot_masks.append(mask)
            overlay = np.where(mask[:, :, None], rgb, frames[index])
            comparison = np.concatenate([frames[index], overlay], axis=1)
            comparisons.append(comparison)
            Image.fromarray(comparison).save(output / f'alignment_{index:03d}.png')
            records.append(record)
            print('Sharpa frame', index, 'mean projection error px', round(np.mean([record[s]['error_px'] for s in hands]), 2), flush=True)
    renderer.delete()
    np.savez_compressed(output / 'robot_render.npz', rgb=rendered, mask=robot_masks)
    (output / 'joints.json').write_text(json.dumps(records))
    (output / 'prepare.json').write_text(json.dumps(dict(config, source=str(source), hand='Sharpa Wave',
        native_scale=scale, mesh_deformation=False, selected=list(map(int, selected)), silhouette_refinement=silhouette,
        objective='Confidence weighted 2D landmarks + 3D prior + temporal joints; optional raster silhouette refinement',
        limitations='Visible SAM mask has no object depth; silhouette cannot establish valid contact or hidden finger pose.'), indent=2))
    imageio.mimwrite(output / 'alignment.mp4', comparisons, fps=config['fps'], macro_block_size=1)


def hand_model(side, output):
    robot = ET.parse(HAND).getroot()
    for mesh in robot.findall('.//mesh'):
        mesh.set('filename', str((HAND.parent.parent / mesh.get('filename')).resolve()))
    if side == 'left':
        mirror_hand(robot, output / 'left_meshes')
    joints = [j for j in robot.findall('joint') if j.get('type') == 'revolute']
    names = [j.get('name') for j in joints]
    lower = [float(j.find('limit').get('lower')) for j in joints]
    upper = [float(j.find('limit').get('upper')) for j in joints]
    return robot, names, np.array(lower), np.array(upper)


def landmarks(robot, names, angles, side):
    transforms = poses(robot, dict(zip(names, angles)))
    wrist = np.array([0., .043 if side == 'right' else -.043, .028])
    points = [wrist]
    for finger, number in FINGERS:
        for joint in ('3', '2', '1'):
            points.append(transforms['J' + number + joint][:3, 3])
        tip = transforms['J' + number + '1'] @ [0, 0, -.0475, 1]
        points.append(tip[:3])
    return np.array(points), transforms


def targets(data, side, frame):
    camera = np.linalg.inv(data['transforms/camera'][frame])
    names = [side + 'Hand']
    for finger, number in FINGERS:
        parts = ['Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip']
        if finger == 'Thumb':
            # Three anatomical thumb joints constrain the four robot joints.
            parts = ['Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip']
        names.extend(side + finger + part for part in parts)
    points = np.array([(camera @ data['transforms/' + name][frame])[:3, 3] for name in names])
    confidence = np.array([data['confidences/' + name][frame] for name in names])
    return points, confidence


def fit(robot, names, lower, upper, side, target, previous):
    neutral, _ = landmarks(robot, names, np.zeros(len(names)), side)
    if previous is None:
        # A rigid palm fit initializes the wrist; finger angles retain URDF limits.
        anchors = [0, 1, 5, 9]
        a, b = neutral[anchors], target[anchors]
        u, _, vt = np.linalg.svd((a - a.mean(0)).T @ (b - b.mean(0)))
        rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
        translation = b.mean(0) - rotation @ a.mean(0)
        previous = np.r_[Rotation.from_matrix(rotation).as_rotvec(), translation, np.zeros(len(names))]
    def residual(parameters):
        points, _ = landmarks(robot, names, parameters[6:], side)
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        return (points @ rotation.T + parameters[3:6] - target).ravel()
    bounds = (np.r_[np.full(6, -np.inf), lower], np.r_[np.full(6, np.inf), upper])
    result = least_squares(residual, previous, bounds=bounds, max_nfev=60, ftol=1e-5)
    points, transforms = landmarks(robot, names, result.x[6:], side)
    wrist = np.eye(4)
    wrist[:3, :3] = Rotation.from_rotvec(result.x[:3]).as_matrix()
    wrist[:3, 3] = result.x[3:6]
    error = np.linalg.norm(residual(result.x).reshape(-1, 3), axis=1)
    return result.x, {name: wrist @ pose for name, pose in transforms.items()}, error


def mesh_nodes(scene, robot):
    nodes = {}
    for link in robot.findall('link'):
        visual = link.find('visual')
        if visual is None:
            continue
        geometry = trimesh.load(visual.find('geometry/mesh').get('filename'), force='mesh', process=False)
        geometry.apply_transform(origin_matrix(visual.find('origin')))
        color = [.08, .085, .09, 1] if link.get('name') != 'base_link' else [.48, .5, .52, 1]
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color, metallicFactor=.4, roughnessFactor=.5)
        mesh = pyrender.Mesh.from_trimesh(geometry, material=material, smooth=False)
        nodes[link.get('name')] = scene.add(mesh)
    return nodes
