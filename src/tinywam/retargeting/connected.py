"""Fit clip-constant R1/Sharpa dimensions and a connected joint trajectory."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F

from tinywam.r1 import SOURCE, origin_matrix, poses

DTYPE = torch.float64


def tensor(value):
    return torch.as_tensor(value, dtype=DTYPE)


def rotation(vector):
    """Differentiable Rodrigues formula, including zero rotation."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(vector.shape[:-1] + (3, 3))
    angle = torch.linalg.vector_norm(vector, dim=-1)[..., None, None]
    return torch.eye(3, dtype=DTYPE) + torch.sinc(angle / torch.pi) * skew + .5 * torch.sinc(angle / (2 * torch.pi)) ** 2 * (skew @ skew)


def transform(rot, position):
    top = torch.cat([rot, position[..., None]], -1)
    last = torch.zeros(top.shape[:-2] + (1, 4), dtype=DTYPE)
    last[..., 0, 3] = 1
    return torch.cat([top, last], -2)


class ConnectedHand:
    def __init__(self, side, hand, wrist_ratio=1., spacer=.012):
        self.side, self.hand = side, hand
        self.wrist_ratio, self.spacer = wrist_ratio, spacer
        self.robot = ET.parse(SOURCE).getroot()
        neutral = poses(self.robot, {})
        center = (neutral['left_arm_link2'][:3, 3] + neutral['right_arm_link2'][:3, 3]) / 2
        self.base = neutral[side + '_arm_base_link'].copy()
        self.base[:3, 3] -= center
        self.shoulder_y = float(neutral[side + '_arm_link2'][1, 3] - center[1])
        self.chain, self.names, low, high = [], [], [], []
        for index in range(1, 7):
            name = f'{side}_arm_joint{index}'
            joint = self.robot.find(f"joint[@name='{name}']")
            origin = origin_matrix(joint.find('origin'))
            if index == 6:
                origin[0, 3] *= wrist_ratio
            self.chain.append((joint.find('child').get('link'), tensor(origin),
                               tensor(np.fromstring(joint.find('axis').get('xyz'), sep=' '))))
            self.names.append(name)
            low.append(float(joint.find('limit').get('lower')))
            high.append(float(joint.find('limit').get('upper')))
        self.lower = np.r_[low, hand.lower]
        self.upper = np.r_[high, hand.upper]
        # The R1 flange faces +X; Sharpa's fingers extend along its +Z.
        self.mount_rotation = tensor(Rotation.from_euler('y', np.pi / 2).as_matrix())
        self.hand_chain = [(p, c, tensor(o), i, tensor(a) if a is not None else None)
                           for p, c, o, i, a in hand.chain]
        self.samples = {}
        for name, node in hand.nodes.items():
            vertices = np.concatenate([p.positions for p in node.mesh.primitives])
            self.samples[name] = tensor(vertices[np.linspace(0, len(vertices) - 1, 24, dtype=int)])

    def forward(self, base, scales, q, shoulder_width=None):
        arm_scale, hand_scale = scales
        position = tensor(self.base[:3, 3]) * arm_scale
        if shoulder_width is not None:
            shift = np.sign(self.shoulder_y) * shoulder_width / 2 - self.shoulder_y * arm_scale
            position = position + torch.stack([shift * 0, shift, shift * 0])
        start = transform(tensor(self.base[:3, :3]), position)
        current = (base @ start).expand(len(q), -1, -1)
        links = {self.side + '_arm_base_link': current}
        for index, (name, origin, axis) in enumerate(self.chain):
            local = transform(origin[:3, :3] @ rotation(q[:, index, None] * axis),
                              (origin[:3, 3] * arm_scale).expand(len(q), -1))
            current = current @ local
            links[name] = current
        # Flange and joint offsets follow the same clip-constant visual wrist ratio.
        mount = transform(self.mount_rotation, torch.stack([.03425 * self.wrist_ratio * arm_scale + self.spacer * hand_scale,
                          arm_scale * 0, arm_scale * 0]))
        links[self.hand.root] = current @ mount
        for parent, child, origin, index, axis in self.hand_chain:
            rot = origin[:3, :3].expand(len(q), -1, -1)
            if index is not None:
                rot = rot @ rotation(q[:, 6 + index, None] * axis)
            local = transform(rot, (origin[:3, 3] * hand_scale).expand(len(q), -1))
            links[child] = links[parent] @ local
        points = torch.stack([links[n][..., :3, 3] for n in self.hand.links], 1)
        arm = torch.stack([links[self.side + '_arm_link2'][..., :3, 3],
                           links[self.side + '_arm_link3'][..., :3, 3],
                           links[self.hand.root][..., :3, 3]], 1)
        return points, arm, links


def project(points, camera, intrinsic):
    camera_points = points @ camera[:, :3, :3].transpose(-1, -2) + camera[:, None, :3, 3]
    pixels = camera_points @ intrinsic.T
    return pixels[..., :2] / pixels[..., 2:].clamp(min=.03)


def load_targets(source, reference, models):
    config = json.loads((source / 'prepare.json').read_text())
    indices = config['indices']
    records = json.loads((reference / 'joints.json').read_text())
    targets = {}
    with h5py.File(source / 'source.hdf5') as data:
        world_camera = data['transforms/camera'][indices]
        for side, model in models.items():
            labels = model.hand.labels
            target = np.stack([data['transforms/' + label][indices, :3, 3] for label in labels], 1)
            confidence = np.stack([data['confidences/' + label][indices] for label in labels], 1)
            arms = np.stack([data['transforms/' + side + part][indices, :3, 3] for part in ['Arm', 'Forearm', 'Hand']], 1)
            accepted = np.array([r[side]['parameters'] for r in records])
            wrist = np.tile(np.eye(4), (len(indices), 1, 1))
            wrist[:, :3, :3] = Rotation.from_rotvec(accepted[:, :3]).as_matrix()
            wrist[:, :3, 3] = accepted[:, 3:6]
            targets[side] = dict(hand=tensor(target), confidence=tensor(confidence), arm=tensor(arms),
                                 wrist=tensor(world_camera @ wrist), accepted_angles=accepted[:, 6:])
    camera = tensor(np.linalg.inv(world_camera))
    intrinsic = tensor(config['intrinsic'])
    for target in targets.values():
        target['pixels'] = project(target['hand'], camera, intrinsic)
    return config, camera, intrinsic, targets


def initial_base(targets):
    left = targets['left']['arm'][:, 0].mean(0).numpy()
    right = targets['right']['arm'][:, 0].mean(0).numpy()
    lateral = left - right
    lateral /= np.linalg.norm(lateral)
    up = np.array([0., 1., 0.])
    up -= lateral * (up @ lateral)
    up /= np.linalg.norm(up)
    rot = np.stack([np.cross(lateral, up), lateral, up], 1)
    return np.r_[Rotation.from_matrix(rot).as_rotvec(), (left + right) / 2]


def initial_joints(models, targets, base_parameters, scale, camera=None, intrinsic=None, shoulder_width=None):
    base = transform(rotation(tensor(base_parameters[:3])), tensor(base_parameters[3:]))
    rng = np.random.default_rng(42)
    sequence = {}
    for side, model in models.items():
        rows = []
        previous = None
        for frame, finger in enumerate(targets[side]['accepted_angles']):
            target = targets[side]
            def residual(angles):
                q = tensor(np.r_[angles, finger][None])
                _, arm, links = model.forward(base, tensor([scale, scale]), q, tensor(shoulder_width) if shoulder_width is not None else None)
                wrist = links[model.hand.root][0].numpy()
                desired = target['wrist'][frame].numpy()
                terms = np.r_[(wrist[:3, 3] - desired[:3, 3]) * 100,
                    Rotation.from_matrix(desired[:3, :3].T @ wrist[:3, :3]).as_rotvec() * 4,
                    (arm[0, :2].numpy() - target['arm'][frame, :2].numpy()).ravel() * 15]
                if camera is not None:
                    shaft = torch.stack([links[side + '_arm_link4'][:, :3, 3], links[side + '_arm_link5'][:, :3, 3]], 1)
                    predicted = project(shaft, camera[frame:frame+1], intrinsic)[0].numpy()
                    actual = project(target['arm'][frame:frame+1, 1:], camera[frame:frame+1], intrinsic)[0].numpy()
                    a, b = np.diff(predicted, axis=0)[0], np.diff(actual, axis=0)[0]
                    terms = np.r_[terms, (a/np.linalg.norm(a) - b/np.linalg.norm(b)) * 8]
                return terms
            starts = [previous] if previous is not None else [rng.uniform(model.lower[:6], model.upper[:6]) for _ in range(8)]
            fits = [least_squares(residual, start, bounds=(model.lower[:6], model.upper[:6]), max_nfev=70) for start in starts]
            best = min(fits, key=lambda fit: np.dot(fit.fun, fit.fun))
            previous = best.x
            rows.append(np.r_[best.x, finger])
        sequence[side] = np.array(rows)
        print('Initialized connected IK:', side, flush=True)
    return sequence


def fit_sequence(models, targets, camera, intrinsic, masks, initial, separate, output, screen_fit=False):
    frames = len(camera)
    base_initial = initial_base(targets)
    scale_count = 2 if separate else 1
    scale_end = 6 + scale_count
    prefix = scale_end + int(screen_fit)
    shoulder_target = float(torch.linalg.vector_norm(targets['left']['arm'][:, 0] - targets['right']['arm'][:, 0], dim=-1).median())
    if initial is None:
        q = initial_joints(models, targets, base_initial, .8, camera if screen_fit else None,
                           intrinsic if screen_fit else None, shoulder_target if screen_fit else None)
        start = np.r_[base_initial, np.full(scale_count, .8), [shoulder_target] if screen_fit else [], q['left'].ravel(), q['right'].ravel()]
    else:
        old = np.load(initial / 'parameters.npz')
        scales = [float(old['scales'][0]), float(old['scales'][1])] if separate else [float(old['scales'][0])]
        if screen_fit:
            scales[1] = .73
        start = np.r_[old['base'], scales, [shoulder_target] if screen_fit else [], old['left'].ravel(), old['right'].ravel()]
    bounds = [(None, None)] * 6 + [(.5, 1.2)] * scale_count
    if screen_fit:
        # User requested a slightly smaller hand; keep its size fixed across the clip.
        bounds[7] = (.73, .73)
        bounds.append((.25, .55))
    for model in models.values():
        bounds += list(zip(np.tile(model.lower, frames), np.tile(model.upper, frames)))
    hand_masks = {}
    grid = np.stack(np.indices(masks.shape[1:])[::-1], -1)
    for side, target in targets.items():
        selected = masks.copy()
        if screen_fit:
            other = 'right' if side == 'left' else 'left'
            for frame, pixels in enumerate(target['pixels'].numpy()):
                direction = pixels[[1, 5, 9, 13]].mean(0) - pixels[0]
                direction /= np.linalg.norm(direction)
                own_distance = cKDTree(pixels).query(grid.reshape(-1, 2))[0].reshape(masks.shape[1:])
                other_distance = cKDTree(targets[other]['pixels'][frame].numpy()).query(grid.reshape(-1, 2))[0].reshape(masks.shape[1:])
                selected[frame] &= ((grid - pixels[0]) @ direction > -4) & (own_distance < other_distance)
        hand_masks[side] = selected
    distance = {side: tensor(np.stack([distance_transform_edt(~mask) for mask in selected]))[:, None]
                for side, selected in hand_masks.items()}
    np.savez_compressed(output / 'hand_target_masks.npz', **hand_masks)
    trace = []

    def unpack(parameters):
        base = transform(rotation(parameters[:3]), parameters[3:6])
        scales = parameters[6:scale_end] if separate else parameters[6].repeat(2)
        width = parameters[scale_end] if screen_fit else None
        q = parameters[prefix:].reshape(2, frames, 28)
        return base, scales, q, width

    def objective(values):
        parameters = tensor(values).requires_grad_()
        base, scales, q, width = unpack(parameters)
        terms = {}
        if screen_fit:
            terms['shoulder_width'] = ((width - shoulder_target) / .02).square()
        for side_index, (side, model) in enumerate(models.items()):
            target = targets[side]
            points, arm, links = model.forward(base, scales, q[side_index], width)
            pixels = project(points, camera, intrinsic)
            image_weight = 3 if screen_fit else 1
            terms[side + '_image'] = image_weight * (F.smooth_l1_loss(pixels / 3, target['pixels'] / 3, reduction='none') * target['confidence'][..., None]).mean()
            terms[side + '_3d'] = .15 * (((points - target['hand']) / .03) ** 2 * target['confidence'][..., None]).mean()
            terms[side + '_arm'] = (((arm[:, :2] - target['arm'][:, :2]) / .035) ** 2).mean()
            if screen_fit:
                # Elbows are below the image; their positive-depth projections still
                # determine the direction of the visible forearm. Do not project shoulders
                # behind the camera or clamp them to a fake near plane.
                actual = project(target['arm'][:, 1:], camera, intrinsic)
                shaft = torch.stack([links[side + '_arm_link4'][:, :3, 3], links[side + '_arm_link5'][:, :3, 3]], 1)
                predicted = project(shaft, camera, intrinsic)
                a = F.normalize(actual[:, 1] - actual[:, 0], dim=-1)
                b = F.normalize(predicted[:, 1] - predicted[:, 0], dim=-1)
                terms[side + '_forearm_direction'] = ((a - b) / .06).square().mean()
                normal = torch.stack([-a[:, 1], a[:, 0]], -1)
                offset = ((predicted - actual[:, 1:]) * normal[:, None]).sum(-1)
                terms[side + '_shaft_position'] = F.smooth_l1_loss(offset / 6, torch.zeros_like(offset))
            wrist = links[model.hand.root]
            wrist_weight = 32 if screen_fit else .5
            terms[side + '_wrist'] = wrist_weight * (((wrist[:, :3, 3] - target['wrist'][:, :3, 3]) / .025) ** 2).mean()
            orientation_weight = 4 if screen_fit else .2
            terms[side + '_orientation'] = orientation_weight * (((wrist[:, :3, :3] - target['wrist'][:, :3, :3]) / .3) ** 2).mean()
            if screen_fit:
                terms[side + '_finger_reference'] = .5 * ((q[side_index, :, 6:] - tensor(target['accepted_angles'])) / .2).square().mean()
            velocity = torch.diff(q[side_index], dim=0)
            terms[side + '_velocity'] = .02 * (velocity / .08).square().mean()
            terms[side + '_acceleration'] = .15 * (torch.diff(velocity, dim=0) / .08).square().mean()
            surface = torch.cat([samples[None] * scales[1] @ links[name][:, :3, :3].transpose(-1, -2)
                                 + links[name][:, None, :3, 3] for name, samples in model.samples.items()], 1)
            screen = project(surface, camera, intrinsic)
            grid = screen / tensor([masks.shape[2] - 1, masks.shape[1] - 1]) * 2 - 1
            outside = F.grid_sample(distance[side], grid[:, :, None], align_corners=True, padding_mode='border')
            silhouette_weight = 2 if screen_fit else .3
            terms[side + '_silhouette'] = silhouette_weight * (outside.clamp(max=12) / 3).square().mean()
        loss = sum(terms.values())
        loss.backward()
        if len(trace) % 50 == 0:
            print('Evaluation', len(trace), 'loss', round(float(loss.detach()), 4), 'scales', scales.detach().numpy().round(4), flush=True)
        trace.append(dict(loss=float(loss.detach()), scales=scales.detach().tolist()))
        return float(loss.detach()), parameters.grad.numpy().copy()

    result = minimize(objective, start, jac=True, method='L-BFGS-B', bounds=bounds,
                      options=dict(maxiter=650, ftol=1e-9, gtol=1e-5, maxls=30, maxcor=30))
    _, scales, q, width = unpack(tensor(result.x))
    extra = dict(shoulder_width=float(width)) if width is not None else {}
    np.savez(output / 'parameters.npz', base=result.x[:6], scales=scales.numpy(), left=q[0].numpy(), right=q[1].numpy(), **extra)
    (output / 'optimization.json').write_text(json.dumps(dict(success=bool(result.success), message=str(result.message),
        iterations=result.nit, evaluations=result.nfev, initial_loss=trace[0]['loss'], final_loss=float(result.fun), trace=trace), indent=2))
    return result.x[:6], scales.numpy(), q.numpy(), float(width) if width is not None else None


def render_sequence(source, reference, output, models, config, camera, intrinsic, targets, base_values, scales, q, shoulder_width=None):
    import pyrender
    import trimesh
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[.65] * 3)
    pose = np.diag([1., -1., -1., 1.])
    k = intrinsic.numpy()
    scene.add(pyrender.IntrinsicsCamera(k[0, 0], k[1, 1], k[0, 2], k[1, 2], znear=.01), pose=pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2), pose=pose)
    nodes = {}
    for side, model in models.items():
        for name, node in model.hand.nodes.items():
            scene.add_node(node)
            for primitive in node.mesh.primitives:
                primitive.positions *= scales[1]
            nodes[name] = node
        for link in model.robot.findall('link'):
            name = link.get('name')
            if name not in [side + '_arm_base_link'] + [side + f'_arm_link{i}' for i in range(1, 7)]:
                continue
            visual = link.find('visual')
            geometry = trimesh.load(SOURCE.parent.parent / 'meshes' / Path(visual.find('geometry/mesh').get('filename')).name, force='mesh')
            geometry.apply_transform(origin_matrix(visual.find('origin')))
            if name in [side + '_arm_link5', side + '_arm_link6']:
                geometry.apply_scale([model.wrist_ratio, 1, 1])
            geometry.apply_scale(scales[0])
            material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[.3, .32, .35, 1], metallicFactor=.3, roughnessFactor=.6)
            nodes[name] = scene.add(pyrender.Mesh.from_trimesh(geometry, material=material))
        adapter = trimesh.creation.cylinder(radius=.024 * min(scales), height=max(model.spacer * scales[1], .001))
        # Cylinder local +Z becomes flange +X, centered between its face and the hand root.
        adapter_pose = np.eye(4)
        adapter_pose[:3, :3] = Rotation.from_euler('y', np.pi / 2).as_matrix()
        adapter_pose[0, 3] = .03425 * model.wrist_ratio * scales[0] + model.spacer * scales[1] / 2
        adapter.apply_transform(adapter_pose)
        nodes[side + '_adapter'] = scene.add(pyrender.Mesh.from_trimesh(adapter))
    frames = np.load(source / 'source_frames.npy')
    accepted = np.load(reference / 'robot_render.npz')
    renderer = pyrender.OffscreenRenderer(frames.shape[2], frames.shape[1])
    base = transform(rotation(tensor(base_values[:3])), tensor(base_values[3:]))
    predictions = {}
    metrics = {}
    for index, (side, model) in enumerate(models.items()):
        points, arm, links = model.forward(base, tensor(scales), tensor(q[index]), tensor(shoulder_width) if shoulder_width is not None else None)
        predictions[side] = links
        error = torch.linalg.vector_norm(project(points, camera, intrinsic) - targets[side]['pixels'], dim=-1).numpy()
        shaft = torch.stack([links[side + '_arm_link4'][:, :3, 3], links[side + '_arm_link5'][:, :3, 3]], 1)
        predicted = project(shaft, camera, intrinsic).numpy()
        actual = project(targets[side]['arm'][:, 1:], camera, intrinsic).numpy()
        a, b = predicted[:, 0] - predicted[:, 1], actual[:, 0] - actual[:, 1]
        angles = np.degrees(np.arccos(np.clip(np.sum(a*b, axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1)))
        metrics[side] = dict(mean_hand_error_px=float(error.mean()), p95_hand_error_px=float(np.percentile(error, 95)),
            mean_forearm_angle_deg=float(angles.mean()), p95_forearm_angle_deg=float(np.percentile(angles, 95)),
            forearm_metric='Visible shaft axis (links4-to5), not elbow-to-palm chord',
            mean_shoulder_error_mm=float(torch.linalg.vector_norm(arm[:, 0] - targets[side]['arm'][:, 0], dim=-1).mean() * 1000),
            mean_elbow_error_mm=float(torch.linalg.vector_norm(arm[:, 1] - targets[side]['arm'][:, 1], dim=-1).mean() * 1000),
            mean_wrist_shift_mm=float(torch.linalg.vector_norm(arm[:, 2] - targets[side]['wrist'][:, :3, 3], dim=-1).mean() * 1000),
            max_arm_joint_step_rad=float(np.abs(np.diff(q[index, :, :6], axis=0)).max()),
            max_finger_joint_step_rad=float(np.abs(np.diff(q[index, :, 6:], axis=0)).max()),
            limits_valid=bool((q[index] >= model.lower - 1e-7).all() and (q[index] <= model.upper + 1e-7).all()))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    movies, rgb_frames, masks, depths = [], [], [], []
    for index, frame in enumerate(frames):
        for side, model in models.items():
            for name, values in predictions[side].items():
                if name in nodes:
                    scene.set_pose(nodes[name], camera[index].numpy() @ values[index].numpy())
            scene.set_pose(nodes[side + '_adapter'], camera[index].numpy() @ predictions[side][side + '_arm_link6'][index].numpy())
        rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        rgb, mask = rgb[:, :, :3], depth > 0
        rgb_frames.append(rgb); masks.append(mask); depths.append(depth)
        overlay = np.where(mask[:, :, None], rgb, frame)
        old = np.where(accepted['mask'][index, :, :, None], accepted['rgb'][index], frame)
        canvas = Image.new('RGB', (frame.shape[1] * 3, frame.shape[0] + 30), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        for column, (panel, label) in enumerate(zip([frame, old, overlay], ['Source', 'Hand-only reference', f'Connected arms + hands ({scales[0]:.3f} / {scales[1]:.3f})'])):
            canvas.paste(Image.fromarray(panel), (column * frame.shape[1], 30))
            draw.text((column * frame.shape[1] + 8, 6), label, fill=(20, 20, 20), font=font)
        movies.append(np.asarray(canvas))
        if index in [0, len(frames) // 2, len(frames) - 1]:
            canvas.save(output / f'comparison_{index:03d}.png')
    renderer.delete()
    imageio.mimwrite(output / 'comparison.mp4', movies, fps=config['fps'], macro_block_size=1)
    np.savez_compressed(output / 'render.npz', rgb=rgb_frames, mask=masks, depth=depths)
    wrist_length = (.058249 + .03425) * models['right'].wrist_ratio * scales[0] + models['right'].spacer * scales[1]
    (output / 'metrics.json').write_text(json.dumps(dict(scales=scales.tolist(), frames=len(frames), hands=metrics,
        wrist_axis_to_palm_mm=float(wrist_length * 1000), shoulder_width_m=shoulder_width,
        constant_dimensions=True, wrist_connection='Fixed flange adapter, no independent hand translation',
        limitations='Visual kinematic fitting only. No contact, self-collision or hardware-execution validation.'), indent=2))
    print(json.dumps(metrics), flush=True)


def run(source, reference, output, hands, separate=False, initial=None, wrist_ratio=1., spacer=.012, screen_fit=False):
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    models = {side: ConnectedHand(side, hand, wrist_ratio, spacer) for side, hand in hands.items()}
    config, camera, intrinsic, targets = load_targets(source, reference, models)
    (output / 'config.json').write_text(json.dumps(dict(source=str(source.resolve()), reference=str(reference.resolve()),
        source_config=config, scale_mode='separate' if separate else 'shared', scale_bounds=[.5, 1.2],
        base='One rigid pose in EgoDex world for the entire clip', wrist_adapter=dict(face_m=.03425, spacer_m=spacer, axial_ratio=wrist_ratio, rotation_y_rad=np.pi/2),
        temporal='Velocity and acceleration penalties on arm/finger joints', silhouette='Projected mesh samples against SAM distance fields',
        screen_fit=screen_fit, fixed_hand_scale=.73 if screen_fit else None,
        shoulder_spacing='Fitted constant, separate from arm length' if screen_fit else 'Native spacing scaled with arm length',
        physics=False), indent=2))
    base, scales, q, width = fit_sequence(models, targets, camera, intrinsic, np.load(source / 'human_masks.npy') > 0, initial, separate, output, screen_fit)
    render_sequence(source, reference, output, models, config, camera, intrinsic, targets, base, scales, q, width)
