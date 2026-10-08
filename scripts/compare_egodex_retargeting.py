"""Run unmodified upstream retargeters on a fixed-size R1 + Sharpa pilot."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrender
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch
import trimesh
from swam.retargeting.geometry import SharpaHand
from swam.retargeting.connected import ConnectedHand, tensor, initial_base, transform, rotation
from swam.r1 import SOURCE, origin_matrix

SIDES = ('left', 'right')
TIPS = [20, 4, 8, 12, 16]
BASES = [19, 3, 7, 11, 15]


def rigid_fit(a, b):
    u, _, vt = np.linalg.svd((a-a.mean(0)).T @ (b-b.mean(0)))
    r = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, b.mean(0) - r @ a.mean(0)
    return t


def write_robot(model, base, destination):
    """Export precisely the FK used by our renderer, without visual mesh loading."""
    robot = ET.Element('robot', name=model.side + '_r1_sharpa')
    ET.SubElement(robot, 'link', name='world')

    def add_joint(name, parent, child, pose, axis=None, bounds=None):
        ET.SubElement(robot, 'link', name=child)
        joint = ET.SubElement(robot, 'joint', name=name, type='fixed' if axis is None else 'revolute')
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'origin', xyz=' '.join(map(str, pose[:3, 3])),
                      rpy=' '.join(map(str, Rotation.from_matrix(pose[:3, :3]).as_euler('xyz'))))
        if axis is not None:
            ET.SubElement(joint, 'axis', xyz=' '.join(map(str, axis)))
            ET.SubElement(joint, 'limit', lower=str(bounds[0]), upper=str(bounds[1]), effort='100', velocity='10')

    parent = model.side + '_arm_base_link'
    add_joint('mount_to_world', 'world', parent, base @ model.base)
    for i, (name, origin, axis) in enumerate(model.chain):
        add_joint(model.names[i], parent, name, origin.numpy(), axis.numpy(), (model.lower[i], model.upper[i]))
        parent = name
    mount = np.eye(4)
    mount[:3, :3] = model.mount_rotation.numpy()
    mount[0, 3] = .03425 + model.spacer
    add_joint('hand_mount', parent, model.hand.root, mount)
    for i, (parent, child, origin, index, axis) in enumerate(model.hand.chain):
        name = model.hand.names[index] if index is not None else 'fixed_hand_' + str(i)
        add_joint(name, parent, child, origin, axis if index is not None else None,
                  (model.hand.lower[index], model.hand.upper[index]) if index is not None else None)
    ET.indent(robot)
    ET.ElementTree(robot).write(destination, encoding='unicode')


def validate_fk(model, robot, adaptor, base):
    rng = np.random.default_rng(7)
    maximum = 0.
    for _ in range(5):
        q = rng.uniform(model.lower, model.upper)
        robot.compute_forward_kinematics(adaptor.forward_qpos(q))
        _, _, links = model.forward(tensor(base), tensor([1., 1.]), tensor(q[None]))
        for name in model.hand.links:
            maximum = max(maximum, float(np.max(np.abs(robot.get_frame_pose(name) - links[name][0].numpy()))))
    if maximum > 1e-8:
        raise ValueError(f'FK mismatch {maximum}')
    return maximum


def arm_ik(robot, adaptor, model, wrists, seed):
    result = []
    q = seed.copy()
    rng = np.random.default_rng(2026)
    for frame, wrist in enumerate(wrists):
        previous = q[:6].copy()

        def residual(arm):
            full = q.copy()
            full[:6] = arm
            robot.compute_forward_kinematics(adaptor.forward_qpos(full))
            pose = robot.get_frame_pose(model.hand.root)
            angular = Rotation.from_matrix(wrist[:3, :3].T @ pose[:3, :3]).as_rotvec()
            return np.r_[(pose[:3, 3] - wrist[:3, 3]) * 30, angular, (arm-previous) * .02]

        seeds = [q[:6]]
        if frame == 0:
            seeds += [rng.uniform(model.lower[:6], model.upper[:6]) for _ in range(8)]
        fits = [least_squares(residual, s, bounds=(model.lower[:6], model.upper[:6]), max_nfev=200) for s in seeds]
        q[:6] = min(fits, key=lambda x: x.cost).x
        result.append(q.copy())
    return np.asarray(result)


def paper_retargeter(model, robot, adaptor, urdf, initial, ablation=0):
    from retargeting.config.core import RobotConfig, RetargetingProfileConfig, RetargetingConfig
    from retargeting.core.retargeter import Retargeter
    links = model.hand.links
    tips = [links[i] for i in TIPS]
    pairs = [['world', tips[0]]] + [[model.hand.root, n] for n in tips]
    pairs += [[tips[0], n] for n in tips[1:]] + [[links[b], links[t]] for b, t in zip(BASES, TIPS)]
    robot_dict = dict(name=model.side + '_r1_sharpa', model=dict(type='urdf', path=str(urdf)),
        actuated_joints=model.names + model.hand.names, initial_qpos=initial.tolist(),
        visual_frame_names=list(dict.fromkeys(robot.frame_names)), wrist_frame_name=model.hand.root, human_hand_scale=1.,
        benchmark=dict(wrist_link_name=model.hand.root, fingertips=[dict(link_name=links[t], human_tip_index=t,
            human_direction_base_index=b, robot_direction_axis='z', is_thumb=i == 0)
            for i, (t, b) in enumerate(zip(TIPS, BASES))]))
    # Preserve published objective weights; only embodiment-specific joint weights are adapted.
    profile = dict(name='r1_sharpa_pilot', robot=str(urdf), method='vector_wrist_joint',
        target=dict(wrist_link_name=model.hand.root, link_pairs=pairs), objective={},
        retargeting=dict(arm_dof=6, human_wrist_index=0,
            joint_position_weights=[0.] * len(initial), joint_velocity_weights=[.1]*6 + [.01]*(len(initial)-6)),
        joint_limit_overrides=[])
    method = dict(type='VECTOR_WRIST_JOINT', ablation_option=ablation,
                  optimizer=dict(class_='VectorWristJointOptimizerV2', params=dict(huber_delta=.02)))
    method['optimizer']['class'] = method['optimizer'].pop('class_')
    retargeter = Retargeter(adaptor, RobotConfig.from_dict(robot_dict),
                            RetargetingProfileConfig.from_dict(profile), RetargetingConfig.from_dict(method))
    retargeter.optimizer.set_joint_limit(retargeter.optimizer.joint_limits, epsilon=0.)
    return retargeter, dict(robot=robot_dict, profile=profile, method=method)


def solve_side(model, targets, wrists, base, old_q, output):
    from retargeting.core.kinematics.pinocchio_model import RobotPinocchio
    from retargeting.core.kinematics.adaptor import RobotAdaptor
    from retargeting.core.types import RetargetingHandObservation
    from dex_retargeting.retargeting_config import RetargetingConfig
    urdf = output / (model.side + '.urdf')
    write_robot(model, base, urdf)
    robot = RobotPinocchio(str(urdf), 'urdf')
    adaptor = RobotAdaptor(robot, model.names + model.hand.names)
    fk_error = validate_fk(model, robot, adaptor, base)
    started = time.perf_counter()
    arm = arm_ik(robot, adaptor, model, wrists, np.clip(old_q[0], model.lower, model.upper))
    print(model.side, 'arm IK', round(time.perf_counter()-started, 2), 's; FK error', fk_error, flush=True)
    solutions, timing, configs = {}, {}, {}
    for method in ('position', 'vector', 'dexpilot'):
        config = dict(type=method, urdf_path=str(urdf), target_joint_names=model.hand.names,
                      low_pass_alpha=1., scaling_factor=1.)
        if method == 'position':
            config.update(target_link_names=[model.hand.links[i] for i in TIPS], target_link_human_indices=TIPS)
        elif method == 'vector':
            config.update(target_origin_link_names=[model.hand.root]*5,
                target_task_link_names=[model.hand.links[i] for i in TIPS], target_link_human_indices=[[0]*5, TIPS])
        else:
            config.update(wrist_link_name=model.hand.root, finger_tip_link_names=[model.hand.links[i] for i in TIPS])
        sequence = RetargetingConfig.from_dict(copy.deepcopy(config)).build()
        sequence.optimizer.set_joint_limit(sequence.joint_limits, epsilon=0.)
        names = sequence.optimizer.robot.dof_joint_names
        assert names == robot.dof_joint_names, 'Upstream joint order differs; explicit remapping required'
        order = [names.index(n) for n in adaptor.actuated_joints_name]
        sequence.set_qpos(adaptor.forward_qpos(arm[0]))
        result = []
        started = time.perf_counter()
        for frame, target in enumerate(targets):
            if method == 'position':
                values = target[TIPS]
            elif method == 'vector':
                values = target[TIPS] - target[0]
            else:
                # DexPilot expects wrist, thumb, index, middle, ring, little ordering.
                canonical = target[[0] + list(range(17, 21)) + list(range(1, 17))]
                idx = sequence.optimizer.target_link_human_indices
                values = canonical[idx[1]] - canonical[idx[0]]
            fixed = adaptor.forward_qpos(arm[frame])[sequence.optimizer.idx_pin2fixed]
            q = sequence.retarget(values, fixed_qpos=fixed)[order]
            result.append(q)
        solutions[method] = np.asarray(result)
        timing[method] = time.perf_counter()-started
        configs[method] = config
        print(model.side, method, round(timing[method], 3), 's', flush=True)
    for label, ablation in [('paper_full', 0), ('paper_actual_distance', 2)]:
        sequence, config = paper_retargeter(model, robot, adaptor, urdf, arm[0], ablation)
        result = []
        started = time.perf_counter()
        for frame, (target, wrist) in enumerate(zip(targets, wrists)):
            local = (target-wrist[:3, 3]) @ wrist[:3, :3]
            observation = RetargetingHandObservation(local, wrist, handedness=model.side)
            # Upstream returns float32; roundoff can put the next initial guess
            # outside strict float64 bounds. Project only the optimizer warm start.
            previous = np.clip(sequence.previous_qpos.astype(np.float64), model.lower+1e-8, model.upper-1e-8)
            result.append(sequence.solve(observation, previous_qpos=previous).qpos)
        solutions[label] = np.asarray(result)
        timing[label] = time.perf_counter()-started
        configs[label] = config
        print(model.side, label, round(timing[label], 3), 's', flush=True)
    (output / (model.side + '_configs.json')).write_text(json.dumps(configs, indent=2))
    return solutions, dict(fk_max_abs_error=fk_error, solve_seconds=timing)


def metrics(points, target, q, model, fps):
    tip_error = np.linalg.norm(points[:, TIPS]-target[:, TIPS], axis=-1)*1000
    pv = points[:, TIPS[1:]] - points[:, TIPS[:1]]
    tv = target[:, TIPS[1:]] - target[:, TIPS[:1]]
    pd = points[:, TIPS] - points[:, BASES]
    td = target[:, TIPS] - target[:, BASES]
    cosine = (pd*td).sum(-1) / np.maximum(np.linalg.norm(pd, axis=-1)*np.linalg.norm(td, axis=-1), 1e-10)
    return dict(tip_mm=float(tip_error.mean()), tip_p95_mm=float(np.percentile(tip_error, 95)),
        thumb_relative_mm=float(np.linalg.norm(pv-tv, axis=-1).mean()*1000),
        fingertip_direction_deg=float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean()),
        joint_step_p95_rad=float(np.percentile(np.abs(np.diff(q, axis=0)), 95)),
        limit_excess_rad=float(max(0., (model.lower-q).max(), (q-model.upper).max())))


def renderer_for(models, intrinsic, size):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[.65]*3)
    camera = np.diag([1., -1., -1., 1.])
    scene.add(pyrender.IntrinsicsCamera(intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2], znear=.01), pose=camera)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2), pose=camera)
    nodes = {}
    for side, model in models.items():
        for name, node in model.hand.nodes.items():
            scene.add_node(node)
            nodes[name] = node
        for link in model.robot.findall('link'):
            name = link.get('name')
            if name not in [side + '_arm_base_link'] + [side + f'_arm_link{i}' for i in range(1, 7)]:
                continue
            visual = link.find('visual')
            geo = trimesh.load(SOURCE.parent.parent/'meshes'/Path(visual.find('geometry/mesh').get('filename')).name, force='mesh')
            geo.apply_transform(origin_matrix(visual.find('origin')))
            material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[.3, .32, .35, 1], metallicFactor=.3, roughnessFactor=.6)
            nodes[name] = scene.add(pyrender.Mesh.from_trimesh(geo, material=material))
    return scene, nodes, pyrender.OffscreenRenderer(*size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', default='use_chopsticks')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=33)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = ROOT/'results/retargeting_import_20261007'/args.clip/'source'
    old = source.parent/'connected'
    config = json.loads((source/'prepare.json').read_text())
    indices = config['indices'][:args.frames]
    staging = pyrender.Scene()
    models = {s: ConnectedHand(s, SharpaHand(s, staging)) for s in SIDES}
    targets, wrists = {}, {}
    with h5py.File(source/'source.hdf5') as data:
        cameras = np.linalg.inv(data['transforms/camera'][indices])
        for s, model in models.items():
            points = np.stack([data['transforms/'+n][indices, :3, 3] for n in model.hand.labels], axis=1)
            arms = np.stack([data['transforms/'+s+n][indices, :3, 3] for n in ['Arm', 'Forearm', 'Hand']], axis=1)
            confidence = np.stack([data['confidences/'+n][indices] for n in model.hand.labels], axis=1)
            targets[s] = dict(hand=points, arm=tensor(arms), confidence=confidence)
            neutral, _ = model.hand.forward(np.r_[np.zeros(6), np.clip(np.zeros(len(model.hand.names)), model.hand.lower, model.hand.upper)])
            anchors = [0, 1, 5, 9, 13]
            wrists[s] = np.stack([rigid_fit(neutral[anchors], p[anchors]) for p in points])
    base_values = initial_base(targets)
    base = transform(rotation(tensor(base_values[:3])), tensor(base_values[3:])).numpy()
    original_q = np.load(old/'parameters.npz')
    all_solutions, diagnostics = {}, {}
    for s in SIDES:
        all_solutions[s], diagnostics[s] = solve_side(models[s], targets[s]['hand'], wrists[s], base, original_q[s], args.output)
    np.savez_compressed(args.output/'solutions.npz', base=base, camera=cameras,
        **{f'{s}_{m}': q for s, methods in all_solutions.items() for m, q in methods.items()})
    report = dict(clip=args.clip, source=str(source), indices=indices, fps=config['fps'], frames=len(indices),
        physical_validation=False, robot_scale=[1., 1.], wrist_ratio=1., spacer_m=.012,
        wrist_alignment='Per-frame rigid palm landmark registration; no scale fit',
        base_alignment='Constant mean human shoulders; native robot shoulder width',
        adaptations='Sharpa landmark mapping, R1 URDF, zero posture weights, arm/finger velocity weights .1/.01; upstream objectives unchanged; bound epsilon set to zero',
        limitations=['No object geometry, object contact, collision constraints, inpainting or object occlusion',
                     'Archived baseline has different scales and is qualitative only; native methods share geometry and calibration',
                     'Unweighted human-keypoint metrics are not ground-truth grasp success; annotation confidence retained separately'],
        diagnostics=diagnostics, metrics={}, versions={})
    for name in ('dex-retargeting', 'objective-retargeting'):
        report['versions'][name] = subprocess.check_output(['git', '-C', str(Path.home()/'.cache/swam-retarget'/name), 'rev-parse', 'HEAD'], text=True).strip()
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    np.savez_compressed(args.output/'targets.npz', **{f'{s}_{k}': np.asarray(v) for s, d in targets.items() for k, v in d.items()})
    predictions = {}
    for method in all_solutions['left']:
        predictions[method], report['metrics'][method] = {}, {}
        for s, model in models.items():
            q = all_solutions[s][method]
            points, _, links = model.forward(tensor(base), tensor([1., 1.]), tensor(q))
            predictions[method].update({n: v.numpy() for n, v in links.items()})
            report['metrics'][method][s] = metrics(points.numpy(), targets[s]['hand'], q, model, config['fps'])
    report['metrics']['archived_scaled'] = {}
    old_base = transform(rotation(tensor(original_q['base'][:3])), tensor(original_q['base'][3:]))
    for s in SIDES:
        model = ConnectedHand(s, models[s].hand, wrist_ratio=.4, spacer=0.)
        q = original_q[s][:len(indices)]
        width = tensor(float(original_q['shoulder_width'])) if 'shoulder_width' in original_q else None
        points, _, _ = model.forward(old_base, tensor(original_q['scales']), tensor(q), width)
        report['metrics']['archived_scaled'][s] = metrics(points.numpy(), targets[s]['hand'], q, model, config['fps'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    # Decode the original pixels instead of upscaling the old 512px preview.
    size = (1024, 576)
    reader = imageio.get_reader(source/'source.mp4')
    frames = [np.asarray(Image.fromarray(reader.get_data(i)).crop(config['crop']).resize(size, Image.Resampling.LANCZOS)) for i in indices]
    reader.close()
    k = np.asarray(config['intrinsic'])
    k[:2] *= 2
    for node in list(staging.nodes):
        staging.remove_node(node)
    scene, nodes, renderer = renderer_for(models, k, size)
    overlays = {}
    for method, links in predictions.items():
        images = []
        with imageio.get_writer(args.output/(method+'.mp4'), fps=config['fps'], macro_block_size=1, quality=8) as writer:
            for i, frame in enumerate(frames):
                for name, poses in links.items():
                    if name in nodes:
                        scene.set_pose(nodes[name], cameras[i] @ poses[i])
                rgb, depth = renderer.render(scene)
                overlay = np.where((depth > 0)[..., None], rgb, frame)
                images.append(overlay)
                writer.append_data(overlay)
        overlays[method] = images
        print('Rendered', method, flush=True)
    renderer.delete()
    original = np.load(old/'render.npz')
    low_frames = np.load(source/'source_frames.npy')[:len(indices)]
    old_images = [np.asarray(Image.fromarray(np.where(m[..., None], rgb, frame)).resize(size))
                  for rgb, m, frame in zip(original['rgb'], original['mask'], low_frames)]
    imageio.mimwrite(args.output/'source.mp4', frames, fps=config['fps'], macro_block_size=1, quality=8)
    panels = [('Source', frames), ('Archived fit (scaled)', old_images), ('Position (native size)', overlays['position']),
              ('DexPilot (native size)', overlays['dexpilot']), ('Paper full (native size)', overlays['paper_full']),
              ('Paper actual distance (native size)', overlays['paper_actual_distance'])]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    with imageio.get_writer(args.output/'comparison.mp4', fps=config['fps'], macro_block_size=1, quality=8) as writer:
        for i in range(len(indices)):
            canvas = Image.new('RGB', (1536, 640), (245, 245, 245))
            draw = ImageDraw.Draw(canvas)
            for p, (label, images) in enumerate(panels):
                x, y = (p%3)*512, (p//3)*320
                draw.text((x+8, y+5), label, font=font, fill=(20, 20, 20))
                canvas.paste(Image.fromarray(images[i]).resize((512, 288)), (x, y+32))
            writer.append_data(np.asarray(canvas))
            if i in [0, len(indices)//2, len(indices)-1]:
                canvas.save(args.output/f'comparison_{i:03d}.png')
    # A fixed right-hand crop across the whole clip; never change crops per method.
    points = targets['right']['hand']
    cam_points = points @ cameras[:, :3, :3].transpose(0, 2, 1) + cameras[:, None, :3, 3]
    projected = cam_points @ k.T
    pixels = projected[..., :2] / projected[..., 2:]
    lo = np.maximum(np.floor(pixels.reshape(-1, 2).min(0)-60).astype(int), [0, 0])
    hi = np.minimum(np.ceil(pixels.reshape(-1, 2).max(0)+60).astype(int), size)
    crop = (*lo.tolist(), *hi.tolist())
    detail_panels = [('Source', frames), ('Archived fit (scaled)', old_images),
                     ('Paper: actual distance', overlays['paper_actual_distance'])]
    with imageio.get_writer(args.output/'right_hand_detail.mp4', fps=config['fps'], macro_block_size=1, quality=8) as writer:
        for i in range(len(indices)):
            canvas = Image.new('RGB', (1536, 544), (245, 245, 245))
            draw = ImageDraw.Draw(canvas)
            for p, (label, images) in enumerate(detail_panels):
                draw.text((p*512+8, 5), label, font=font, fill=(20, 20, 20))
                patch = Image.fromarray(images[i]).crop(crop)
                patch.thumbnail((512, 512))
                canvas.paste(patch, (p*512+(512-patch.width)//2, 32+(512-patch.height)//2))
            writer.append_data(np.asarray(canvas))
            if i in [0, len(indices)//2, len(indices)-1]:
                canvas.save(args.output/f'detail_{i:03d}.png')
    print(json.dumps(report['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
