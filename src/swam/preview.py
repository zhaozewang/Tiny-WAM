"""Render the R1 + ISyHand assembly and check native wrist transforms."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from swam.r1 import ROOT, COMMIT, assemble, floor_height, poses, solve_pose


def convert(output):
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    return UrdfConverter(UrdfConverterCfg(
        asset_path=str(output / 'r1_isyhand.urdf'), usd_dir=str(output / 'asset'),
        usd_file_name='r1_isyhand.usd', fix_base=True, merge_fixed_joints=True,
        make_instanceable=False, force_usd_conversion=True,
        collider_type='convex_decomposition', self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=3., damping=.1))))


def color_hands(stage):
    import isaaclab.sim as sim
    from pxr import UsdShade
    for name, color in [('fingers', (.012,) * 3), ('palm', (.32,) * 3)]:
        cfg = sim.PreviewSurfaceCfg(diffuse_color=color, roughness=.55)
        cfg.func('/World/Materials/hand_' + name, cfg)
    for prim in stage.Traverse():
        name = prim.GetName()
        if name == 'visuals':
            name = prim.GetParent().GetName()
        elif not name.endswith('_hand_base_link') or '/visuals/' not in str(prim.GetPath()):
            continue
        if '_hand_' not in name:
            continue
        part = name.split('_hand_')[1]
        material = 'palm' if part in ('base_link', 'J51', 'J52') else 'fingers'
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(stage.GetPrimAtPath('/World/Materials/hand_' + material)),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants)


def scene(asset, height, device, with_table=True):
    import isaaclab.sim as sim
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from swam.workcell import spawn_workcell
    import omni.replicator.core as rep

    context = sim.SimulationContext(sim.SimulationCfg(dt=1 / 120, device=device))
    ground = sim.GroundPlaneCfg()
    ground.func('/World/ground', ground)
    light = sim.DomeLightCfg(intensity=1800)
    light.func('/World/light', light)
    table = {}
    if with_table:
        table = spawn_workcell(context.stage, include_body=False)
        table = {key: table[key] for key in ('menagerie_commit', 'table', 'table_bounds')}
    cfg = ArticulationCfg(
        prim_path='/World/Robot', spawn=sim.UsdFileCfg(usd_path=asset),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, height)),
        actuators={'joints': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=3., damping=.1)})
    robot = Articulation(cfg)
    context.reset()
    color_hands(context.stage)
    camera = sim.PinholeCameraCfg(focal_length=28., horizontal_aperture=24.)
    camera.func('/World/overview', camera)
    context.set_camera_view((2.6, -2.8, 2.0), (.35, 0, .85), '/World/overview')
    product = rep.create.render_product('/World/overview', (960, 960))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
    rgb.attach([product])
    return context, robot, rgb, table


def camera_views(xml, angles, height):
    head = poses(xml, angles)['zed_link'][:3, 3] + [0, 0, height]
    return [
        ('head', head + [.04, 0, 0], (.47, 0, .91), 18.),
        ('left', (1.15, 1.10, 1.45), (.37, 0, 1.05), 28.),
        ('right', (1.15, -1.10, 1.45), (.37, 0, 1.05), 28.),
    ]


def synchronized_cameras(context, views):
    """Attach all three cameras before rendering a shared robot state."""
    import isaaclab.sim as sim
    import omni.replicator.core as rep

    cameras = {}
    for name, eye, target, focal in views:
        path = '/World/camera_' + name
        cfg = sim.PinholeCameraCfg(focal_length=focal, horizontal_aperture=24.)
        cfg.func(path, cfg)
        context.set_camera_view(tuple(eye), target, path)
        product = rep.create.render_product(path, (960, 960))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
        rgb.attach([product])
        cameras[name] = (rgb, product)
    return cameras


def random_preview(context, robot, xml, angles, height, output, frames):
    """Render bounded random joint-pose animation, without physics integration."""
    from contextlib import ExitStack
    import imageio.v2 as imageio
    import torch

    fps = 15
    rng = np.random.default_rng(42)
    names = robot.joint_names
    center = np.array([angles.get(name, 0.) for name in names])
    limits = np.array([[float(xml.find(f"joint[@name='{name}']/limit").get(key))
                        for key in ('lower', 'upper')] for name in names])
    low, high = limits.T.copy()
    arm = np.array(['_arm_joint' in name for name in names])
    low[arm] = np.maximum(low[arm], center[arm] - .15)
    high[arm] = np.minimum(high[arm], center[arm] + .15)
    knots = rng.uniform(low, high, (int(np.ceil(frames / fps)) + 1, len(names)))
    knots[0] = np.clip(center, low, high)
    views = camera_views(xml, angles, height)
    cameras = synchronized_cameras(context, views)
    states = []
    errors = []
    with ExitStack() as stack:
        writers = {name: stack.enter_context(imageio.get_writer(
            output / f'{name}.mp4', fps=fps, codec='libx264', quality=8,
            macro_block_size=1)) for name in cameras}
        together = stack.enter_context(imageio.get_writer(
            output / 'synchronized.mp4', fps=fps, codec='libx264', quality=8,
            macro_block_size=1))
        for frame in range(frames):
            index, remainder = divmod(frame, fps)
            fraction = remainder / fps
            blend = fraction * fraction * (3 - 2 * fraction)
            q = knots[index] * (1 - blend) + knots[index + 1] * blend
            value = torch.tensor(q[None], dtype=torch.float32, device=robot.device)
            robot.write_joint_state_to_sim(value, torch.zeros_like(value))
            context.forward()
            robot.update(1 / 120)
            expected = poses(xml, dict(zip(names, q)))
            for name in robot.body_names:
                if name in expected:
                    actual = robot.data.body_pos_w[0, robot.body_names.index(name)].cpu().numpy()
                    errors.append(float(np.linalg.norm(actual - expected[name][:3, 3] - [0, 0, height])))
            # No joint state changes or physics steps between the three images.
            for _ in range(16 if frame == 0 else 2):
                context.render()
            images = []
            for name, (rgb, _) in cameras.items():
                image = np.asarray(rgb.get_data())[..., :3].copy()
                writers[name].append_data(image)
                images.append(image)
                if frame in (0, frames // 2, frames - 1):
                    imageio.imwrite(output / f'{name}_{frame:03d}.png', image)
            together.append_data(np.concatenate(images, axis=1))
            states.append(q)
            if frame % 30 == 0:
                print('Synchronized random frame', frame, flush=True)
    assert max(errors) < 1e-4, max(errors)
    np.savez(output / 'trajectory.npz', qpos=np.array(states), joint_names=names,
             timestamps=np.arange(frames) / fps, knots=knots, limits=limits)
    (output / 'three_cameras.json').write_text(json.dumps([
        dict(name=name, eye=list(eye), target=list(target), focal_length_mm=focal,
             horizontal_aperture_mm=24., resolution=[960, 960])
        for name, eye, target, focal in views], indent=2))
    return dict(mode='random_kinematic_animation', frames=frames, fps=fps,
                duration_s=frames / fps, seed=42, arm_offset_bound_rad=.15,
                hand_range='URDF joint limits', target_interval_s=1.,
                interpolation='cubic smoothstep for display', physics_steps=0,
                synchronization='three render products share each recorded joint state',
                max_body_fk_error_m=max(errors))


def render(context, robot, rgb, xml, angles, height, output, frames):
    import imageio.v2 as imageio
    import torch

    data_dir = ROOT / 'assets/hand_replay'
    data = np.load(data_dir / 'states.npz')
    hand_names = json.loads((data_dir / 'summary.json').read_text())['lab_joint_order']
    expected = poses(xml, angles)
    errors = []
    assert robot.num_joints == 48, robot.joint_names
    q = torch.zeros((1, robot.num_joints), device=robot.device)
    for name, angle in angles.items():
        q[0, robot.joint_names.index(name)] = angle
    # Arm poses remain fixed; replay hand angles only to inspect the wrist attachment.
    with imageio.get_writer(output / 'r1_isyhand.mp4', fps=15) as writer:
        for frame in range(frames):
            step = min(frame * 2, len(data['qpos']) - 1)
            for i, side in enumerate(('left', 'right')):
                for j, name in enumerate(hand_names):
                    q[0, robot.joint_names.index(side + '_hand_' + name)] = float(data['qpos'][step, i, j])
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            context.forward()
            robot.update(1 / 120)
            for side in ('left', 'right'):
                link = side + '_arm_link6'
                actual = robot.data.body_pos_w[0, robot.body_names.index(link)].cpu().numpy()
                errors.append(float(np.linalg.norm(actual - expected[link][:3, 3] - [0, 0, height])))
            for _ in range(8 if frame == 0 else 2):
                context.render()
            image = np.asarray(rgb.get_data())[..., :3].copy()
            writer.append_data(image)
            if frame in (0, frames // 2):
                imageio.imwrite(output / f'preview_{frame:03d}.png', image)
            if frame % 30 == 0:
                print('Frame', frame, flush=True)
    # Inspect the actual attachment at close range, using the same recorded pose.
    context.set_camera_view((1.25, -1.1, 1.4), (.38, 0, 1.04), '/World/overview')
    for _ in range(8):
        context.render()
    imageio.imwrite(output / 'wrist_closeup.png', np.asarray(rgb.get_data())[..., :3])
    # Three views of the identical pose, before hiding the table for video references.
    from pxr import UsdGeom
    head = expected['zed_link'][:3, 3] + [0, 0, height]
    camera = UsdGeom.Camera(context.stage.GetPrimAtPath('/World/overview'))
    views = camera_views(xml, angles, height)
    for name, eye, target, focal in views:
        camera.GetFocalLengthAttr().Set(focal)
        context.set_camera_view(tuple(eye), target, '/World/overview')
        for _ in range(16):
            context.render()
        imageio.imwrite(output / f'view_{name}.png', np.asarray(rgb.get_data())[..., :3])
    (output / 'three_cameras.json').write_text(json.dumps([
        dict(name=name, eye=list(eye), target=list(target), focal_length_mm=focal,
             resolution=[960, 960], table_visible=True)
        for name, eye, target, focal in views], indent=2))
    # A head-mounted reference isolates the actual robot appearance for video editing.
    from pxr import UsdGeom
    if context.stage.GetPrimAtPath('/World/Table'):
        context.stage.RemovePrim('/World/Table')
    UsdGeom.Camera(context.stage.GetPrimAtPath('/World/overview')).GetFocalLengthAttr().Set(18.)
    UsdGeom.Imageable(context.stage.GetPrimAtPath('/World/ground')).MakeInvisible()
    head = expected['zed_link'][:3, 3] + [0, 0, height]
    context.set_camera_view(tuple(head + [.04, 0, 0]), (.43, 0, .97), '/World/overview')
    for _ in range(8):
        context.render()
    imageio.imwrite(output / 'ego_reference.png', np.asarray(rgb.get_data())[..., :3])
    (output / 'ego_camera.json').write_text(json.dumps({'eye': (head + [.04, 0, 0]).tolist(),
                                                     'target': [.43, 0, .97], 'table_visible': False}))
    assert max(errors) < 1e-4, max(errors)
    normals = {side: expected[side + '_hand_base_link'][2, 2] for side in ('left', 'right')}
    assert max(normals.values()) < -.99, normals
    return {'max_wrist_fk_error_m': max(errors), 'palm_normal_z': normals}


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--hand-scale', type=float, default=.85)
    parser.add_argument('--pose-reference', type=Path, default=ROOT / 'assets/arm_pose.json')
    parser.add_argument('--no-table', action='store_true')
    parser.add_argument('--random-preview', action='store_true',
                        help='Synchronized three-view random arm/hand pose animation (no physics)')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert 0 < args.hand_scale <= 1.
    xml = assemble(output / 'r1_isyhand.urdf', args.hand_scale)
    height = floor_height(xml)
    if args.pose_reference:
        reference = json.loads(args.pose_reference.read_text())
        ik = reference['ik']
        angles = {}
        for side in ('left', 'right'):
            for i, angle in enumerate(ik[side]['joint_angles_rad'], 1):
                angles[f'{side}_arm_joint{i}'] = angle
        ik = {'reference': str(args.pose_reference), 'reference_ik': ik}
    else:
        angles, ik = solve_pose(xml, height)
    print('IK', ik, flush=True)
    for source in [Path(__file__), ROOT / 'src/swam/r1.py',
                   ROOT / 'src/swam/workcell.py']:
        shutil.copy2(source, output / source.name)

    launcher = AppLauncher(args)
    converter = convert(output)
    context, robot, rgb, table = scene(converter.usd_path, height, args.device, with_table=not args.no_table)
    if args.random_preview:
        checks = random_preview(context, robot, xml, angles, height, output, args.frames)
    else:
        checks = render(context, robot, rgb, xml, angles, height, output, args.frames)
    result = dict(source_commit=COMMIT, source_urdf='R1/urdf/r1_v2_1_0.urdf',
                  status='kinematic_assembly_checked', total_dof=48, arm_dof_each=6,
                  hand_dof_each=18, base_height_m=height, fixed='base, wheels and torso',
                  hand_scale=args.hand_scale,
                  handedness={'right': 'source hand', 'left': 'XZ-reflected geometry, frames, axes and inertia'},
                  mount_xyz_m={side: [.03425 + .025 * args.hand_scale - .001,
                                    sign * .021 * args.hand_scale, -.03375 * args.hand_scale]
                               for side, sign in [('left', 1), ('right', -1)]},
                  ik=ik, table=table, **checks,
                  limitations=['Kinematic visualization; no physics stepping or control policy test',
                               'Left hand is a mathematical mirror of the supplied right-hand asset',
                               'Scaled visual spacer; physical adapter and payload unverified',
                               'Scaled geometry retains original mass/inertia; appearance preview only',
                               'Collision-free motion and dynamic actuator settings not validated'])
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    launcher.app.close()


if __name__ == '__main__':
    main()
