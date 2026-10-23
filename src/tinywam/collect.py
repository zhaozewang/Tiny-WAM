"""Collect physical random-action R1 trajectories with synchronous camera streams."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil

import numpy as np

from tinywam.r1 import ROOT, assemble, floor_height


def physical_scene(asset, height, device):
    import isaaclab.sim as sim
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from tinywam.workcell import spawn_workcell
    from tinywam.preview import color_hands

    context = sim.SimulationContext(sim.SimulationCfg(dt=1 / 120, device=device))
    ground = sim.GroundPlaneCfg()
    ground.func('/World/ground', ground)
    light = sim.DomeLightCfg(intensity=1800)
    light.func('/World/light', light)
    spawn_workcell(context.stage)
    cfg = ArticulationCfg(
        prim_path='/World/Robot',
        spawn=sim.UsdFileCfg(usd_path=asset,
            articulation_props=sim.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=16,
                solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, height)),
        actuators={
            'arms': ImplicitActuatorCfg(joint_names_expr=['.*_arm_joint.*'],
                                       stiffness=1000., damping=80.),
            'hands': ImplicitActuatorCfg(joint_names_expr=['.*_hand_.*'],
                                        stiffness=3., damping=.1)})
    robot = Articulation(cfg)
    context.reset()
    color_hands(context.stage)
    return context, robot


def cameras(context, views):
    import isaaclab.sim as sim
    import omni.replicator.core as rep
    result = []
    for name, eye, target, focal in views:
        path = '/World/camera_' + name
        cfg = sim.PinholeCameraCfg(focal_length=focal, horizontal_aperture=24.)
        cfg.func(path, cfg)
        context.set_camera_view(tuple(eye), tuple(target), path)
        product = rep.create.render_product(path, (256, 256))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
        rgb.attach([product])
        result.append((name, rgb, product))
    return result


def advance(context, robot, target, steps):
    robot.set_joint_position_target(target)
    for _ in range(steps):
        robot.write_data_to_sim()
        context.step(render=False)
        robot.update(1 / 120)


def record_trial(context, robot, camera_streams, center, limits, arm_mask, output, seed, views):
    import imageio.v2 as imageio
    import torch

    output.mkdir(exist_ok=False)
    rng = np.random.default_rng(seed)
    lower, upper = limits.T.copy()
    lower[arm_mask] = np.maximum(lower[arm_mask], center[arm_mask] - .15)
    upper[arm_mask] = np.minimum(upper[arm_mask], center[arm_mask] + .15)
    # Interior hand targets avoid driving directly against anatomical hard limits.
    span = upper - lower
    lower[~arm_mask] += .1 * span[~arm_mask]
    upper[~arm_mask] -= .1 * span[~arm_mask]
    initial = np.clip(center, lower, upper)
    knots = rng.uniform(lower, upper, (6, len(center)))
    knots[0] = initial
    value = torch.tensor(initial[None], dtype=torch.float32, device=robot.device)
    robot.write_joint_state_to_sim(value, torch.zeros_like(value))
    robot.reset()
    advance(context, robot, value, 240)
    for _ in range(12):
        context.render()
    arrays = {key: [] for key in ('qpos', 'qvel', 'rgb', 'actions', 'applied_torque')}
    with ExitStack() as stack:
        writers = [stack.enter_context(imageio.get_writer(output / f'{name}.mp4',
                    fps=24, codec='libx264', quality=8)) for name, _, _ in camera_streams]
        together = stack.enter_context(imageio.get_writer(output / 'synchronized.mp4',
                    fps=24, codec='libx264', quality=8))
        for t in range(121):
            for _ in range(2):
                context.render()
            qpos = robot.data.joint_pos[0].detach().cpu().numpy().copy()
            qvel = robot.data.joint_vel[0].detach().cpu().numpy().copy()
            assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
            assert np.max(np.abs(qvel)) < 100, 'Unstable joint velocity'
            images = np.stack([np.asarray(rgb.get_data())[..., :3].copy()
                               for _, rgb, _ in camera_streams])
            arrays['qpos'].append(qpos)
            arrays['qvel'].append(qvel)
            arrays['rgb'].append(images)
            arrays['applied_torque'].append(robot.data.applied_torque[0].detach().cpu().numpy().copy())
            if t < 120:
                for writer, image in zip(writers, images):
                    writer.append_data(image)
                together.append_data(np.concatenate(list(images), axis=1))
            if t in (0, 60, 119):
                imageio.imwrite(output / f'frame_{t:03d}.png', np.concatenate(list(images), axis=1))
            if t == 120:
                break
            # a_t is the actual position target held through the next five physics steps.
            phase = (t + 1) / 24
            index = min(int(phase), 4)
            fraction = phase - index
            blend = fraction * fraction * (3 - 2 * fraction)
            command = knots[index] * (1 - blend) + knots[index + 1] * blend
            arrays['actions'].append(command)
            target = torch.tensor(command[None], dtype=torch.float32, device=robot.device)
            advance(context, robot, target, 5)
            if t % 24 == 0:
                print(json.dumps(dict(trial=seed, frame=t, max_abs_velocity=float(np.abs(qvel).max()))), flush=True)
    arrays = {key: np.asarray(values) for key, values in arrays.items()}
    np.savez_compressed(output / 'trajectory.npz', **arrays,
        joint_names=robot.joint_names, timestamps=np.arange(121) / 24,
        action_timestamps=np.arange(120) / 24, limits=limits, knots=knots)
    qpos = arrays['qpos']
    group_motion = {}
    for side in ('left', 'right'):
        for group in ('arm', 'hand'):
            ids = [i for i, name in enumerate(robot.joint_names) if name.startswith(side + '_' + group)]
            group_motion[side + '_' + group] = float(np.ptp(qpos[:, ids], axis=0).max())
    excess = np.maximum(limits[:, 0] - qpos, qpos - limits[:, 1])
    summary = dict(seed=seed, frames=120, fps=24, duration_s=5, observations=121,
        actions=120, physics_steps=600, warmup_steps=240, mode='physical_joint_target_policy',
        action_definition='48 joint position targets a_t held over [t/24, (t+1)/24)',
        max_abs_velocity=float(np.abs(arrays['qvel']).max()),
        max_joint_limit_excess_rad=float(np.maximum(excess, 0).max()),
        movement_range_rad=group_motion, cameras=views,
        limitations=['No manipulation object; random robot motion only',
                    'Scaled hand retains source mass and inertia; not hardware calibrated',
                    'Implicit actuator applied_torque is an estimate, not a force sensor'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    assert min(group_motion.values()) > .01, group_motion
    assert summary['max_joint_limit_excess_rad'] < .15, summary
    return summary


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=5)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    xml = assemble(output / 'r1_isyhand.urdf', .85)
    height = floor_height(xml)
    reference = json.loads((ROOT / 'assets/arm_pose.json').read_text())
    angles = {f'{side}_arm_joint{i}': angle for side in ('left', 'right')
              for i, angle in enumerate(reference['ik'][side]['joint_angles_rad'], 1)}
    shutil.copy2(__file__, output / 'collect.py')
    launcher = AppLauncher(args)
    from tinywam.preview import convert, camera_views
    converted = convert(output)
    context, robot = physical_scene(converted.usd_path, height, args.device)
    assert robot.num_joints == 48
    limits = np.array([[float(xml.find(f"joint[@name='{name}']/limit").get(k))
                       for k in ('lower', 'upper')] for name in robot.joint_names])
    arm_mask = np.array(['_arm_joint' in n for n in robot.joint_names])
    center = np.array([angles.get(name, 0.) for name in robot.joint_names])
    views = camera_views(xml, angles, height)
    streams = cameras(context, views)
    summaries = []
    for trial in range(args.trials):
        rng = np.random.default_rng(1000 + trial)
        randomized = []
        for name, eye, target, focal in views:
            eye = np.asarray(eye) + rng.uniform(-.04, .04, 3)
            target = np.asarray(target) + rng.uniform(-.02, .02, 3)
            context.set_camera_view(tuple(eye), tuple(target), '/World/camera_' + name)
            randomized.append(dict(name=name, eye=eye.tolist(), target=target.tolist(),
                                   focal_length_mm=focal, resolution=[256, 256]))
        summary = record_trial(context, robot, streams, center, limits, arm_mask,
                               output / f'trial_{trial:02d}', 100 + trial, randomized)
        summaries.append(summary)
    (output / 'summary.json').write_text(json.dumps(dict(status='collected', trials=summaries), indent=2))
    print('COLLECTION_COMPLETE', flush=True)
    launcher.app.close()


if __name__ == '__main__':
    main()
