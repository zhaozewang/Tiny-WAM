"""Closed-loop evaluation against recorded reaching trajectories."""
import json
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from tinywam.paths import asset_filename
from tinywam.shards import atomic_json, sha256


def trials(settings):
    yield 'replay', 0, settings.noise_seed
    yield 'hold', 0, settings.noise_seed
    for steps in settings.steps:
        yield f'feedback_{steps}', steps, settings.noise_seed
    steps = settings.ablation_steps
    yield f'feedback_{steps}_seed{settings.extra_noise_seed}', steps, settings.extra_noise_seed
    yield f'feedback_nogoal_{steps}', steps, settings.noise_seed


def status(output, stage, **values):
    atomic_json(output / 'status.json', dict(stage=stage, updated=time.time(), **values))


def validate_inputs(experiment):
    paths = experiment.paths
    for name in ('checkpoint', 'reference', 'tokenizers', 'image_model', 'scene_assets'):
        path = getattr(paths, name)
        if path is None or not path.exists():
            raise FileNotFoundError(f'Missing paths.{name}: {path}')
    for relative in ('r1_isyhand.urdf', 'asset/r1_isyhand.usd'):
        if not (paths.scene_assets / relative).is_file():
            raise FileNotFoundError(paths.scene_assets / relative)
    if paths.output.exists():
        raise FileExistsError(f'Choose a new evaluation output: {paths.output}')


class Scene:
    """Own the simulator state, camera poses and five-step control interval."""

    def __init__(self, assets, device):
        from pxr import Usd, PhysxSchema
        from tinywam.r1 import floor_height
        from tinywam.collect_long import make_scene

        # Resolve relocated meshes without changing the recorded scene settings.
        xml = ET.parse(assets / 'r1_isyhand.urdf').getroot()
        for mesh in xml.findall('.//mesh'):
            mesh.set('filename', asset_filename(mesh.get('filename')))
        height = floor_height(xml)
        usd = str(assets / 'asset/r1_isyhand.usd')
        self.context, self.robot, self.streams, _ = make_scene(usd, height, 1, device)
        self.names = self.robot.joint_names
        self.limits = self.robot.root_physx_view.get_dof_limits()[0].to(device)

        # Reject a USD whose articulation settings differ from the v5 recordings.
        self.iterations = []
        root = self.context.stage.GetPrimAtPath('/World/envs/env_0/Robot')
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                continue
            api = PhysxSchema.PhysxArticulationAPI(prim)
            position = api.GetSolverPositionIterationCountAttr().Get()
            velocity = api.GetSolverVelocityIterationCountAttr().Get()
            self.iterations.append([position, velocity])
        assert self.iterations and all(pair == [32, 8] for pair in self.iterations)

    def set_cameras(self, cameras):
        from pxr import Gf, UsdGeom
        for camera in cameras:
            eye = Gf.Vec3d(*camera['eye'])
            target = Gf.Vec3d(*camera['target'])
            pose = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
            path = '/World/envs/env_0/camera_' + camera['name']
            transform = UsdGeom.Xformable(self.context.stage.GetPrimAtPath(path))
            transform.ClearXformOpOrder()
            transform.AddTransformOp().Set(pose)

    def render(self):
        import torch
        self.context.render()
        self.context.render()
        for camera in self.streams:
            camera.update(1 / 24, force_recompute=True)
        images = [camera.data.output['rgb'][0, ..., :3] for camera in self.streams]
        return torch.stack(images).cpu().numpy().copy()

    def reset(self, qpos, qvel):
        root = self.robot.data.default_root_state.clone()
        self.robot.write_root_state_to_sim(root)
        self.robot.write_joint_state_to_sim(qpos[None], qvel[None])
        self.robot.reset()
        self.robot.set_joint_position_target(qpos[None])
        self.robot.write_data_to_sim()
        self.robot.update(1 / 120)

    def advance(self, command):
        import torch
        self.robot.set_joint_position_target(command[None])
        peak = velocity = 0.
        for _ in range(5):
            self.robot.write_data_to_sim()
            self.context.step(render=False)
            self.robot.update(1 / 120)
            qpos = self.robot.data.joint_pos[0]
            qvel = self.robot.data.joint_vel[0]
            assert torch.isfinite(qpos).all() and torch.isfinite(qvel).all()
            excess = torch.maximum(self.limits[:, 0] - qpos, qpos - self.limits[:, 1])
            peak = max(peak, float(excess.clamp_min(0).max()))
            velocity = max(velocity, float(qvel.abs().max()))
        if peak > .15 or velocity > 100:
            raise RuntimeError(f'Physics invalid: excess {peak}, velocity {velocity}')
        return peak, velocity


class Predictor:
    """Accumulate actual observations and predict the next committed command."""

    def __init__(self, model, codec, settings, condition, steps, seed, no_goal):
        import torch
        self.model, self.codec, self.settings = model, codec, settings
        self.goals, self.goal_times, self.camera, _ = condition
        if no_goal:
            self.goals = self.goal_times = None
        self.steps = steps
        self.generator = torch.Generator(device=settings.device).manual_seed(seed)
        self.history = self.engine = None
        self.latencies = []
        self.setup_seconds = []

    def predict(self, rgb, qpos, qvel, command, frame):
        import torch
        from tinywam.flow_sampling import next_frame

        # Encode only the actual current observation and the already committed action.
        tick = time.perf_counter()
        pixels = torch.from_numpy(rgb[self.camera].copy())
        state = torch.cat((qpos, qvel))[None]
        current = self.codec.encode(pixels, state, command[None])
        times = torch.arange(frame + 2, device=self.settings.device).float()[None] / 24
        if self.settings.backend == 'reference':
            if self.history is None:
                self.history = current
            else:
                self.history = tuple(torch.cat((h, x), 1) for h, x in zip(self.history, current))
            result = next_frame(self.model, self.history, times, self.goals, self.goal_times,
                                task='full', steps=self.steps, generator=self.generator)
        else:
            result, setup = self.cached_prediction(current, times, frame)
            tick += setup

        # Keep decoding on the device; timing ends at the same synchronization point.
        checkpoint = self.codec.checkpoints['action']
        latent = result[2] * checkpoint['latent_scale'] + checkpoint['latent_mean']
        decoded = self.codec.models['action'].decoder(latent)
        next_command = (decoded * checkpoint['scale'] + checkpoint['mean'])[0, 0]
        assert torch.isfinite(next_command).all()
        torch.cuda.synchronize()
        self.latencies.append(time.perf_counter() - tick)
        return next_command

    def cached_prediction(self, current, times, frame):
        import torch
        from tinywam.flow_inference import FlowInference
        if self.engine is None:
            self.engine = FlowInference(self.model, self.goals, self.goal_times,
                                        task='full', max_frames=121)
        self.engine.append(current, times[:, -2:-1])

        # Capture once; report setup separately from steady-state prediction latency.
        setup = 0.
        use_graph = self.settings.backend == 'cuda_graph'
        if frame == 0 and use_graph:
            torch.cuda.synchronize()
            tick = time.perf_counter()
            self.engine.capture_append()
            self.engine.capture(self.steps)
            torch.cuda.synchronize()
            setup = time.perf_counter() - tick
            self.setup_seconds.append(setup)
        result = self.engine.predict(times[:, -1:], steps=self.steps,
                                     generator=self.generator, use_graph=use_graph)
        return result, setup


def select_condition(sample, settings):
    import torch
    from tinywam.flow_tasks import prepare_window
    values = {}
    for name, value in sample['latents'].items():
        values[name] = value[None].to(settings.device)
    generator = torch.Generator(device=settings.device).manual_seed(settings.goal_seed)
    inputs, _, metadata = prepare_window(**values, task='full', generator=generator)
    valid = inputs['goal_valid'][0]
    indices = metadata['goal_indices'][0, valid]
    return (inputs['goals'][:, valid], inputs['goal_times'][:, valid],
            int(metadata['camera'][0]), indices)


def comparison_frame(rgb, recorded, label, mode, frame, error):
    from PIL import Image, ImageDraw
    caption = f'{label} | {mode} | t={frame / 24:.2f}s | final-pose RMSE={error:.3f} rad (not a goal)'
    reference = 'Recorded trajectory (comparison only; future video is NOT model input)'
    rows = []
    for pixels, text in ((rgb, caption), (recorded, reference)):
        combined = np.concatenate(list(pixels), axis=1)
        image = Image.fromarray(combined)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 768, 18), fill='black')
        draw.text((4, 3), text, fill='white')
        rows.append(np.asarray(image))
    return np.concatenate(rows, axis=0)


class Trial:
    """One reset, synchronized video and 120 control intervals."""

    def __init__(self, scene, predictor, sample, label, mode, path, output):
        self.scene, self.predictor = scene, predictor
        self.raw, self.label, self.mode = sample['raw'], label, mode
        self.path, self.output = path, output
        self.qpos, self.qvel, self.actions = [], [], []
        self.peak = self.target_peak = self.velocity = 0.
        self.failure = None
        device = predictor.settings.device
        self.goal = self.raw['qpos'][-1].to(device).float()
        self.command = self.raw['actions'][0].to(device).float()

    def run(self):
        import imageio.v2 as imageio
        device = self.predictor.settings.device
        qpos = self.raw['qpos'][0].to(device).float()
        qvel = self.raw['qvel'][0].to(device).float()
        self.scene.reset(qpos, qvel)
        for _ in range(4):
            self.scene.render()
        self.path.mkdir()
        writer = imageio.get_writer(self.path / 'comparison.mp4', fps=24, codec='libx264', quality=8)
        try:
            for frame in range(121):
                self.observe(frame, writer)
        except (RuntimeError, AssertionError) as error:
            self.failure = str(error)
            (self.path / 'failure.txt').write_text(traceback.format_exc())
            if self.mode == 'replay':
                raise
        finally:
            writer.close()

    def observe(self, frame, writer):
        import imageio.v2 as imageio
        qpos = self.scene.robot.data.joint_pos[0].clone()
        qvel = self.scene.robot.data.joint_vel[0].clone()
        self.qpos.append(qpos.cpu().numpy())
        self.qvel.append(qvel.cpu().numpy())
        rgb = self.scene.render()
        error = float((qpos - self.goal).square().mean().sqrt())
        recorded = np.asarray(self.raw['rgb'][frame])
        picture = comparison_frame(rgb, recorded, self.label, self.mode, frame, error)
        if frame in (0, 60, 120):
            imageio.imwrite(self.path / f'frame_{frame:03d}.png', picture)
        if frame > 0:
            writer.append_data(picture)
        if frame == 120:
            return
        self.control(rgb, qpos, qvel, frame)
        if frame % 24 == 0:
            status(self.output, 'running', sample=self.label, mode=self.mode,
                   frame=frame, goal_rmse_rad=error)

    def control(self, rgb, qpos, qvel, frame):
        import torch
        next_command = None
        if self.mode.startswith('feedback') and frame < 119:
            next_command = self.predictor.predict(rgb, qpos, qvel, self.command, frame)
        if self.mode == 'replay':
            self.command = self.raw['actions'][frame].to(self.predictor.settings.device).float()

        # Execute a_t before installing the prediction a_(t+1).
        self.actions.append(self.command.cpu().numpy())
        limits = self.scene.limits
        excess = torch.maximum(limits[:, 0] - self.command, self.command - limits[:, 1])
        self.target_peak = max(self.target_peak, float(excess.clamp_min(0).max()))
        peak, velocity = self.scene.advance(self.command)
        self.peak = max(self.peak, peak)
        self.velocity = max(self.velocity, velocity)
        if next_command is not None:
            self.command = next_command


def rmse(actual, target):
    return float(np.sqrt(np.mean((actual - target) ** 2)))


def trial_metrics(trial, sample, condition, steps, seed):
    settings = trial.predictor.settings
    raw = sample['raw']
    qpos = np.array(trial.qpos)
    goal = trial.goal.cpu().numpy()
    initial_error = rmse(qpos[0], goal)
    final_error = rmse(qpos[-1], goal)
    latencies = trial.predictor.latencies
    row = dict(split=sample['metadata']['split'], sample=trial.label,
               episode=sample['episode'], start=sample['start'], mode=trial.mode,
               steps=steps, noise_seed=seed, goal_seed=settings.goal_seed, camera=condition[2],
               completed=trial.failure is None, failure=trial.failure, frames=len(qpos),
               initial_goal_rmse_rad=initial_error, final_goal_rmse_rad=final_error,
               error_reduction_fraction=1 - final_error / initial_error,
               max_joint_limit_excess_rad=trial.peak, max_command_limit_excess_rad=trial.target_peak,
               max_abs_velocity=trial.velocity, inference_backend=settings.backend,
               cache_setup_seconds=trial.predictor.setup_seconds)
    row['mean_encode_predict_decode_ms'] = float(np.mean(latencies) * 1000) if latencies else None
    row['prediction_ms_p95'] = float(np.percentile(latencies, 95) * 1000) if latencies else None

    # Separate supplied waypoint errors from trajectory imitation and final-pose errors.
    indices = condition[3].cpu()
    row['waypoints'] = []
    for index in indices:
        frame = int(index)
        if frame >= len(qpos):
            continue
        error = rmse(qpos[frame], raw['qpos'][frame].numpy())
        row['waypoints'].append(dict(frame=frame, time_seconds=frame / 24, qpos_rmse_rad=error))
    errors = [waypoint['qpos_rmse_rad'] for waypoint in row['waypoints']]
    row['mean_waypoint_rmse_rad'] = float(np.mean(errors)) if errors else None
    row['trajectory_qpos_rmse_rad'] = rmse(qpos, raw['qpos'][:len(qpos)].numpy())
    actions = np.array(trial.actions)
    row['recorded_action_rmse_rad'] = rmse(actions[1:], raw['actions'][1:len(actions)].numpy())
    add_group_metrics(row, qpos, goal, raw, indices, trial.scene.names)
    if trial.mode == 'replay':
        error = rmse(qpos, raw['qpos'].numpy())
        row['recording_replay_qpos_rmse_rad'] = error
        assert error < .02, f'Replay mismatch {error}: scene must match before policy evaluation'
    return row


def add_group_metrics(row, qpos, goal, raw, indices, names):
    row['groups'], row['waypoint_groups'] = {}, {}
    for group in ('left_arm', 'left_hand', 'right_arm', 'right_hand'):
        joints = [i for i, name in enumerate(names) if name.startswith(group)]
        row['groups'][group] = rmse(qpos[-1, joints], goal[joints])
        errors = []
        for index in indices:
            frame = int(index)
            if frame < len(qpos):
                target = raw['qpos'][frame, joints].numpy()
                errors.append(rmse(qpos[frame, joints], target))
        row['waypoint_groups'][group] = errors


def save_trajectory(trial, condition):
    indices = condition[3].cpu()
    np.savez_compressed(trial.path / 'trajectory.npz',
                        qpos=np.array(trial.qpos), qvel=np.array(trial.qvel),
                        actions=np.array(trial.actions), goal=trial.goal.cpu().numpy(),
                        goal_frames=indices.numpy(), goal_qpos=trial.raw['qpos'][indices].numpy(),
                        joint_names=np.array(trial.scene.names),
                        pipeline_seconds=trial.predictor.latencies)


def protocol(experiment, scene):
    import torch
    from tinywam import collect_long
    paths, settings = experiment.paths, experiment.evaluate
    return dict(source_step=experiment.benchmark.expected_step,
                checkpoint=str(paths.checkpoint), checkpoint_sha256=sha256(paths.checkpoint),
                reference=str(paths.reference), reference_sha256=sha256(paths.reference),
                script_sha256=sha256(Path(__file__)), scene_sha256=sha256(Path(collect_long.__file__)),
                device=torch.cuda.get_device_name(), inference_backend=settings.backend,
                sampling_steps=list(settings.steps), trials=[],
                condition=f'2–3 timed qpos goals sampled by prepare_window(seed={settings.goal_seed})',
                protocol='Actual accumulated RGB/qpos/qvel and committed actions. Supplied a0; predict a_(t+1) from o_t and a_t, execute a_t, then install a_(t+1). Imagined RGB/proprio are discarded.',
                physics=dict(dt=1 / 120, control_hz=24, solver_iterations=scene.iterations,
                             hand_damping=.3, contact_offset=.001),
                latency_excludes=['rendering and physics', 'CUDA graph capture, reported separately'],
                limitations=['Synchronous simulated time, not a real-time deadline test',
                             'First five seconds of selected reference episodes, not dataset-wide success',
                             'One common goal/camera draw; paired flow counts, extra seed and no-goal ablation'])


def evaluate_sample(experiment, scene, model, codec, sample, label, report):
    import torch
    settings, output = experiment.evaluate, experiment.paths.output
    assert sample['metadata']['joint_names'] == scene.names
    condition = select_condition(sample, settings)
    report['goal_frames'] = condition[3].cpu().tolist()
    report['camera'] = condition[2]
    atomic_json(output / 'protocol.json', report)
    scene.set_cameras(sample['metadata']['cameras'])

    # Retain the original goal-pose reset/render sequence before trial resets.
    goal = sample['raw']['qpos'][-1].to(settings.device).float()
    scene.reset(goal, torch.zeros_like(goal))
    for _ in range(6):
        scene.render()
    for mode, steps, seed in trials(settings):
        status(output, 'running', split=label, mode=mode, frame=0)
        predictor = Predictor(model, codec, settings, condition, steps, seed, 'nogoal' in mode)
        path = output / f'{label}_{mode}'
        trial = Trial(scene, predictor, sample, label, mode, path, output)
        trial.run()
        row = trial_metrics(trial, sample, condition, steps, seed)
        save_trajectory(trial, condition)
        report['trials'].append(row)
        atomic_json(output / 'summary.json', report)
        print('RESULT', json.dumps(row), flush=True)


def evaluate(experiment):
    import torch
    from tinywam.flow_inference import frozen_inference_model
    from .codec import InputCodec, load_model
    settings, paths = experiment.evaluate, experiment.paths
    torch.set_num_threads(4)
    status(paths.output, 'loading_scene')
    scene = Scene(paths.scene_assets, settings.device)

    # Load the exact checkpoint and frozen modality normalizers.
    status(paths.output, 'loading_models')
    model = load_model(paths.checkpoint, experiment.benchmark.expected_step, settings.device)
    if settings.backend != 'reference':
        model = frozen_inference_model(model)
    reference = torch.load(paths.reference, weights_only=False)
    codec = InputCodec(reference, paths.tokenizers, paths.image_model, settings.device)
    for checkpoint in codec.checkpoints.values():
        assert checkpoint['joint_names'] == scene.names
    report = protocol(experiment, scene)
    atomic_json(paths.output / 'protocol.json', report)
    for label in settings.splits or list(reference['samples']):
        sample = reference['samples'][label]
        evaluate_sample(experiment, scene, model, codec, sample, label, report)
    status(paths.output, 'complete', trials=len(report['trials']))


def run(experiment):
    validate_inputs(experiment)
    from isaaclab.app import AppLauncher
    output = experiment.paths.output
    output.mkdir(parents=True, exist_ok=False)
    (output / 'experiment.toml').write_bytes(experiment.source.read_bytes())
    launcher = None
    try:
        launcher = AppLauncher(headless=True, enable_cameras=True, device=experiment.evaluate.device)
        import torch
        with torch.inference_mode():
            evaluate(experiment)
    except BaseException:
        error = traceback.format_exc()
        (output / 'failure.txt').write_text(error)
        status(output, 'failed', error=error)
        raise
    finally:
        if launcher is not None:
            launcher.app.close()
