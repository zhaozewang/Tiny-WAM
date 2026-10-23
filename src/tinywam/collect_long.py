"""Parallel uninterrupted physical trajectories, chunked only for lossless storage."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np

from tinywam.r1 import ROOT, assemble, floor_height
from tinywam.shards import atomic_json


def make_scene(asset, height, count, device):
    import torch
    import isaaclab.sim as sim
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import TiledCamera, TiledCameraCfg
    from isaacsim.core.cloner import GridCloner
    from pxr import Sdf
    from tinywam.workcell import spawn_workcell
    from tinywam.preview import color_hands

    context = sim.SimulationContext(sim.SimulationCfg(dt=1 / 120, device=device))
    ground = sim.GroundPlaneCfg()
    ground.func('/World/ground', ground)
    light = sim.DomeLightCfg(intensity=1800)
    light.func('/World/light', light)
    cloner = GridCloner(spacing=8.)
    cloner.define_base_env('/World/envs')
    paths = cloner.generate_paths('/World/envs/env', count)
    context.stage.DefinePrim(paths[0], 'Xform')
    spawn_workcell(context.stage)
    layer = context.stage.GetRootLayer()
    Sdf.CopySpec(layer, '/World/Table', layer, paths[0] + '/Table')
    context.stage.RemovePrim('/World/Table')
    cfg = ArticulationCfg(prim_path='/World/envs/env_.*/Robot',
        spawn=sim.UsdFileCfg(usd_path=asset,
            articulation_props=sim.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=32,
                solver_velocity_iteration_count=8)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, height)),
        actuators={
            'arms': ImplicitActuatorCfg(joint_names_expr=['.*_arm_joint.*'], stiffness=1000., damping=80.),
            'hands': ImplicitActuatorCfg(joint_names_expr=['.*_hand_.*'], stiffness=3., damping=.3)})
    robot = Articulation(cfg)
    # Collision meshes are nested USD instances; schema setters skip instances.
    from pxr import UsdPhysics, PhysxSchema
    from collections import Counter
    for prim in list(context.stage.Traverse()):
        if prim.IsInstance() and '/collisions' in str(prim.GetPath()):
            prim.SetInstanceable(False)
    before=Counter()
    after=Counter()
    for prim in context.stage.Traverse():
        if str(prim.GetPath()).startswith(paths[0]+'/Robot') and prim.HasAPI(UsdPhysics.CollisionAPI):
            api=PhysxSchema.PhysxCollisionAPI.Apply(prim)
            before[str(api.GetContactOffsetAttr().Get())]+=1
            api.CreateContactOffsetAttr(.001)
            api.CreateRestOffsetAttr(0.)
            after[str(api.GetContactOffsetAttr().Get())]+=1
    assert sum(after.values()) > 0, 'No editable robot colliders'
    print('COLLISION_OFFSETS',json.dumps(dict(before=before,after=after)),flush=True)
    offsets = cloner.clone(paths[0], paths, replicate_physics=True, copy_from_source=True)
    cloner.filter_collisions(context.cfg.physics_prim_path, '/World/collisions', paths, ['/World/ground'])
    streams = []
    for name, focal in [('head', 18.), ('left', 28.), ('right', 28.)]:
        camera = TiledCamera(TiledCameraCfg(prim_path='/World/envs/env_.*/camera_' + name,
            height=256, width=256, data_types=['rgb'],
            spawn=sim.PinholeCameraCfg(focal_length=focal, horizontal_aperture=24., clipping_range=(.01, 5.))))
        streams.append(camera)
    context.reset()
    color_hands(context.stage)
    assert robot.num_instances == count, (robot.num_instances, count)
    return context, robot, streams, torch.tensor(offsets, dtype=torch.float32, device=device)


def policy(seed, center, limits, arm, duration, names):
    rng = np.random.default_rng(seed)
    low, high = limits.T.copy()
    amplitude = float(rng.uniform(.08, .15))
    low[arm] = np.maximum(low[arm], center[arm] - amplitude)
    high[arm] = np.minimum(high[arm], center[arm] + amplitude)
    span = high - low
    low[~arm] += .1 * span[~arm]
    high[~arm] -= .1 * span[~arm]
    for j,name in enumerate(names):
        if arm[j]:
            continue
        joint=name.rsplit('_',1)[-1]
        if joint in ('J51','J52'):
            low[j],high[j]=-.08,.08
        elif joint.endswith('4'):
            low[j],high[j]=(-.25,.25) if joint=='J24' else (-.10,.10)
        else:
            low[j],high[j]=.05,1.0
    initial = np.clip(center, low, high)
    knot_times, targets = [0.], [initial]
    while knot_times[-1] < duration:
        interval = float(rng.uniform(1., 3.))
        target = rng.uniform(low, high)
        # Occasional holds expose motion-to-rest transitions without resets.
        if rng.random() < .15:
            target = targets[-1].copy()
        knot_times.append(knot_times[-1] + interval)
        targets.append(target)
    knot_times, targets = np.asarray(knot_times), np.asarray(targets)
    times = (np.arange(int(duration * 24)) + 1) / 24
    indices = np.searchsorted(knot_times, times, side='right') - 1
    phase = (times - knot_times[indices]) / (knot_times[indices+1] - knot_times[indices])
    blend = phase * phase * (3 - 2 * phase)
    commands = targets[indices] * (1-blend[:, None]) + targets[indices+1] * blend[:, None]
    return initial.astype('float32'), commands.astype('float32'), dict(
        seed=seed, family='conservative_independent_joint_targets_v1', target_lower=low.tolist(), target_upper=high.tolist(), knot_interval_s=[1.,3.], arm_amplitude_rad=amplitude, knot_times=knot_times.tolist(), targets=targets.tolist())


def save_block(folder, block, observations, commands, start):
    arrays = {key: np.stack([x[key] for x in observations]) for key in ('rgb', 'qpos', 'qvel')}
    arrays['actions'] = commands
    arrays['frame_indices'] = np.arange(start, start + len(observations))
    path = folder / f'block_{block:05d}.npz'
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    path.with_suffix('.tmp').replace(path)


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ids', type=int, nargs='+', required=True)
    parser.add_argument('--physics-only', action='store_true', help='Diagnostic only: no image dataset is produced')
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--assets', type=Path, default=ROOT / 'results/long_dataset_assets')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless, args.enable_cameras = True, True
    args.assets.mkdir(parents=True, exist_ok=True)
    xml = assemble(args.assets / 'r1_isyhand.urdf', .85)
    height = floor_height(xml)
    reference = json.loads((ROOT / 'assets/arm_pose.json').read_text())
    angles = {f'{side}_arm_joint{i}': q for side in ('left','right')
              for i,q in enumerate(reference['ik'][side]['joint_angles_rad'],1)}
    folders = [args.root / 'staging' / f'episode_{i:05d}' for i in args.ids]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=False)
    progress = args.root / 'worker_status.json'
    atomic_json(progress, dict(stage='initializing', ids=args.ids, updated=time.time()))
    launcher = AppLauncher(args)
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    converter = UrdfConverter(UrdfConverterCfg(asset_path=str(args.assets / 'r1_isyhand.urdf'),
        usd_dir=str(args.assets / 'asset'), usd_file_name='r1_isyhand.usd', fix_base=True,
        merge_fixed_joints=True, make_instanceable=False, force_usd_conversion=False,
        collider_type='convex_decomposition', self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=3., damping=.3))))
    context, robot, streams, origins = make_scene(converter.usd_path, height, len(args.ids), args.device)
    from tinywam.collect import advance
    from tinywam.preview import camera_views
    import torch
    names = robot.joint_names
    limits = np.array([[float(xml.find(f"joint[@name='{name}']/limit").get(k))
                       for k in ('lower','upper')] for name in names])
    arm = np.array(['_arm_joint' in name for name in names])
    center = np.array([angles.get(name,0.) for name in names])
    policies = [policy(710000+i, center, limits, arm, args.seconds, names) for i in args.ids]
    commands = np.stack([p[1] for p in policies], axis=1)
    initial = torch.tensor(np.stack([p[0] for p in policies]), device=args.device)
    # Reset exactly once, before warm-up. Never reset within a trajectory or block.
    robot.write_joint_state_to_sim(initial, torch.zeros_like(initial))
    robot.reset()
    advance(context, robot, initial, 240)
    views = camera_views(xml, angles, height)
    camera_records = [[] for _ in folders]
    for m,(name, eye,target,focal) in enumerate(views):
        eyes, targets = [], []
        for e,identifier in enumerate(args.ids):
            rng = np.random.default_rng(810000+identifier*3+m)
            local_eye = np.asarray(eye) + rng.uniform(-.04,.04,3)
            local_target = np.asarray(target) + rng.uniform(-.02,.02,3)
            eyes.append(local_eye); targets.append(local_target)
            camera_records[e].append(dict(name=name, eye=local_eye.tolist(), target=local_target.tolist(), focal_length_mm=focal))
        # Set a local USD transform explicitly; viewport helpers in this build
        # mishandle camera transforms below translated environment parents.
        from pxr import Gf, UsdGeom
        for e in range(len(folders)):
            prim=context.stage.GetPrimAtPath(f'/World/envs/env_{e}/camera_{name}')
            pose=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eyes[e]),Gf.Vec3d(*targets[e]),Gf.Vec3d(0,0,1)).GetInverse()
            xf=UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(pose)
    for folder, identifier, item, cameras in zip(folders, args.ids, policies, camera_records):
        split = 'validation' if identifier % 14 == 12 else 'test' if identifier % 14 == 13 else 'train'
        atomic_json(folder/'metadata.json', dict(episode=identifier, split=split,
            fps=24, duration_s=args.seconds, resolution=[256,256], joint_names=names,
            limits=limits.tolist(), policy=item[2], cameras=cameras,
            action_definition='48 joint position targets, held for 5 physics steps after the observation',
            reset_count=1, physics_dt=1/120, source_inertia_unscaled=True,
            physics_only=args.physics_only, solver_iterations=[32,8], hand_damping=.3, contact_offset_m=.001,
            camera_clipping_m=[.01,5.], environment_spacing_m=8.))
    for _ in range(12):
        context.render()
    from pxr import UsdGeom
    cache = UsdGeom.XformCache()
    camera_debug=[]
    for prim in context.stage.Traverse():
        if prim.GetTypeName() == 'Camera':
            transform=cache.GetLocalToWorldTransform(prim)
            camera_debug.append(dict(path=str(prim.GetPath()),matrix=[list(row) for row in transform]))
    for e, records in enumerate(camera_records):
        for record in records:
            prim=context.stage.GetPrimAtPath(f'/World/envs/env_{e}/camera_'+record['name'])
            pose=cache.GetLocalToWorldTransform(prim)
            expected=np.asarray(record['eye'])+origins[e].cpu().numpy()
            assert np.allclose(np.asarray(pose.ExtractTranslation()),expected,atol=1e-5), 'Camera pose mismatch'
    atomic_json(args.root/'camera_debug.json',dict(origins=origins.cpu().tolist(),
        root_positions=robot.data.root_pos_w.cpu().tolist(),cameras=camera_debug))
    buffers = [[] for _ in folders]
    pending = []
    total = args.seconds * 24
    start = time.monotonic()
    ranges_min = np.full((len(folders),48),np.inf)
    ranges_max = np.full((len(folders),48),-np.inf)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for frame in range(total+1):
            if not args.physics_only:
                context.render()
                context.render()
                images = []
                for camera in streams:
                    camera.update(1/24, force_recompute=True)
                    images.append(camera.data.output['rgb'][..., :3].cpu().numpy().copy())
                rgb = np.stack(images, axis=1)
                if frame == 0:
                    variation=rgb.astype('float32').std(axis=(2,3)).mean(-1)
                    atomic_json(args.root/'image_check.json',dict(pixel_std=variation.tolist()))
                    assert variation.min() > 2, 'Camera produced a blank image; reject batch'
            else:
                rgb = np.zeros((len(folders),3,1,1,3),dtype=np.uint8)
            qpos = robot.data.joint_pos.cpu().numpy().copy()
            qvel = robot.data.joint_vel.cpu().numpy().copy()
            assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
            assert np.abs(qvel).max() < 100, 'Unstable simulation'
            excess = np.maximum(limits[:,0]-qpos,qpos-limits[:,1])
            if excess.max() >= .15:
                env, joint = np.unravel_index(excess.argmax(), excess.shape)
                atomic_json(args.root/'rejection.json', dict(frame=frame,episode=args.ids[env],
                    joint=names[joint],qpos=float(qpos[env,joint]),limits=limits[joint].tolist(),
                    excess=float(excess.max())))
                raise RuntimeError('Joint limit violation: '+json.dumps(json.loads((args.root/'rejection.json').read_text())))
            ranges_min = np.minimum(ranges_min,qpos); ranges_max = np.maximum(ranges_max,qpos)
            for e in range(len(folders)):
                buffers[e].append(dict(rgb=rgb[e],qpos=qpos[e],qvel=qvel[e]))
            if frame and frame % 120 == 0:
                for task in pending:
                    task.result()
                pending = []
                block = frame//120-1
                for e,folder in enumerate(folders):
                    pending.append(executor.submit(save_block,folder,block,buffers[e],commands[frame-120:frame,e],frame-120))
                    buffers[e] = buffers[e][-1:]
                atomic_json(progress,dict(stage='collecting',ids=args.ids,frame=frame,total_frames=total,
                    elapsed_s=time.monotonic()-start,updated=time.time()))
                print(json.dumps(dict(frame=frame,total=total,ids=args.ids,elapsed_s=time.monotonic()-start)),flush=True)
            if frame < total:
                advance(context,robot,torch.tensor(commands[frame],device=args.device),5)
        for task in pending:
            task.result()
    for e,folder in enumerate(folders):
        movement = ranges_max[e]-ranges_min[e]
        groups = {}
        for side in ('left','right'):
            for part in ('arm','hand'):
                group = [j for j,n in enumerate(names) if n.startswith(side+'_'+part)]
                groups[side+'_'+part] = float(movement[group].max())
        assert min(groups.values()) > .01, groups
        metadata=json.loads((folder/'metadata.json').read_text())
        atomic_json(folder/'complete.json',dict(episode=args.ids[e],split=metadata['split'],fps=24,
            duration_s=args.seconds,actions=total,observations=total+1,blocks=total//120,block_actions=120,
            motion_range_rad=groups,wall_seconds=time.monotonic()-start))
    atomic_json(progress,dict(stage='complete',ids=args.ids,updated=time.time()))
    print('BATCH_COMPLETE',flush=True)
    launcher.app.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import sys, traceback
        error=traceback.format_exc()
        traceback.print_exc()
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1])
            atomic_json(root/'worker_status.json',dict(stage='failed',error=error,updated=time.time()))
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)  # Isaac teardown may hang after an exception.

