"""Parallel uninterrupted physical trajectories, chunked only for lossless storage."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np

from swam.r1 import ROOT, assemble, floor_height
from swam.shards import atomic_json


from swam.parallel_scene import make_scene
from swam.collection_contract import original_commands, metadata as sampling_metadata


def policy(seed, center, limits, arm, duration, names):
    initial, knots, commands = original_commands(seed, center, limits, arm, duration)
    return initial.astype('float32'), commands, dict(seed=seed,
        family='original_smoothstep_1s', contract=sampling_metadata(),
        knots=knots.tolist())


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
    parser.add_argument('--assets', type=Path, default=ROOT / 'results/original_random_5s_512')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless, args.enable_cameras = True, True
    import xml.etree.ElementTree as ET
    xml = ET.parse(args.assets / 'r1_isyhand.urdf').getroot()
    assert args.seconds > 0 and args.seconds % 5 == 0
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
    context, robot, streams, origins = make_scene(str(args.assets/'asset/r1_isyhand.usd'), height, len(args.ids), args.device, partitions=True, solver_position_iterations=64, solver_velocity_iterations=8)
    from swam.preview import camera_views
    import torch
    torch.set_num_threads(4)
    names = robot.joint_names
    limits = np.array([[float(xml.find(f"joint[@name='{name}']/limit").get(k))
                       for k in ('lower','upper')] for name in names])
    arm = np.array(['_arm_joint' in name for name in names])
    center = np.array([angles.get(name,0.) for name in names])
    policies = [policy(100+i, center, limits, arm, args.seconds, names) for i in args.ids]
    commands = np.stack([p[1] for p in policies], axis=1)
    initial = torch.tensor(np.stack([p[0] for p in policies]), device=args.device)
    runtime_limits=robot.root_physx_view.get_dof_limits().cpu().numpy()
    assert np.allclose(runtime_limits,limits[None],atol=1e-6)
    from pxr import Usd,PhysxSchema
    authored=[]
    for prim in Usd.PrimRange(context.stage.GetPrimAtPath('/World/envs/env_0/Robot')):
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            api=PhysxSchema.PhysxArticulationAPI(prim)
            authored.append([api.GetSolverPositionIterationCountAttr().Get(),api.GetSolverVelocityIterationCountAttr().Get()])
    assert authored and all(v==[64,8] for v in authored), authored
    limit_tensor=torch.tensor(limits,device=args.device,dtype=torch.float32)
    peak_excess=torch.zeros(len(folders),device=args.device)
    peak_velocity=torch.zeros_like(peak_excess)
    frame=-1
    def advance(context,robot,target,steps):
        robot.set_joint_position_target(target)
        for _ in range(steps):
            robot.write_data_to_sim();context.step(render=False);robot.update(1/120)
            q=robot.data.joint_pos;v=robot.data.joint_vel
            excess=torch.maximum(limit_tensor[:,0]-q,q-limit_tensor[:,1]).clamp_min(0)
            peak_excess.copy_(torch.maximum(peak_excess,excess.amax(1)))
            peak_velocity.copy_(torch.maximum(peak_velocity,v.abs().amax(1)))
        if not bool(torch.isfinite(peak_excess).all() & torch.isfinite(peak_velocity).all()) or float(peak_excess.max())>=.15 or float(peak_velocity.max())>=100:
            evidence=dict(max_joint_limit_excess_rad=peak_excess.cpu().tolist(),max_abs_velocity=peak_velocity.cpu().tolist(),ids=args.ids,observation_frame=frame)
            atomic_json(args.root/'rejection.json',evidence)
            raise RuntimeError('Physics substep validation failed: '+json.dumps(evidence))
    # Reset exactly once, before warm-up. Never reset within a trajectory or block.
    robot.write_joint_state_to_sim(initial, torch.zeros_like(initial))
    robot.reset()
    advance(context, robot, initial, 240)
    views = camera_views(xml, angles, height)
    camera_records = [[] for _ in folders]
    camera_rngs=[np.random.default_rng(1000+i) for i in args.ids]
    for m,(name, eye,target,focal) in enumerate(views):
        eyes, targets = [], []
        for e,identifier in enumerate(args.ids):
            rng = camera_rngs[e]
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
            fps=24, duration_s=args.seconds, resolution=[512,512], joint_names=names,
            limits=limits.tolist(), policy=item[2], cameras=cameras,
            action_definition='48 joint position targets, held for 5 physics steps after the observation',
            reset_count=1, physics_dt=1/120, source_inertia_unscaled=True,
            physics_only=args.physics_only, solver_iterations=[64,8], hand_damping=.1,
            protocol='original_actions_solver64_v1', visual_partitions=True,
            camera_clipping_m=list(streams[0].cfg.spawn.clipping_range), environment_spacing_m=8.,
            initial_joint_positions=item[0].tolist()))
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
    with ThreadPoolExecutor(max_workers=4) as executor:
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
                    elapsed_s=time.monotonic()-start,max_joint_limit_excess_rad=float(peak_excess.max()),updated=time.time()))
                print(json.dumps(dict(frame=frame,total=total,ids=args.ids,elapsed_s=time.monotonic()-start)),flush=True)
            if frame < total:
                advance(context,robot,torch.tensor(commands[frame],device=args.device,dtype=torch.float32),5)
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
            motion_range_rad=groups,wall_seconds=time.monotonic()-start,physics_audit_hz=120,max_joint_limit_excess_rad=float(peak_excess[e]),max_abs_velocity=float(peak_velocity[e])))
    atomic_json(progress,dict(stage='complete',ids=args.ids,updated=time.time()))
    print('BATCH_COMPLETE',flush=True)
    import sys
    sys.stdout.flush();sys.stderr.flush();os._exit(0)


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

