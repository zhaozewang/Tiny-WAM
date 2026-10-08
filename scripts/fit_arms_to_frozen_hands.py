"""Add wrist-anchored R1 arms without refitting any accepted hand parameters."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import pyrender
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from swam.retargeting.anchored_arm import WristAnchoredArm
from swam.retargeting.geometry import SharpaHand, project
from swam.retargeting.segmentation import load_frames
from swam.retargeting.occlusion import composite
from occlude_rgb_robot_hands import sha256, panels


def sleeve_axis(mask,wrist):
    y,x = np.nonzero(mask)
    if len(x)<12:
        raise ValueError('Sleeve mask too small for forearm direction')
    points = np.stack([x,y],-1).astype(float)
    distance = np.linalg.norm(points-wrist,axis=1)
    far = points[distance>=np.quantile(distance,.55)]
    end = np.median(far,axis=0)
    direction = end-wrist
    direction /= np.linalg.norm(direction)
    return end,direction


def shaft(arm,wrist,q,k):
    links = arm.backward(wrist,q)
    start = links[arm.side+'_arm_link4']
    points = np.array([[.04,0,0],[.22,0,0]])@start[:3,:3].T+start[:3,3]
    return project(points,k)


def fit(arm,wrist,k,sleeve,target,protected,scene,renderer,previous,axis):
    end,direction = axis
    normal = np.array([-direction[1],direction[0]])
    # Both vendor arms use the same limit signs, despite mirrored link origins.
    base = np.array([0.,1.2,-1.4,0.,0.,0.])
    base = np.clip(base,arm.lower,arm.upper)
    lower,upper = arm.lower.copy(),arm.upper.copy()
    if previous is not None:
        # Visual arm-only rate cap; never relax the immutable wrist constraint.
        lower = np.maximum(lower,previous-.3)
        upper = np.minimum(upper,previous+.3)
        base = np.clip(base,lower,upper)
    anchor = project(wrist[None,:3,3],k)[0]

    def line_error(v):
        q = base.copy(); q[4:] = v
        uv = shaft(arm,wrist,q,k)
        vector = uv[0]-uv[1]
        vector /= np.linalg.norm(vector)
        return np.r_[6*(vector-direction), ((uv-end)@normal)/35,
                     np.maximum(0,-((uv-anchor)@direction))/30]

    starts = [np.array([p,r]) for p in [-1.,0.,1.] for r in [-2.,0.,2.]]
    if previous is not None:
        starts.insert(0,previous[4:])
    solutions = [least_squares(line_error,np.clip(s,lower[4:],upper[4:]),
                               bounds=(lower[4:],upper[4:]),max_nfev=60) for s in starts]
    best = min(solutions,key=lambda r:np.dot(r.fun,r.fun))
    base[4:] = best.x
    outside = distance_transform_edt(~target)
    mask_count = max(int(sleeve.sum()),1)
    calls = 0

    def evaluate(q):
        nonlocal calls
        calls += 1
        arm.show(scene,wrist,q)
        depth = renderer.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY)
        visible = (depth>0)&~protected
        out = float(np.minimum(outside[visible],70).mean()) if visible.any() else 100.
        missing = float((sleeve&~visible).sum()/mask_count)
        direction_cost = float(np.sum(line_error(q[4:])**2))
        continuity = 0. if previous is None else .5*float(np.mean((q[2:]-previous[2:])**2))
        loss = out+12*missing+.35*direction_cost+continuity
        return loss,dict(outside_px=out,uncovered_sleeve_fraction=missing,objective=loss)

    candidates = [base]
    if previous is not None:
        candidates.append(previous.copy())
    evaluated = [evaluate(q) for q in candidates]
    chosen = int(np.argmin([r[0] for r in evaluated]))
    q = candidates[chosen].copy()
    score,stats = evaluated[chosen]
    for factor in [2,1,1,.5,.25]:
        for d in [2,3,4,5]:
            candidate,best_stats = q,stats
            for sign in (-1,1):
                trial = q.copy()
                trial[d] = np.clip(trial[d]+sign*.06*factor,lower[d],upper[d])
                value,info = evaluate(trial)
                if value<score:
                    score,candidate,best_stats = value,trial,info
            q,stats = candidate,best_stats
    stats['calls'] = calls
    stats['shaft_direction_target'] = direction.tolist()
    return q,stats


def run(args):
    output = args.output
    output.mkdir(parents=True,exist_ok=False)
    accepted = ROOT/'results/rgb_robot_bottle_alignment_v3'
    source = ROOT/'results/retargeting_import_20261007/screw_unscrew_bottle_cap/source'
    hashes = {p.name:sha256(p) for p in (accepted/'native_parameters.npz',accepted/'robot_render.npz')}
    poses = np.load(accepted/'native_parameters.npz')
    original = np.load(accepted/'robot_render.npz')
    labels = np.load(ROOT/'results/rgb_robot_occlusion_v1/screw_unscrew_bottle_cap/segmentation/labels.npy')
    sleeves = np.load(ROOT/'results/rgb_bottle_sleeves_v1/labels.npy')
    frames,config = load_frames(source,True)
    if not np.array_equal(poses['indices'],np.arange(len(frames))):
        raise ValueError('Frozen hand frame ordering differs from source')
    k = np.load(ROOT/'results/rgb_robot_hands_v1/screw_unscrew_bottle_cap/qpos.npz')['intrinsic'].copy()
    k[:2] /= 2
    scene = pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.65]*3)
    scene.add(pyrender.IntrinsicsCamera(k[0,0],k[1,1],k[0,2],k[1,2],znear=.01,zfar=100),pose=np.diag([1.,-1.,-1.,1.]))
    scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=np.diag([1.,-1.,-1.,1.]))
    hands = {side:SharpaHand(side,scene) for side in ('left','right')}
    arms = {side:WristAnchoredArm(side) for side in hands}
    for arm in arms.values():
        arm.add_visuals(scene)
    renderer = pyrender.OffscreenRenderer(960,540)
    wrist_poses = {side:np.array([hand.forward(p)[1][hand.root] for p in poses[side]]) for side,hand in hands.items()}
    axes = {}
    for side,identity in [('left',1),('right',2)]:
        uv = project(wrist_poses[side][:,:3,3],k)
        endpoints = np.array([sleeve_axis(sleeves[i]==identity,uv[i])[0] for i in range(len(frames))])
        endpoints = gaussian_filter1d(endpoints,1,axis=0,mode='nearest')
        dirs = endpoints-uv
        dirs /= np.linalg.norm(dirs,axis=-1,keepdims=True)
        axes[side] = list(zip(endpoints,dirs))
    rows,qpos,bases = [],{s:[] for s in hands},{s:[] for s in hands}
    previous = {s:None for s in hands}
    rendered,render_masks = [],[]
    max_wrist_error,max_joint_error = 0.,0.
    with imageio.get_writer(output/'robot_arms.mp4',fps=config['fps'],quality=9,macro_block_size=1) as video, \
         imageio.get_writer(output/'comparison.mp4',fps=config['fps'],quality=9,macro_block_size=1) as comparison:
        for i in args.frames:
            start = time.perf_counter()
            row = {'frame':i,'source_frame':config['indices'][i]}
            for side,identity in [('left',1),('right',2)]:
                for hand in hands.values():
                    for node in hand.nodes.values(): node.mesh.is_visible=False
                for other,arm in arms.items():
                    for node in arm.nodes.values(): node.mesh.is_visible=other==side
                wrist = wrist_poses[side][i]
                target = (sleeves[i]==identity)|(labels[i]==identity+1)
                protected = original['mask'][i]|np.isin(labels[i],[1,4])
                q,stats = fit(arms[side],wrist,k,sleeves[i]==identity,target,protected,scene,renderer,previous[side],axes[side][i])
                previous[side] = q
                qpos[side].append(q)
                links = arms[side].show(scene,wrist,q)
                bases[side].append(links[side+'_arm_base_link'])
                restored = arms[side].forward(links[side+'_arm_base_link'],q)
                error = float(np.max(np.abs(restored-wrist)))
                assert error<1e-10
                max_wrist_error = max(max_wrist_error,error)
                p = poses[side][i]
                hand_points,transforms = hands[side].forward(p)
                relative = np.linalg.inv(wrist)@np.stack([transforms[n] for n in hands[side].links])
                reconstructed = restored@relative
                max_joint_error = max(max_joint_error,float(np.max(np.abs(reconstructed[:,:3,3]-hand_points))))
                row[side] = stats
            for hand in hands.values():
                for node in hand.nodes.values(): node.mesh.is_visible=False
            for arm in arms.values():
                for node in arm.nodes.values(): node.mesh.is_visible=True
            arm_rgb,arm_depth = renderer.render(scene)
            frame = np.asarray(Image.fromarray(frames[i]).resize((960,540),Image.Resampling.LANCZOS))
            objects = np.isin(labels[i],[1,4])
            before,visible_hand = composite(frame,original['rgb'][i],original['mask'][i],objects)
            # Preserve every accepted visible hand pixel; arms are a separate proximal layer.
            arm_mask = (arm_depth>0)&~objects&~original['mask'][i]
            after = np.where(arm_mask[...,None],arm_rgb,before)
            assert np.array_equal(after[visible_hand],before[visible_hand])
            assert np.array_equal(after[objects],frame[objects])
            video.append_data(after)
            comparison.append_data(np.asarray(panels([before,after],['Approved hands (unchanged)','Wrist-anchored arms'])))
            rendered.append(arm_rgb);render_masks.append(arm_mask)
            diagnostic = frame.copy()
            for side,identity,color in [('left',1,[80,160,255]),('right',2,[240,100,200])]:
                m = sleeves[i]==identity
                diagnostic[m] = (.5*diagnostic[m]+.5*np.array(color)).astype(np.uint8)
            if i in (0,8,16,24,32):
                panels([frame,before,after],['Source','Approved hands','Anchored arms']).save(output/f'comparison_{i:03d}.png')
                Image.fromarray(after).save(output/f'frame_{i:03d}.png')
                Image.fromarray(diagnostic).save(output/f'sleeves_{i:03d}.png')
            row['seconds'] = time.perf_counter()-start
            rows.append(row)
            print(json.dumps(row),flush=True)
    renderer.delete()
    assert hashes=={p.name:sha256(p) for p in (accepted/'native_parameters.npz',accepted/'robot_render.npz')}
    np.savez_compressed(output/'arms.npz',**qpos,**{s+'_base':v for s,v in bases.items()},indices=args.frames)
    np.savez_compressed(output/'arm_render.npz',rgb=rendered,mask=render_masks,indices=args.frames)
    (output/'report.json').write_text(json.dumps(dict(accepted=str(accepted),frozen_sha256=hashes,
        frames=rows,hand_parameters_modified=False,visible_hand_pixels_identical=True,
        max_wrist_transform_error=max_wrist_error,max_hand_joint_position_error=max_joint_error,
        arm_scale=1.,wrist_spacer_m=.012,arm_only_max_joint_step_rad=.3,
        method='Reverse native R1 kinematics, wrist hard-constrained, visible sleeve fit with arm-only variables',
        limitations=['Floating per-hand arm bases, not one shared fixed torso.',
                     'Virtual camera depth is not calibrated physical depth.',
                     'No collision/contact verification. Proximal layer preserves approved hand pixels.',
                     'No human-arm inpainting outside robot coverage.']),indent=2)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,nargs='+',default=list(range(33)))
    run(parser.parse_args())
