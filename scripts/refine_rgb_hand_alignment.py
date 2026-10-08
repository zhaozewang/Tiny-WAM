"""Experimental visible-silhouette refinement of frozen RGB hand retargeting.

Uses source RGB instance masks, not new physical depth/contact supervision.
Keeps the original render and poses untouched. All errors against WiLoR are
pseudo-target errors, not human annotation accuracy.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import pyrender
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from swam.retargeting.geometry import SharpaHand, project
from swam.retargeting.occlusion import composite
from swam.retargeting.segmentation import load_frames
from occlude_rgb_robot_hands import panels, sha256

ORDER = [0]+list(range(5,21))+list(range(1,5))
TIPS = [4,8,12,16,20]


def native_parameters(q):
    return np.r_[Rotation.from_euler('XYZ',q[3:6]).as_rotvec(),q[:3],q[6:]]


def refine(hand, scene, renderer, initial, target_uv, labels, identity, k, sweeps, proposal=None):
    height, width = labels.shape
    points, _ = hand.forward(initial)
    before_uv = project(points,k)
    wrist = target_uv[0]
    direction = target_uv[[1,5,9,13]].mean(0)-wrist
    direction /= np.linalg.norm(direction)
    yy,xx = np.indices(labels.shape)
    # SAM includes the forearm, which the hand mesh is not supposed to replace.
    region = ((np.stack([xx,yy],-1)-wrist)@direction > -12)
    target = (labels==identity)&region
    objects = np.isin(labels,[1,4])
    outside = distance_transform_edt(~target)
    hit_xy = np.round(target_uv).astype(int)
    hit_xy[:,0] = hit_xy[:,0].clip(0,width-1)
    hit_xy[:,1] = hit_xy[:,1].clip(0,height-1)
    visible_targets = (outside[hit_xy[:,1],hit_xy[:,0]]<5)
    weights = np.full(21,.04)
    weights[TIPS] = np.where(visible_targets[TIPS],1.,.05)
    weights[0] = 0
    lower = np.r_[np.full(6,-np.inf),hand.lower]
    upper = np.r_[np.full(6,np.inf),hand.upper]
    # Native robot dimensions stay fixed; depth is a virtual-camera parameter.
    pixels_per_unit = k[0,0]/initial[5]
    steps = np.r_[np.full(3,.025),[2/pixels_per_unit,2/pixels_per_unit,.04],
                  np.full(len(hand.names),.035)]
    calls = 0

    def evaluate(p):
        nonlocal calls
        calls += 1
        hand.show(scene,p)
        visible = (renderer.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY)>0)&~objects&region
        union = (visible|target).sum()
        iou = float((visible&target).sum()/max(union,1))
        out = float(np.minimum(outside[visible],20).mean()) if visible.any() else 20.
        points,_ = hand.forward(p)
        uv = project(points,k)
        errors = np.linalg.norm(uv-target_uv,axis=-1)
        anchoring = float((weights*errors).sum()/weights.sum())
        prior = float(((p[6:]-initial[6:])**2).mean())
        # Visible-mask agreement addresses palm placement absent from fingertip fitting.
        loss = 20*(1-iou)+out+.3*anchoring+.5*prior
        return loss,dict(iou=iou,outside_px=out,visible_landmark_px=anchoring,
                         all_landmarks_px=float(errors.mean()),objective=loss)

    q = initial.copy()
    score,before = evaluate(q)
    after = before
    if proposal is not None:
        candidate = np.clip(proposal,lower,upper)
        value,stats = evaluate(candidate)
        if value < score:
            q,score,after = candidate,value,stats
    for factor in sweeps:
        for d,step in enumerate(steps*factor):
            candidate = q
            candidate_info = after
            for sign in (-1,1):
                trial = q.copy()
                trial[d] = np.clip(trial[d]+sign*step,lower[d],upper[d])
                value,stats = evaluate(trial)
                if value < score:
                    score,candidate,candidate_info = value,trial,stats
            q,after = candidate,candidate_info
    hand.show(scene,q)
    return q,dict(before=before,after=after,calls=calls,
                  initial_keypoints_uv=before_uv.tolist(),target_uv=target_uv.tolist(),
                  visible_pseudo_targets=visible_targets.tolist())


def run(args):
    clip = 'screw_unscrew_bottle_cap'
    base = ROOT/'results/rgb_robot_hands_v1'/clip
    segmentation = ROOT/'results/rgb_robot_occlusion_v1'/clip/'segmentation'
    source = ROOT/'results/retargeting_import_20261007'/clip/'source'
    records = json.loads((ROOT/'results/rgb_hands_wilor_v2'/clip/'predictions.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    frames,config = load_frames(source,True)
    labels = np.load(segmentation/'labels.npy')
    frozen = np.load(base/'qpos.npz')
    proposals = None
    if args.corrections_from:
        proposed = np.load(args.corrections_from/'native_parameters.npz')
        if not np.array_equal(proposed['indices'],np.arange(len(frames))):
            raise ValueError('Temporal correction requires every source frame in order')
        proposals = {}
        for side in ('left','right'):
            initial = np.array([native_parameters(q) for q in frozen[side]])
            # Smooth only the added correction, not the approved source motion.
            delta = gaussian_filter1d(proposed[side]-initial,sigma=1.,axis=0,mode='nearest')
            proposals[side] = initial+delta
    original = np.load(base/'robot_render.npz')
    hashes = {p.name:sha256(p) for p in (base/'qpos.npz',base/'robot_render.npz')}
    k = frozen['intrinsic'].copy()
    k[:2] /= 2
    scene = pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.65]*3)
    hands = {s:SharpaHand(s,scene) for s in ('left','right')}
    scene.add(pyrender.IntrinsicsCamera(k[0,0],k[1,1],k[0,2],k[1,2],znear=.01,zfar=100),pose=np.diag([1.,-1.,-1.,1.]))
    scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=np.diag([1.,-1.,-1.,1.]))
    renderer = pyrender.OffscreenRenderer(960,540)
    allstats, poses, rgbs, masks = [],{s:[] for s in hands},[],[]
    for i in args.frames:
        start = time.perf_counter()
        stats = {'index':i,'source_index':config['indices'][i]}
        for side,hand in hands.items():
            for other,h in hands.items():
                for node in h.nodes.values():
                    node.mesh.is_visible = other==side
            prediction = next(p for p in records[i]['wilor'] if int(p['is_right'])==int(side=='right'))
            uv = np.array(prediction['wilor_preds']['pred_keypoints_2d'])[0,ORDER]/2
            initial = native_parameters(frozen[side][i])
            p,info = refine(hand,scene,renderer,initial,uv,labels[i],2 if side=='left' else 3,k,
                            [] if proposals is not None else args.sweeps,
                            None if proposals is None else proposals[side][i])
            poses[side].append(p)
            stats[side] = info
        for hand in hands.values():
            for node in hand.nodes.values():
                node.mesh.is_visible = True
        rgb,depth = renderer.render(scene)
        rgbs.append(rgb); masks.append(depth>0)
        frame = np.asarray(Image.fromarray(frames[i]).resize((960,540),Image.Resampling.LANCZOS))
        object_mask = np.isin(labels[i],[1,4])
        before,_ = composite(frame,original['rgb'][i],original['mask'][i],object_mask)
        after,_ = composite(frame,rgb,depth>0,object_mask)
        crop = (320,195,870,540)
        close = panels([np.asarray(Image.fromarray(im).crop(crop).resize((800,500))) for im in (frame,before,after)],
                       ['Source','Before (same object masks)','Visible-silhouette refinement'])
        close.save(args.output/f'comparison_{i:03d}.png')
        Image.fromarray(after).save(args.output/f'frame_{i:03d}.png')
        # Diagnostics distinguish anatomical pseudo-joints from fitted robot joints.
        diagnostic = Image.fromarray(frame)
        draw = ImageDraw.Draw(diagnostic)
        for side in hands:
            info = stats[side]
            for j,(target,old) in enumerate(zip(info['target_uv'],info['initial_keypoints_uv'])):
                tx,ty = target; ox,oy = old
                draw.line([(tx,ty),(ox,oy)],fill='yellow',width=1)
                draw.ellipse((tx-2,ty-2,tx+2,ty+2),outline='red',width=1)
                draw.ellipse((ox-2,oy-2,ox+2,oy+2),outline='blue',width=1)
                if j in [0,1,5,9,13,17]:
                    draw.text((ox+3,oy),str(j),fill='blue')
        diagnostic.save(args.output/f'initial_projection_{i:03d}.png')
        stats['seconds'] = time.perf_counter()-start
        allstats.append(stats)
        print(json.dumps(dict(index=i,seconds=stats['seconds'],
                              metrics={s:{k:stats[s][k] for k in ('before','after')} for s in hands})),flush=True)
    renderer.delete()
    np.savez_compressed(args.output/'robot_render.npz',rgb=rgbs,mask=masks,indices=args.frames)
    np.savez_compressed(args.output/'native_parameters.npz',**poses,indices=args.frames)
    assert hashes == {p.name:sha256(p) for p in (base/'qpos.npz',base/'robot_render.npz')}
    report = dict(frames=allstats,source_indices=[config['indices'][i] for i in args.frames],
                  frozen_sha256=hashes,sweeps=args.sweeps,hand_scale=1.,
                  method='Custom coordinate descent on visible SAM silhouette plus weak visible WiLoR landmark anchors',
                  caveats=['Mask agreement is the optimization target, not independent ground truth.',
                           'Not a contact or physical pose validation. Morphology differs from the human hand.'])
    if args.corrections_from:
        report['temporal_correction'] = dict(source=str(args.corrections_from.resolve()),
                                            gaussian_sigma_frames=1.,baseline_motion_smoothed=False,
                                            accepted_only_if_frame_objective_improves=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.frames==list(range(len(frames))):
        with imageio.get_writer(args.output/'robot_hands.mp4',fps=config['fps'],quality=9,macro_block_size=1) as w, \
             imageio.get_writer(args.output/'comparison.mp4',fps=config['fps'],quality=9,macro_block_size=1) as c:
            for i in args.frames:
                w.append_data(imageio.imread(args.output/f'frame_{i:03d}.png'))
                c.append_data(imageio.imread(args.output/f'comparison_{i:03d}.png'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,nargs='+',default=[0,16,32])
    parser.add_argument('--all-frames',action='store_true')
    parser.add_argument('--corrections-from',type=Path)
    parser.add_argument('--sweeps',type=float,nargs='+',default=[2,1,1,.5])
    args = parser.parse_args()
    if args.all_frames:
        args.frames = list(range(33))
    run(args)
