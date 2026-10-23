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
from swam.retargeting.visibility import pose_validity, filter_valid_runs
from occlude_rgb_robot_hands import panels, sha256

ORDER = [0]+list(range(5,21))+list(range(1,5))
TIPS = [4,8,12,16,20]


def native_parameters(q):
    return np.r_[Rotation.from_euler('XYZ',q[3:6]).as_rotvec(),q[:3],q[6:]]


def refine(hand, scene, renderer, initial, target_uv, labels, identity, k, sweeps, proposal=None, memoize=False):
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
    raster_calls = 0
    evaluations = {}
    gpu_objective = hasattr(renderer,'configure_score')
    if gpu_objective:
        renderer.configure_score(outside,target,~objects&region)

    def metrics(p,points,iou,out):
        uv = project(points,k)
        errors = np.linalg.norm(uv-target_uv,axis=-1)
        anchoring = float((weights*errors).sum()/weights.sum())
        prior = float(((p[6:]-initial[6:])**2).mean())
        loss = 20*(1-iou)+out+.3*anchoring+.5*prior
        return loss,dict(iou=iou,outside_px=out,visible_landmark_px=anchoring,
                         all_landmarks_px=float(errors.mean()),objective=loss)

    def evaluate(p):
        nonlocal calls,raster_calls
        if gpu_objective:return evaluate_many([p])[0]
        calls += 1
        key=p.tobytes()
        if memoize and key in evaluations:
            return evaluations[key]
        raster_calls += 1
        points,_ = hand.show(scene,p)
        visible = (renderer.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY)>0)&~objects&region
        union = (visible|target).sum()
        iou = float((visible&target).sum()/max(union,1))
        out = float(np.minimum(outside[visible],20).mean()) if visible.any() else 20.
        # Visible-mask agreement addresses palm placement absent from fingertip fitting.
        result=metrics(p,points,iou,out)
        if memoize:evaluations[key]=result
        return result

    def evaluate_many(parameters):
        nonlocal calls,raster_calls
        if not gpu_objective:return [evaluate(p) for p in parameters]
        calls += len(parameters)
        pending = {}
        for p in parameters:
            key=p.tobytes()
            if key in evaluations or key in pending:continue
            points,_=hand.show(scene,p)
            renderer.queue_score(scene,len(pending))
            pending[key]=(p,points)
            raster_calls+=1
        if pending:
            reduced=renderer.read_scores(len(pending))
            for (key,(p,points)),(intersection,union,count,distance) in zip(pending.items(),reduced):
                evaluations[key]=metrics(p,points,float(intersection/max(union,1)),
                                         float(distance/count) if count else 20.)
        return [evaluations[p.tobytes()] for p in parameters]

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
            trials=[]
            for sign in (-1,1):
                trial = q.copy()
                trial[d] = np.clip(trial[d]+sign*step,lower[d],upper[d])
                trials.append(trial)
            for trial,(value,stats) in zip(trials,evaluate_many(trials)):
                if value < score:
                    score,candidate,candidate_info = value,trial,stats
            q,after = candidate,candidate_info
    hand.show(scene,q)
    return q,dict(before=before,after=after,calls=calls,raster_calls=raster_calls,
                  initial_keypoints_uv=before_uv.tolist(),target_uv=target_uv.tolist(),
                  visible_pseudo_targets=visible_targets.tolist())


def run(args):
    clip = 'screw_unscrew_bottle_cap'
    base = getattr(args,'base',None) or ROOT/'results/rgb_robot_hands_v1'/clip
    segmentation = getattr(args,'segmentation',None) or ROOT/'results/rgb_robot_occlusion_v1'/clip/'segmentation'
    source = getattr(args,'source',None) or ROOT/'results/retargeting_import_20261007'/clip/'source'
    predictions = getattr(args,'predictions',None) or ROOT/'results/rgb_hands_wilor_v2'/clip
    records = json.loads((predictions/'predictions.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    frames,config = load_frames(source,True)
    if args.all_frames:
        args.frames = list(range(len(frames)))
    labels = np.load(segmentation/'labels.npy')
    frozen = np.load(base/'qpos.npz')
    valid = {s:pose_validity(frozen,s) for s in ('left','right')}
    if not np.array_equal(frozen['source_indices'],config['indices']):
        raise ValueError('Hand poses and source frame indices differ')
    if [r['source_frame'] for r in records] != config['indices']:
        raise ValueError('RGB predictions and source frame indices differ')
    metadata = json.loads((segmentation/'segmentation.json').read_text())
    if metadata['source_indices'] != config['indices']:
        raise ValueError('Segmentation and source frame indices differ')
    proposals = None
    if args.corrections_from:
        proposed = np.load(args.corrections_from/'native_parameters.npz')
        if not np.array_equal(proposed['indices'],np.arange(len(frames))):
            raise ValueError('Temporal correction requires every source frame in order')
        proposals = {}
        for side in ('left','right'):
            initial = np.array([native_parameters(q) if v else np.full_like(q,np.nan)
                                for q,v in zip(frozen[side],valid[side])])
            # Smooth only the added correction, not the approved source motion.
            delta = filter_valid_runs(proposed[side]-initial,valid[side])
            proposals[side] = initial+delta
    original = np.load(base/'robot_render.npz')
    hashes = {p.name:sha256(p) for p in (base/'qpos.npz',base/'robot_render.npz')}
    k = frozen['intrinsic'].copy()
    k[:2] /= 2
    scene = pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.65]*3)
    hands = {s:SharpaHand(s,scene) for s in ('left','right')}
    scene.add(pyrender.IntrinsicsCamera(k[0,0],k[1,1],k[0,2],k[1,2],znear=.01,zfar=100),pose=np.diag([1.,-1.,-1.,1.]))
    scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=np.diag([1.,-1.,-1.,1.]))
    if getattr(args,'gpu_objective',False):
        from swam.retargeting.gpu_silhouette import GpuSilhouetteRenderer
        renderer = GpuSilhouetteRenderer(960,540)
    elif getattr(args,'fast_mask_readback',False):
        from swam.retargeting.silhouette_renderer import SilhouetteRenderer
        renderer = SilhouetteRenderer(960,540)
    else:
        renderer = pyrender.OffscreenRenderer(960,540)
    allstats, poses, rgbs, masks = [],{s:[] for s in hands},[],[]
    for i in args.frames:
        start = time.perf_counter()
        stats = {'index':i,'source_index':config['indices'][i]}
        for side,hand in hands.items():
            if not valid[side][i]:
                poses[side].append(np.full(6+len(hand.names),np.nan))
                stats[side] = dict(status='no_unambiguous_rgb_detection',fit_performed=False)
                continue
            for other,h in hands.items():
                for node in h.nodes.values():
                    node.mesh.is_visible = other==side
            prediction = next(p for p in records[i]['wilor'] if int(p['is_right'])==int(side=='right'))
            uv = np.array(prediction['wilor_preds']['pred_keypoints_2d'])[0,ORDER]/2
            initial = native_parameters(frozen[side][i])
            identity = 2 if side=='left' else 3
            if np.count_nonzero(labels[i]==identity)<12:
                p,info = initial,dict(status='hand_mask_missing',fit_performed=False)
                hand.show(scene,p)
            else:
                p,info = refine(hand,scene,renderer,initial,uv,labels[i],identity,k,
                            [] if proposals is not None else args.sweeps,
                            None if proposals is None else proposals[side][i],getattr(args,'fast_mask_readback',False))
            poses[side].append(p)
            stats[side] = info
        for side,hand in hands.items():
            for node in hand.nodes.values():
                node.mesh.is_visible = bool(valid[side][i])
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
            if 'target_uv' not in info:
                continue
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
                              metrics={s:{key:stats[s][key] for key in ('before','after','status')
                                          if key in stats[s]} for s in hands})),flush=True)
    renderer.delete()
    np.savez_compressed(args.output/'robot_render.npz',rgb=rgbs,mask=masks,indices=args.frames)
    np.savez_compressed(args.output/'native_parameters.npz',**poses,indices=args.frames,
                        **{s+'_valid':v[args.frames] for s,v in valid.items()})
    assert hashes == {p.name:sha256(p) for p in (base/'qpos.npz',base/'robot_render.npz')}
    report = dict(frames=allstats,source_indices=[config['indices'][i] for i in args.frames],
                  fast_mask_readback=getattr(args,'fast_mask_readback',False),
                  gpu_objective=getattr(args,'gpu_objective',False),
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
    parser.add_argument('--fast-mask-readback',action='store_true')
    parser.add_argument('--gpu-objective',action='store_true',help='Experimental GPU silhouette reduction with paired-candidate readback')
    parser.add_argument('--corrections-from',type=Path)
    parser.add_argument('--sweeps',type=float,nargs='+',default=[2,1,1,.5])
    for name in ('source','base','segmentation','predictions'):
        parser.add_argument('--'+name,type=Path)
    args = parser.parse_args()
    run(args)
