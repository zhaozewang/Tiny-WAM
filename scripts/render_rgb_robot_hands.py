"""Retarget approved WiLoR RGB poses to native-size Sharpa hands for visual review.

Uses the published joint objective as initialization, then explicit RGB reprojection
refinement. Floating wrists and virtual intrinsics are not metric robot trajectories.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrender
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
from retargeting.core.kinematics.pinocchio_model import RobotPinocchio
from retargeting.core.kinematics.adaptor import RobotAdaptor
from retargeting.core.types import RetargetingHandObservation
from swam.retargeting.geometry import SharpaHand
from compare_egodex_retargeting import rigid_fit, paper_retargeter, TIPS, BASES

ORDER = [0] + list(range(5,21)) + list(range(1,5))


def floating_robot(hand, path):
    robot = ET.Element('robot', name=hand.root+'_floating')
    ET.SubElement(robot, 'link', name='world')
    names, lower, upper = [], [], []
    parent = 'world'
    for i in range(6):
        child = 'floating_'+str(i)
        name = child+'_joint'
        names.append(name)
        lo, hi = (-50., 50.) if i<3 else (-np.pi, np.pi)
        if i == 2:
            lo, hi = .01, 100.
        lower.append(lo); upper.append(hi)
        ET.SubElement(robot, 'link', name=child)
        j = ET.SubElement(robot, 'joint', name=name, type='prismatic' if i<3 else 'revolute')
        ET.SubElement(j, 'parent', link=parent)
        ET.SubElement(j, 'child', link=child)
        ET.SubElement(j, 'axis', xyz=' '.join(map(str, np.eye(3)[i%3])))
        ET.SubElement(j, 'limit', lower=str(lo), upper=str(hi), effort='1', velocity='100')
        parent = child
    for link in hand.robot.findall('link'):
        ET.SubElement(robot, 'link', name=link.get('name'))
    j = ET.SubElement(robot, 'joint', name='wrist_mount', type='fixed')
    ET.SubElement(j, 'parent', link=parent)
    ET.SubElement(j, 'child', link=hand.root)
    for joint in hand.robot.findall('joint'):
        robot.append(copy.deepcopy(joint))
    ET.indent(robot)
    ET.ElementTree(robot).write(path, encoding='unicode')
    return SimpleNamespace(side=hand.root.split('_')[0], hand=hand, names=names,
        lower=np.r_[lower, hand.lower], upper=np.r_[upper, hand.upper])


def refine_pixels(robot, adaptor, model, initial, target_uv, target_xyz, k, previous):
    links = model.hand.links
    weights = np.full(21, .6)
    weights[TIPS] = 4.
    weights[BASES] = 1.5
    weights[0] = .25
    uv = torch.tensor(target_uv, dtype=torch.float64)
    xyz = torch.tensor(target_xyz, dtype=torch.float64)
    intrinsic = torch.tensor(k, dtype=torch.float64)
    weight = torch.tensor(weights, dtype=torch.float64)
    direction = F.normalize(xyz[TIPS]-xyz[BASES], dim=-1)
    posture = torch.tensor(initial, dtype=torch.float64)
    prev = torch.tensor(initial if previous is None else previous, dtype=torch.float64)

    def objective(q):
        full = adaptor.forward_qpos(q)
        robot.compute_forward_kinematics(full)
        points = np.stack([robot.get_frame_pose(n)[:3,3] for n in links])
        p = torch.tensor(points, dtype=torch.float64, requires_grad=True)
        qt = torch.tensor(q, dtype=torch.float64, requires_grad=True)
        projected = p @ intrinsic.T
        pixels = projected[:,:2] / projected[:,2:].clamp(min=.01)
        image_loss = (F.smooth_l1_loss((pixels-uv)/8., torch.zeros_like(uv), reduction='none').mean(-1)*weight).mean()
        orientation = .35*((F.normalize(p[TIPS]-p[BASES], dim=-1)-direction)**2).mean()
        relative = .05*(((p-p[:1])-(xyz-xyz[:1]))/.03).square().mean()
        q_loss = .002*(qt[6:]-posture[6:]).square().mean() + .005*(qt[3:]-prev[3:]).square().mean()
        loss = image_loss + orientation + relative + q_loss
        loss.backward()
        robot.compute_jacobians(full)
        jacobian = adaptor.backward_jacobian(np.stack([robot.get_frame_space_jacobian(n)[:3] for n in links]))
        gradient = np.einsum('ij,ijk->k', p.grad.numpy(), jacobian) + qt.grad.numpy()
        return float(loss), gradient

    result = minimize(objective, np.clip(initial, model.lower, model.upper), jac=True, method='L-BFGS-B',
                      bounds=list(zip(model.lower,model.upper)), options=dict(maxiter=250, ftol=1e-10, gtol=1e-6))
    robot.compute_forward_kinematics(adaptor.forward_qpos(result.x))
    points = np.stack([robot.get_frame_pose(n)[:3,3] for n in links])
    projected = points @ k.T
    error = np.linalg.norm(projected[:,:2]/projected[:,2:]-target_uv, axis=-1)
    return result.x, dict(tip_mean_px=float(error[TIPS].mean()), all_mean_px=float(error.mean()),
        tip_max_px=float(error[TIPS].max()), iterations=int(result.nit), success=bool(result.success), message=str(result.message))


def run(clip, predictions_root, output, legacy_reference=True):
    output.mkdir(parents=True, exist_ok=False)
    records = json.loads((predictions_root/clip/'predictions.json').read_text())
    summary = json.loads((predictions_root/clip/'summary.json').read_text())
    width,height = summary['input_resolution']
    f = records[0]['wilor'][0]['wilor_preds']['scaled_focal_length']
    k = np.array([[f,0,width/2],[0,f,height/2],[0,0,1.]])
    scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[.65]*3)
    hands = {s: SharpaHand(s,scene) for s in ('left','right')}
    models, robots, adaptors, solvers, previous = {}, {}, {}, {}, {}
    per_side = {s: [] for s in hands}
    qpos = {s: [] for s in hands}
    pose_frames = []
    for i, record in enumerate(records):
        frame_poses = {}
        for s,hand in hands.items():
            detected = [p for p in record['wilor'] if int(p['is_right']) == int(s=='right')]
            if len(detected) != 1:
                raise ValueError(f'Frame {i} {s}: expected exactly one RGB detection')
            prediction = detected[0]['wilor_preds']
            if not np.isclose(prediction['scaled_focal_length'], f):
                raise ValueError('Inconsistent virtual intrinsics')
            xyz = np.array(prediction['pred_keypoints_3d'])[0,ORDER] + np.array(prediction['pred_cam_t_full'])[0]
            uv = np.array(prediction['pred_keypoints_2d'])[0,ORDER]
            check = xyz @ k.T
            np.testing.assert_allclose(check[:,:2]/check[:,2:], uv, atol=.01)
            neutral, _ = hand.forward(np.r_[np.zeros(6), np.clip(np.zeros(len(hand.names)), hand.lower, hand.upper)])
            wrist = rigid_fit(neutral[[0,1,5,9,13]], xyz[[0,1,5,9,13]])
            if s not in models:
                urdf = output/(s+'.urdf')
                model = floating_robot(hand, urdf)
                robot = RobotPinocchio(str(urdf), 'urdf')
                adaptor = RobotAdaptor(robot, model.names+hand.names)
                initial = np.r_[wrist[:3,3], Rotation.from_matrix(wrist[:3,:3]).as_euler('XYZ'),
                                np.clip(np.zeros(len(hand.names)),hand.lower,hand.upper)]
                solver, config = paper_retargeter(model, robot, adaptor, urdf, initial, ablation=2)
                # Floating translations have virtual-camera units; published arm regularization is not appropriate.
                # The explicit pixel stage regularizes rotation and finger motion, not virtual depth.
                models[s], robots[s], adaptors[s], solvers[s] = model,robot,adaptor,solver
                previous[s] = None
                (output/(s+'_paper_config.json')).write_text(json.dumps(config,indent=2))
            model,robot,adaptor,solver = models[s],robots[s],adaptors[s],solvers[s]
            observation = RetargetingHandObservation((xyz-wrist[:3,3])@wrist[:3,:3], wrist, handedness=s)
            # Rebase the floating root on each independent RGB observation instead of dragging virtual depth.
            prior = solver.previous_qpos.astype(float).copy()
            prior[:3] += wrist[:3,3]-prior[:3]
            prior = np.clip(prior, model.lower+1e-8, model.upper-1e-8)
            start = time.perf_counter()
            initial = solver.solve(observation, previous_qpos=prior).qpos.astype(float)
            q, stats = refine_pixels(robot, adaptor, model, initial, uv, xyz, k, previous[s])
            stats['seconds'] = time.perf_counter()-start
            stats['limit_excess_rad_or_m'] = float(max(0.,(model.lower-q).max(),(q-model.upper).max()))
            previous[s] = q
            solver.reset(q)
            qpos[s].append(q)
            per_side[s].append(stats)
            frame_poses.update({name: robot.get_frame_pose(name).copy() for name in hand.nodes})
        pose_frames.append(frame_poses)
        if i%8 == 0:
            print(clip,i+1,'/',len(records),{s:round(per_side[s][-1]['tip_mean_px'],2) for s in hands},flush=True)
    np.savez_compressed(output/'qpos.npz', **{s:np.array(q) for s,q in qpos.items()}, intrinsic=k,
                        source_indices=[r['source_frame'] for r in records])
    scene.add(pyrender.IntrinsicsCamera(f/2,f/2,width/4,height/4,znear=.01,zfar=100.), pose=np.diag([1.,-1.,-1.,1.]))
    scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=np.diag([1.,-1.,-1.,1.]))
    renderer = pyrender.OffscreenRenderer(width//2,height//2)
    reader = imageio.get_reader(Path(summary['source'])/'source.mp4')
    old_reader = (imageio.get_reader(ROOT/'results'/('retarget_papers_chopsticks_v2' if clip=='use_chopsticks' else 'retarget_papers_bottle_v2')/'paper_actual_distance.mp4')
                  if legacy_reference else None)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    rgb_frames, masks = [], []
    boxes = np.array([p['hand_bbox'] for r in records for p in r['wilor'] if p['is_right']==1])
    lo = np.maximum(boxes[:,:2].min(0)-30,[0,0])/2
    hi = np.minimum(boxes[:,2:].max(0)+30,[width,height])/2
    crop = tuple(np.r_[lo,hi].astype(int))
    with imageio.get_writer(output/'robot_hands.mp4',fps=summary['fps'],macro_block_size=1,quality=8) as writer, \
         imageio.get_writer(output/'comparison.mp4',fps=summary['fps'],macro_block_size=1,quality=8) as comparison, \
         imageio.get_writer(output/'right_hand_detail.mp4',fps=summary['fps'],macro_block_size=1,quality=8) as detail:
        for i,(record,poses) in enumerate(zip(records,pose_frames)):
            frame = np.asarray(Image.fromarray(reader.get_data(record['source_frame'])).resize((width//2,height//2),Image.Resampling.LANCZOS))
            for s,hand in hands.items():
                for name,node in hand.nodes.items():
                    scene.set_pose(node,poses[name])
            rgb,depth = renderer.render(scene)
            mask = depth>0
            overlay = np.where(mask[...,None],rgb,frame)
            rgb_frames.append(rgb); masks.append(mask)
            writer.append_data(overlay)
            old = (np.asarray(Image.fromarray(old_reader.get_data(i)).resize((width//2,height//2)))
                   if old_reader is not None else frame)
            panels = [frame,old,overlay]
            labels = ['Source','Previous: EgoDex-driven' if legacy_reference else 'Source (legacy comparison disabled)','New: RGB-driven hands']
            canvas = Image.new('RGB',(1536,320),(245,245,245))
            close = Image.new('RGB',(1800,640),(245,245,245))
            for j,(panel,label) in enumerate(zip(panels,labels)):
                ImageDraw.Draw(canvas).text((j*512+8,5),label,font=font,fill=(25,25,25))
                canvas.paste(Image.fromarray(panel).resize((512,288)),(j*512,32))
                ImageDraw.Draw(close).text((j*600+8,8),label,font=font,fill=(25,25,25))
                patch = Image.fromarray(panel).crop(crop)
                patch.thumbnail((600,600))
                close.paste(patch,(j*600+(600-patch.width)//2,40+(600-patch.height)//2))
            comparison.append_data(np.asarray(canvas)); detail.append_data(np.asarray(close))
            if i in [0,len(records)//2,len(records)-1]:
                canvas.save(output/f'comparison_{i:03d}.png')
                close.save(output/f'detail_{i:03d}.png')
    reader.close()
    if old_reader is not None:
        old_reader.close()
    renderer.delete()
    np.savez_compressed(output/'robot_render.npz',rgb=rgb_frames,mask=masks)
    report = dict(source_predictions=str(predictions_root/clip),source_indices=[r['source_frame'] for r in records],
        fps=summary['fps'],hand_scale=1.,method='Official paper ablation 2 initialization + custom RGB pixel/direction refinement',
        virtual_intrinsic=k.tolist(), floating_wrists=True, physical_arms=False, old_baseline_has_arms=legacy_reference,
        object_occlusion=False,human_inpainting=False,pose_metrics=per_side,
        mean_tip_error_native_rgb_px={s:float(np.mean([r['tip_mean_px'] for r in rows])) for s,rows in per_side.items()},
        caveats=['Pixel errors are against WiLoR predictions, not human ground truth',
                 'Virtual-camera depth is not physical scene depth; do not execute these wrist translations on hardware',
                 'Original human pixels remain outside robot coverage; no object-contact or collision validation'],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['mean_tip_error_native_rgb_px']),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions',type=Path,default=ROOT/'results/rgb_hands_wilor_v2')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--clips',nargs='+',default=['use_chopsticks','screw_unscrew_bottle_cap'])
    parser.add_argument('--no-legacy-reference',action='store_true',
                        help='Skip archived comparison-only videos; does not change the hand solver or render')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    for clip in args.clips:
        run(clip,args.predictions,args.output/clip,legacy_reference=not args.no_legacy_reference)


if __name__ == '__main__':
    main()
