"""Summarize a local alignment pilot and export all-frame visual contact sheets."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pyrender

from refine_rgb_hand_alignment import ROOT, SharpaHand, native_parameters, project


def run(output):
    report = json.loads((output/'report.json').read_text())
    poses = np.load(output/'native_parameters.npz')
    base = np.load(ROOT/'results/rgb_robot_hands_v1/screw_unscrew_bottle_cap/qpos.npz')
    k = base['intrinsic'].copy()
    k[:2] /= 2
    summary = {}
    for side in ('left','right'):
        hand = SharpaHand(side,pyrender.Scene())
        old_uv = np.array([project(hand.forward(native_parameters(q))[0],k) for q in base[side]])
        new_uv = np.array([project(hand.forward(q)[0],k) for q in poses[side]])
        before = np.array([r[side]['before']['iou'] for r in report['frames']])
        after = np.array([r[side]['after']['iou'] for r in report['frames']])
        summary[side] = dict(mask_iou_before_mean=float(before.mean()),mask_iou_after_mean=float(after.mean()),
                             first_frame_mask_iou=[float(before[0]),float(after[0])],
                             mask_iou_regressed_frames=int((after<before).sum()),
                             joint_limit_excess=float(max(0.,(hand.lower-poses[side][:,6:]).max(),
                                                           (poses[side][:,6:]-hand.upper).max())))
        for name,uv in [('before',old_uv),('after',new_uv)]:
            accel = np.linalg.norm(np.diff(uv,n=2,axis=0),axis=-1)
            summary[side][name+'_projected_accel_px_per_frame2'] = dict(mean=float(accel.mean()),p95=float(np.quantile(accel,.95)))
        summary[side]['caveat'] = 'Image acceleration includes real motion and is not isolated jitter or ground-truth pose error.'
    for block,start in enumerate(range(0,len(poses['indices']),12)):
        canvas = Image.new('RGB',(1440,6*228),'white')
        for cell,i in enumerate(poses['indices'][start:start+12]):
            panel = Image.open(output/f'comparison_{i:03d}.png').crop((1600,28,2400,528))
            panel.thumbnail((720,205))
            x,y = (cell%2)*720,(cell//2)*228
            canvas.paste(panel,(x,y+20))
            ImageDraw.Draw(canvas).text((x+5,y+3),f'Frame {i}',fill='black')
        canvas.save(output/f'all_frames_{block}.jpg',quality=95)
    (output/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    run(parser.parse_args().output)
