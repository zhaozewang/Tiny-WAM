"""Verify a random retarget pilot and render an unfiltered source/result reel."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from swam.retargeting.segmentation import load_frames
from swam.retargeting.visibility import pose_validity
from swam.retargeting.occlusion import composite
from swam.retargeting.anchored_arm import WristAnchoredArm


def limits(side):
    path=ROOT/f'assets/sharpa/wave_01/{side}_sharpa_wave/{side}_sharpa_wave.urdf'
    joints=[j for j in ET.parse(path).getroot().findall('joint') if j.get('type')=='revolute']
    return np.array([float(j.find('limit').get('lower')) for j in joints]),np.array([float(j.find('limit').get('upper')) for j in joints])


def review(root,baseline=None):
    manifest=json.loads((root/'manifest.json').read_text())
    out=root/'review'
    out.mkdir(exist_ok=False)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    rows=[]
    contact=Image.new('RGB',(1920,320*len(manifest['clips'])),'white')
    with imageio.get_writer(out/'all_samples.mp4',fps=15,quality=9,macro_block_size=1) as reel:
        for row,item in enumerate(manifest['clips']):
            case=root/'clips'/item['key']
            entry=dict(member=item['member'],key=item['key'],visual_verdict='pending_manual_review')
            rows.append(entry)
            try:
                source,config=load_frames(case/'source',True)
                count=len(source)
                poses=np.load(case/'smooth/native_parameters.npz')
                with np.load(case/'smooth/robot_render.npz') as archive:
                    hand={name:archive[name] for name in archive.files}
                with np.load(case/'arms/arm_render.npz') as archive:
                    arms={name:archive[name] for name in archive.files}
                qarm=np.load(case/'arms/arms.npz')
                labels=np.load(case/'segment/labels.npy')
                assert hand['rgb'].shape==(count,540,960,3)
                assert arms['rgb'].shape==hand['rgb'].shape
                np.testing.assert_array_equal(poses['indices'],np.arange(count))
                report=json.loads((case/'arms/report.json').read_text())
                assert report['visible_hand_pixels_identical'] and not report['hand_parameters_modified']
                for name,sha in report['frozen_sha256'].items():
                    assert hashlib.sha256((case/'smooth'/name).read_bytes()).hexdigest()==sha
                hand_counts={};arm_counts={};arm_checks={}
                for side in ('left','right'):
                    valid=pose_validity(poses,side);hand_counts[side]=int(valid.sum())
                    assert poses[side].shape==(count,28)
                    assert np.isnan(poses[side][~valid]).all()
                    lo,hi=limits(side)
                    assert np.all(poses[side][valid,6:]>=lo-1e-8)
                    assert np.all(poses[side][valid,6:]<=hi+1e-8)
                    av=qarm[side+'_valid'];arm_counts[side]=int(av.sum())
                    assert not (av&~valid).any()
                    assert np.isfinite(qarm[side][av]).all()
                    assert np.isnan(qarm[side][~av]).all()
                    model=WristAnchoredArm(side)
                    q=qarm[side][av]
                    excess=max(0.,float(np.max(model.lower-q)),float(np.max(q-model.upper))) if len(q) else 0.
                    pairs=av[1:]&av[:-1]
                    steps=np.abs(np.diff(qarm[side],axis=0))[pairs]
                    step=float(steps.max()) if len(steps) else 0.
                    assert excess<1e-8 and step<=.3+1e-8
                    arm_checks[side]=dict(limit_excess=excess,maximum_contiguous_frame_step=step)
                detection=json.loads((root/'hands'/item['key']/'report.json').read_text())
                alignment=json.loads((case/'smooth/report.json').read_text())
                ious={}
                for side in ('left','right'):
                    valid_rows=[f[side] for f in alignment.get('frames',[]) if 'before' in f[side]]
                    ious[side]={k:float(np.mean([r[k]['iou'] for r in valid_rows])) if valid_rows else None for k in ('before','after')}
                info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                    '-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(case/'arms/robot_arms.mp4')]))['streams'][0]
                assert int(info['nb_read_frames'])==count
                assert info['width']==960 and info['height']==540
                entry.update(processing_verified=True,frames=count,fps=config['fps'],hand_valid_frames=hand_counts,
                             arm_valid_frames=arm_counts,arm_checks=arm_checks,mask_fit_iou=ious,
                             detection_status={s:[r['detection_status'] for r in detection['pose_metrics'][s]] for s in hand_counts},
                             hand_parameters_unchanged=True,hand_pixels_unchanged=True,
                             wrist_transform_error=report['max_wrist_transform_error'],
                             note='Unresolved missing hands are hidden. ROI recovery provenance is in predictions/summary.json. IoU uses fitted SAM masks, not independent ground truth; null means not recomputed after pose merging.')
                with imageio.get_writer(out/(item['key']+'.mp4'),fps=config['fps'],quality=9,macro_block_size=1) as writer:
                    for i,frame in enumerate(source):
                        frame=np.asarray(Image.fromarray(frame).resize((960,540),Image.Resampling.LANCZOS))
                        before,visible=composite(frame,hand['rgb'][i],hand['mask'][i],np.isin(labels[i],[1,4]))
                        after=np.where(arms['mask'][i,...,None],arms['rgb'][i],before)
                        np.testing.assert_array_equal(before[visible],after[visible])
                        canvas=Image.new('RGB',(1920,388),'white')
                        for j,(im,title) in enumerate(zip([frame,before,after],['Source RGB','Robot hands + observed occlusion','Frozen hands + fitted arms'])):
                            canvas.paste(Image.fromarray(im).resize((640,360)),(j*640,28))
                            ImageDraw.Draw(canvas).text((j*640+6,5),f'{item["key"].split("_test_")[-1]} | {title}',font=font,fill='black')
                        writer.append_data(np.asarray(canvas));reel.append_data(np.asarray(canvas))
                        if i in (0,count//2,count-1):
                            canvas.save(out/f'{item["key"]}_{i:03d}.png')
                            col=[0,count//2,count-1].index(i)
                            contact.paste(Image.fromarray(after).resize((568,320)),(col*640,row*320))
                            ImageDraw.Draw(contact).text((col*640+5,row*320+5),f'{row}: frame {i}',font=font,fill='black')
            except Exception as error:
                entry.update(processing_verified=False,error=repr(error))
                print(item['key'],'FAILED',repr(error),flush=True)
                # A failed sample must remain visible in an unfiltered review reel.
                fallback,config=load_frames(case/'source',True)
                with imageio.get_writer(out/(item['key']+'_failed.mp4'),fps=config['fps'],quality=9,macro_block_size=1) as writer:
                    for frame in fallback:
                        canvas=Image.new('RGB',(1920,388),'white')
                        canvas.paste(Image.fromarray(frame).resize((640,360)),(0,28))
                        draw=ImageDraw.Draw(canvas)
                        draw.text((8,5),item['key']+' | Source RGB',font=font,fill='black')
                        draw.text((660,60),'Processing/verification failed; sample retained',font=font,fill='black')
                        draw.text((660,90),repr(error)[:110],font=font,fill='black')
                        writer.append_data(np.asarray(canvas));reel.append_data(np.asarray(canvas))
    contact.save(out/'contact.png')
    if baseline is not None:
        baseline_checks={}
        with imageio.get_writer(out/'before_after.mp4',fps=15,quality=9,macro_block_size=1) as video:
            for item in manifest['clips']:
                key=item['key'];oldcase=baseline/'clips'/key;case=root/'clips'/key
                oldposes=np.load(oldcase/'smooth/native_parameters.npz')
                newposes=np.load(case/'smooth/native_parameters.npz')
                intrinsic=np.load(root/'hands'/key/'qpos.npz')['intrinsic'].copy();intrinsic[:2]/=2
                checks={}
                for side in ('left','right'):
                    keep=pose_validity(oldposes,side)
                    np.testing.assert_array_equal(oldposes[side][keep],newposes[side][keep])
                    checks[side]=dict(old_valid=int(keep.sum()),new_valid=int(pose_validity(newposes,side).sum()),
                                      original_valid_poses_exactly_preserved=True)
                    for tag,directory in [('before',oldcase),('after',case)]:
                        arm=np.load(directory/'arms/arms.npz')
                        valid=arm[side+'_valid']
                        xyz=arm[side+'_base'][valid,:3,3]
                        projected=xyz@intrinsic.T
                        uv=projected[:,:2]/projected[:,2:]
                        inside=((uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<540)&(xyz[:,2]>0))
                        checks[side][tag+'_projected_base_inside_frame']=int(inside.sum())
                        checks[side][tag+'_visible_arm_frames']=int(valid.sum())
                baseline_checks[key]=checks
                with imageio.get_reader(oldcase/'arms/robot_arms.mp4') as oldvideo, \
                     imageio.get_reader(case/'arms/robot_arms.mp4') as newvideo:
                    for i in range(len(item['indices'])):
                        canvas=Image.new('RGB',(1920,568),'white')
                        canvas.paste(Image.fromarray(oldvideo.get_data(i)),(0,28))
                        canvas.paste(Image.fromarray(newvideo.get_data(i)),(960,28))
                        draw=ImageDraw.Draw(canvas)
                        draw.text((8,5),key+' | Before',font=font,fill='black')
                        draw.text((968,5),'After | same frames, original valid hand poses preserved',font=font,fill='black')
                        video.append_data(np.asarray(canvas))
                        if i in (0,9,17,22,len(item['indices'])-1):
                            canvas.save(out/f'before_after_{key}_{i:03d}.png')
        (out/'baseline_checks.json').write_text(json.dumps(baseline_checks,indent=2)+'\n')
    protocol=json.loads((root/'protocol.json').read_text()) if (root/'protocol.json').exists() else {}
    result=dict(seed=manifest['seed'],population_size=manifest.get('population_size'),samples=rows,
                segmentation=protocol.get('prompts','Manual per-sample point prompts + SAM2; not an automatic segmentation benchmark'),
                interpretation='Software invariants and source coverage are separate from visual quality; no dataset-wide guarantee')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    snapshot=out/'source_snapshot'
    for p in [*ROOT.glob('src/swam/retargeting/*.py'),*ROOT.glob('scripts/*rgb*py'),ROOT/'scripts/fit_arms_to_frozen_hands.py']:
        dst=snapshot/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    (out/'working_diff.patch').write_bytes(subprocess.check_output(['git','diff','--',
        'src/swam/retargeting','scripts/reestimate_egodex_rgb.py','scripts/render_rgb_robot_hands.py',
        'scripts/refine_rgb_hand_alignment.py','scripts/fit_arms_to_frozen_hands.py','tests/test_rgb_robot_hands.py'],cwd=ROOT))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    review(args.output.resolve(),args.baseline.resolve() if args.baseline else None)
