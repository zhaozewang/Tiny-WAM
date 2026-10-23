"""Generate pose-guided SAM prompts and accept only connected proximal evidence.

This is automatic re-prompting, not manual correction or ground-truth anatomy.
The fixed hand fit is never modified. Original usable sleeve masks are retained.
"""
from swam.paths import external_path
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sample_rgb_retarget_batch import SAM, command_for, save
from evaluate_rgb_heldout import execute


def repair(root, item, original):
    import pyrender
    from swam.retargeting.geometry import SharpaHand, project
    from swam.retargeting.anchored_arm import supported_forearm_mask
    key=item['key']; case=root/'clips'/key
    poses=np.load(case/'smooth/native_parameters.npz')
    labels=np.load(case/'segment/labels.npy')
    sleeves=np.load(original/'clips'/key/'sleeves/labels.npy')
    k=np.load(root/'hands'/key/'qpos.npz')['intrinsic'].copy(); k[:2]/=2
    hands={s:SharpaHand(s,pyrender.Scene()) for s in ('left','right')}
    geometry={}; objects=[]; kept=sleeves.copy(); needs={}
    height,width=labels.shape[1:]
    for side,identity in [('left',1),('right',2)]:
        geometry[side]={}; bad=[]
        for i,valid in enumerate(poses[side+'_valid']):
            if not valid: continue
            points, transforms=hands[side].forward(poses[side][i])
            w=project(transforms[hands[side].root][None,:3,3],k)[0]
            p=project(points[[1,5,9,13]],k).mean(0)
            geometry[side][i]=(w,p)
            mask=supported_forearm_mask(sleeves[i]==identity,w,p)
            if not mask.any(): bad.append(i)
        needs[side]=bad
        annotations=[]; last=-16
        for i in bad:
            if i-last<16: continue
            w,p=geometry[side][i]; v=w-p
            positives=[w+factor*v for factor in (.5,1.)]
            positives=[xy for xy in positives if 5<=xy[0]<width-5 and 5<=xy[1]<height-5]
            if not positives: continue
            negatives=[p,p-.5*v]
            negatives=[xy for xy in negatives if 0<=xy[0]<width and 0<=xy[1]<height]
            annotations.append(dict(frame=i,points=[(xy*np.array([512/width,288/height])).tolist()
                                                   for xy in positives+negatives],
                                    labels=[1]*len(positives)+[0]*len(negatives)))
            last=i
        if annotations:
            objects.append(dict(id=identity,name=side+' pose-guided forearm',role='hand',annotations=annotations))
    destination=case/'sleeves'
    if objects:
        prompt=case/'automatic_sleeves_prompts.json'
        save(prompt,dict(size=[512,288],objects=objects,
                        annotation_source='Automatic frozen-wrist/palm geometry; no manual output correction.'))
        candidate=case/'sleeves_candidate'
        command=[SAM,Path(__file__),'--output',root,'--source',original,'--key',key,'--sam-worker']
        if not execute(case,'automatic_sleeves',command):
            raise RuntimeError('Automatic forearm segmentation failed')
        proposal=np.load(candidate/'labels.npy')
    else:
        proposal=np.zeros_like(sleeves)
    records=[]
    for side,identity in [('left',1),('right',2)]:
        for i in needs[side]:
            w,p=geometry[side][i]
            mask=supported_forearm_mask(proposal[i]==identity,w,p)
            # Never claim another observed hand or manipulated object as sleeve.
            mask &= ~np.isin(labels[i],[1,4,3 if side=='left' else 2])
            length=np.linalg.norm(w-p)
            if mask.sum()>max(4096,8*length**2): mask[:]=False
            mask=supported_forearm_mask(mask,w,p)
            kept[i][kept[i]==identity]=0
            mask &= (kept[i]==0)
            kept[i][mask]=identity
            records.append(dict(frame=i,side=side,accepted_pixels=int(mask.sum()),
                                status='automatic_reprompt' if mask.any() else 'missing_forearm_evidence'))
    destination.mkdir(exist_ok=False)
    np.save(destination/'labels.npy',kept)
    metadata=json.loads((original/'clips'/key/'sleeves/segmentation.json').read_text())
    metadata.update(repair='geometry-guided SAM re-prompt, validated observed pixels only',
                    original=str(original/'clips'/key/'sleeves'),automatic_repairs=records)
    save(destination/'segmentation.json',metadata)
    save(destination/'repair.json',records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--key',required=True)
    p.add_argument('--sam-worker',action='store_true')
    a=p.parse_args(); root=a.output.resolve()
    item=next(i for i in json.loads((root/'manifest.json').read_text())['clips'] if i['key']==a.key)
    if a.sam_worker:
        from swam.retargeting.segmentation import track
        case=root/'clips'/a.key
        track(case/'source',case/'sleeves_candidate',case/'automatic_sleeves_prompts.json',
              sam_root=external_path('sam', ''),native=True,output_size=(960,540))
    else:
        repair(root,item,a.source.resolve())
