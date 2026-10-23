"""Periodic detector/SAM object refresh with continuous SAM hand tracking.

Object instances are unioned under label 1, matching the downstream occlusion
contract. Hand identities stay 2/3. No robot pose or fitting parameters change.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from sample_rgb_retarget_batch import ROOT,SAMROOT,save


def run(root,key,prompt_source):
    import torch
    sys.path.insert(0,str(SAMROOT));sys.path.insert(0,str(ROOT/'src'))
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from tinywam.retargeting.segmentation import assign_labels
    torch.set_num_threads(4)
    case=root/'clips'/key;source=case/'source';out=case/'segment';out.mkdir(exist_ok=False)
    config=json.loads((source/'prepare.json').read_text())
    detections=json.loads((case/'grounded_objects/detections.json').read_text())
    prompts=json.loads((prompt_source/'clips'/key/'segment_prompts.json').read_text())
    hands=[o for o in prompts['objects'] if o['id'] in (2,3)]
    if not hands:raise ValueError('No initialized hands; requires review')
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(SAMROOT/'sam2.1_hiera_large.pt'),
        device='cuda',hydra_overrides_extra=['++model.non_overlap_masks=true'])
    image_predictor=SAM2ImagePredictor(predictor)
    n=len(config['indices']);labels=np.lib.format.open_memmap(out/'labels.npy',mode='w+',dtype=np.uint16,shape=(n,540,960))
    covered=np.zeros(n,dtype=bool);audit=[];start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='frames-',dir=out) as directory:
        with imageio.get_reader(source/'source.mp4') as reader:
            for i,index in enumerate(config['indices']):
                im=Image.fromarray(reader.get_data(index)).crop(config['crop'])
                im.save(Path(directory)/f'{i:05d}.jpg',quality=95)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=predictor.init_state(directory,offload_video_to_cpu=True,offload_state_to_cpu=True)
            width,height=Image.open(Path(directory)/'00000.jpg').size
            for hand in hands:
                for annotation in hand['annotations']:
                    points=np.array(annotation['points'],dtype=np.float32)*[width/512,height/288]
                    predictor.add_new_points_or_box(state,frame_idx=annotation['frame'],obj_id=hand['id'],
                        points=points,labels=np.array(annotation['labels']))
            for record in detections['records']:
                index=record['frame'];im=np.array(Image.open(Path(directory)/f'{index:05d}.jpg').convert('RGB'))
                image_predictor.set_image(im);union=np.zeros(im.shape[:2],dtype=bool);proposals=[]
                for obj in record['boxes']:
                    box=np.array(obj['box'],dtype=np.float32)
                    box_area=float((box[2]-box[0])*(box[3]-box[1]))
                    if box_area>.8*width*height:
                        proposals.append(dict(**obj,accepted=False,reason='Scene-sized box is not a localized foreground object'))
                        continue
                    masks,scores,_=image_predictor.predict(box=box,multimask_output=False)
                    mask=masks[0].astype(bool);area=int(mask.sum())
                    accepted=area>0 and area<=box_area*2
                    proposals.append(dict(**obj,sam_score=float(scores[0]),area=area,accepted=accepted))
                    if accepted:union|=mask
                # Missing detection is not evidence that an existing object vanished.
                if union.any():predictor.add_new_mask(state,frame_idx=index,obj_id=1,mask=union)
                audit.append(dict(frame=index,proposals=proposals,mask_pixels=int(union.sum())))
            image_predictor.reset_predictor()
            if not any(r['mask_pixels'] for r in audit):raise ValueError('No accepted object masks')
            for index,identities,logits in predictor.propagate_in_video(state,start_frame_idx=0):
                scores=logits[:,0].float().cpu().numpy()
                scores=np.stack([np.asarray(Image.fromarray(s).resize((960,540),Image.Resampling.BILINEAR)) for s in scores])
                labels[index]=assign_labels(scores,identities);covered[index]=True
    labels.flush()
    if not covered.all():raise RuntimeError('Incomplete temporal coverage')
    for name,ids in [('object_masks',[1]),('human_masks',[2,3])]:
        target=np.lib.format.open_memmap(out/(name+'.npy'),mode='w+',dtype=bool,shape=labels.shape)
        for i in range(n):target[i]=np.isin(labels[i],ids)
        target.flush();del target
    save(out/'prompts.json',dict(size=[512,288],objects=hands,object_method='periodic grounded mask union'))
    save(out/'segmentation.json',dict(source=str(source.resolve()),source_indices=config['indices'],
        source_prepare_sha256=hashlib.sha256((source/'prepare.json').read_bytes()).hexdigest(),
        model_config='configs/sam2.1/sam2.1_hiera_l.yaml',checkpoint=str(SAMROOT/'sam2.1_hiera_large.pt'),
        input_size=[width,height],output_size=[960,540],native_input=True,non_overlap=True,
        seconds=time.perf_counter()-start,objects=hands+[dict(id=1,role='object',name='grounded foreground object union')],
        method='grounded_refresh_v2',max_box_image_fraction=.8,object_interval=detections['interval'],anchor_audit=audit,
        semantics='visible masks only, not metric contact/depth; no hand fit changes'))
    print(key,'complete',round(time.perf_counter()-start,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prompt-source',type=Path,required=True);p.add_argument('--key',required=True)
    a=p.parse_args();run(a.output.resolve(),a.key,a.prompt_source.resolve())
