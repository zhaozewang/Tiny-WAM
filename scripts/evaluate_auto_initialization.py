"""Paired automatic-prompt experiment; never overwrite manual baseline masks."""
from tinywam.paths import external_path
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from evaluate_rgb_heldout import execute, run as heldout_stage
from sample_rgb_retarget_batch import ROOT, save

PROMPT_PY=Path(external_path('retarget', 'prompt-env/bin/python'))


def run(source,root,stage,policy='original'):
    manifest=json.loads((source/'manifest.json').read_text())
    if stage=='prepare':
        root.mkdir(parents=True,exist_ok=False)
        save(root/'manifest.json',manifest)
        save(root/'provenance.json',dict(source=str(source),split='paired development comparison',
            change='Automatic local VLM first-frame SAM initialization; retargeting parameters unchanged',
            bulk_release=False))
        (root/'source_review').symlink_to(source/'source_review',target_is_directory=True)
        (root/'initial_predictions').symlink_to(source/'initial_predictions',target_is_directory=True)
        for item in manifest['clips']:
            case=root/'clips'/item['key'];case.mkdir(parents=True)
            (case/'source').symlink_to(source/'clips'/item['key']/'source',target_is_directory=True)
    elif stage=='initialize':
        if not execute(root,'automatic_initialize',[PROMPT_PY,ROOT/'scripts/initialize_rgb_prompts.py','--output',root,'--policy',policy]):
            raise RuntimeError('Automatic initialization failed; do not release bulk job')
        heldout_stage(root,manifest,'prompts')
    elif stage in ('segment','sleeves'):
        heldout_stage(root,manifest,stage)
        rows=json.loads((root/'timings.json').read_text())
        if any(t['status']!='complete' for t in rows):raise RuntimeError('Failed stage; no bulk release')
    elif stage=='review':
        out=root/'mask_review';out.mkdir(exist_ok=False)
        colors=np.array([[0,0,0],[255,200,20],[40,160,255],[255,65,100],[80,220,80]],dtype=np.uint8)
        stats=[]
        for item in manifest['clips']:
            key=item['key'];image=Image.open(root/'source_review'/(key+'.png')).convert('RGB')
            canvas=Image.new('RGB',(1920,1120),'white');draw=ImageDraw.Draw(canvas)
            row={}
            for j,kind in enumerate(('segment','sleeves')):
                old=np.load(source/'clips'/key/kind/'labels.npy',mmap_mode='r')
                new=np.load(root/'clips'/key/kind/'labels.npy',mmap_mode='r')
                row[kind]={}
                for identity in (range(1,5) if kind=='segment' else range(1,3)):
                    a=old==identity;b=new==identity;union=(a|b).sum()
                    row[kind][str(identity)]=dict(paired_mask_iou=float((a&b).sum()/union) if union else None,
                        new_frames_with_mask=int(b.sum((1,2)).astype(bool).sum()))
                for col,(labels,title) in enumerate([(old[0],'Manual initialization'),(new[0],'Automatic initialization')]):
                    pic=np.array(image);mask=labels>0
                    pic[mask]=(.55*pic[mask]+.45*colors[labels[mask]]).astype('uint8')
                    canvas.paste(Image.fromarray(pic),(960*col,560*j+20))
                    draw.text((960*col+8,560*j+5),kind+' | '+title,fill='black')
            canvas.save(out/(key+'.jpg'));stats.append(dict(key=key,masks=row))
            labels=np.load(root/'clips'/key/'segment/labels.npy',mmap_mode='r')
            sleeves=np.load(root/'clips'/key/'sleeves/labels.npy',mmap_mode='r')
            indices=json.loads((root/'clips'/key/'source/prepare.json').read_text())['indices']
            strip=Image.new('RGB',(1920,3*560),'white');pen=ImageDraw.Draw(strip)
            with imageio.get_reader(root/'clips'/key/'source/source.mp4') as reader:
                for row_index,frame_index in enumerate((0,len(indices)//2,len(indices)-1)):
                    source_image=np.array(Image.fromarray(reader.get_data(indices[frame_index])).resize((960,540)))
                    for col,(mask,title) in enumerate(((labels[frame_index],'Hands and objects'),(sleeves[frame_index],'Forearms'))):
                        pic=source_image.copy();active=mask>0
                        pic[active]=(.55*pic[active]+.45*colors[mask[active]]).astype('uint8')
                        strip.paste(Image.fromarray(pic),(col*960,row_index*560+20))
                        pen.text((col*960+8,row_index*560+4),f'{title} | frame {frame_index}',fill='black')
            strip.save(out/(key+'_temporal.jpg'))
        save(out/'comparison.json',dict(samples=stats,
            caveat='Manual masks are a paired reference, not ground truth. Visual source inspection still required.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','initialize','segment','sleeves','review'],required=True)
    p.add_argument('--policy',choices=['original','cross-target'],default='original')
    a=p.parse_args();run(a.source.resolve(),a.output.resolve(),a.stage,a.policy)
