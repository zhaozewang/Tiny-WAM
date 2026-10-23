"""Local semantic vocabulary and periodically grounded object proposals."""
import argparse
import json
from pathlib import Path
import time
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from sample_rgb_retarget_batch import save

VOCAB_PROMPT='''These are three frames from the same egocentric manipulation video.
Return ONLY a JSON list of short English object category names for ALL physical
objects being held, moved, opened, used as tools, or acted upon by the hands.
Include small tools as well as their target objects. Include objects visible in
any frame, even if partly covered. Use simple visual nouns suitable for an object
detector (no actions, colors, locations, or long referring expressions). Do not
include hands, people, arms, clothes, floors, tables, walls, or background decor.
Do not give coordinates. At most 12 category names. No markdown or explanation.'''


def frames(case,item,indices):
    with imageio.get_reader(case/'source/source.mp4') as reader:
        for i in indices:
            yield i,Image.fromarray(reader.get_data(item['indices'][i])).convert('RGB')


def vocabulary(root,model_path):
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    torch.set_num_threads(4)
    out=root/'vocabulary';out.mkdir(exist_ok=False)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,
        dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
    manifest=json.loads((root/'manifest.json').read_text())
    for item in manifest['clips']:
        key=item['key'];n=len(item['indices']);indices=sorted({0,n//2,n-1})
        content=[dict(type='image',image=im.resize((960,540))) for _,im in frames(root/'clips'/key,item,indices)]
        content.append(dict(type='text',text=VOCAB_PROMPT))
        inputs=processor.apply_chat_template([dict(role='user',content=content)],tokenize=True,
            add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False).to(model.device)
        with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=256,do_sample=False)
        raw=processor.decode(tokens[0,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
        (out/(key+'.txt')).write_text(raw)
        if raw.strip().startswith('```'):raw='\n'.join(raw.strip().splitlines()[1:-1])
        names=json.loads(raw)
        if not isinstance(names,list) or not 1<=len(names)<=12 or not all(isinstance(x,str) and 0<len(x)<80 for x in names):
            raise ValueError('Invalid object vocabulary: '+key)
        save(out/(key+'.json'),dict(names=names,frames=indices,prompt=VOCAB_PROMPT,model=str(model_path),local_only=True))
        print(key,names,flush=True)


def detect(root,model_path,interval):
    import torch
    from torchvision.ops import nms
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    torch.set_num_threads(4)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(model_path,local_files_only=True).cuda().eval()
    manifest=json.loads((root/'manifest.json').read_text())
    for item in manifest['clips']:
        key=item['key'];case=root/'clips'/key;out=case/'grounded_objects';out.mkdir(exist_ok=False)
        names=json.loads((root/'vocabulary'/(key+'.json')).read_text())['names']
        excluded={'hand','hands','arm','arms','person','people','clothes','floor','table','wall','background'}
        names=[name for name in names if not set(name.lower().split())&excluded]
        task=item['member'].split('/')[-2].replace('_',' ')
        queries=names+[task]
        records=[];n=len(item['indices']);start=time.perf_counter()
        for index,im in frames(case,item,sorted(set(range(0,n,interval))|{n-1})):
            inputs=processor(images=im,text='. '.join(queries)+'.',return_tensors='pt').to('cuda')
            with torch.inference_mode():pred=model(**inputs)
            result=processor.post_process_grounded_object_detection(pred,inputs.input_ids,
                threshold=.25,text_threshold=.25,target_sizes=[im.size[::-1]])[0]
            boxes=result['boxes'].cpu();scores=result['scores'].cpu()
            labels=result.get('text_labels',result.get('labels'))
            keep=nms(boxes,scores,.75).tolist()
            rows=[]
            for j in keep:
                box=boxes[j].tolist();w,h=im.size
                box=[max(0,min(v,(w-1 if k%2==0 else h-1))) for k,v in enumerate(box)]
                if box[2]<=box[0] or box[3]<=box[1]:continue
                rows.append(dict(box=box,score=float(scores[j]),name=str(labels[j])))
            records.append(dict(frame=index,boxes=rows,size=list(im.size)))
            if index in (0,n-1) or abs(index-n//2)<interval:
                pic=im.resize((960,540));draw=ImageDraw.Draw(pic)
                for r in rows:
                    b=np.array(r['box'])*np.array([960/im.width,540/im.height]*2)
                    draw.rectangle(b.tolist(),outline='red',width=2);draw.text(tuple(b[:2]),r['name'],fill='red')
                pic.save(out/f'boxes_{index:05d}.jpg')
        save(out/'detections.json',dict(model=str(model_path),queries=queries,
            task_context_source='Recorded task category only; no poses or coordinates',threshold=.25,text_threshold=.25,
            nms_iou=.75,interval=interval,records=records,seconds=time.perf_counter()-start))
        if not any(r['boxes'] for r in records):raise ValueError('No object detected: '+key)
        print(key,len(records),'grounded frames',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['vocabulary','detect'],required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--interval',type=int,default=16)
    a=p.parse_args()
    if a.stage=='vocabulary':vocabulary(a.output.resolve(),a.model.resolve())
    else:detect(a.output.resolve(),a.model.resolve(),a.interval)
