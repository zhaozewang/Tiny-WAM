"""Local VLM first-frame prompts for a separately evaluated automatic variant."""
from swam.paths import external_path
import argparse
import copy
import json
from pathlib import Path
import time
from PIL import Image

PROMPT='''Annotate this egocentric manipulation image for point-prompted segmentation.
Return ONLY JSON, with keys "segment" and "sleeves". Each value is a list of
objects with keys "id", "name", "positive_points", "negative_points".
Each points list is [[x1,y1],[x2,y2]], normalized to 0..1000 (not pixel units).
segment: id 2 = the wearer's LEFT hand, id 3 = RIGHT hand, id 1 and optionally 4 =
the manipulated foreground objects that may occlude the robot hands. Use at most
two objects. Do not label the table, floor, walls, or large background as objects.
IMPORTANT: include the held/manipulated object as id 1, BEFORE the hands. Object
segmentation is essential to prevent robot hands from covering the object. If
there is a second nearby manipulated object, include it as id 4. Do not omit a
visible object just because it is partly covered by a hand.
sleeves: id 1 = LEFT forearm or sleeve, id 2 = RIGHT forearm or sleeve.
For hands use 2 positive points inside the visible palm/fingers, not on objects.
For objects use 2 positive points well inside the visible object, not on skin.
For forearms use 2 positive points on the proximal arm/sleeve, NOT on the hand.
Add 2 negative points on immediately adjacent competing hand/object regions.
Never invent a missing or fully offscreen hand/arm/object. Empty lists are allowed.
All points must be visibly on the intended surface, not at bounding-box centers
that fall outside it. No explanation, no markdown, no additional keys.'''

POSITIVE_PROMPT='''Annotate this egocentric manipulation image for segmentation.
Return ONLY JSON with keys "segment" and "sleeves", each a list of objects with
keys "id", "name", "positive_points", "negative_points". Coordinates are
normalized to 0..1000. Every negative_points list MUST be empty; exclusion points
will be constructed automatically from competing targets, not guessed.
segment: id 2 = wearer's LEFT hand, id 3 = RIGHT hand; id 1 and id 4 = manipulated
objects. Identify the small tool or object actually held between the fingers
FIRST (id 1), not just a nearby larger container or surface. Include the other
object being manipulated or acted upon as id 4 when visible. Use both object slots
when the interaction involves two different objects. Do not label background
walls, tables or floors. Use two positive points on different visible parts of
each object, avoiding skin, shadows, holes and background. A partially occluded
object is still an object. For each hand use two positive points inside visible
palm or fingers, never on the held object.
sleeves: id 1 = LEFT forearm or sleeve, id 2 = RIGHT forearm or sleeve. Use two
positive points inside each visible proximal forearm, not on hand or background.
Do not invent invisible regions. Omit a fully offscreen target. No explanation,
markdown or additional keys.'''


def validate(raw,require_forearm=True):
    text=raw.strip()
    if text.startswith('```'):
        lines=text.splitlines();text='\n'.join(lines[1:-1])
    value=json.loads(text)
    if set(value)!={'segment','sleeves'}:raise ValueError('Unexpected prompt keys')
    for kind,allowed in [('segment',{1,2,3,4}),('sleeves',{1,2})]:
        used=set()
        normalized=[]
        for row in value[kind]:
            if isinstance(row,dict):
                if set(row)!={'id','name','positive_points','negative_points'}:
                    raise ValueError('Invalid annotation fields')
                row=[row['id'],row['name'],row['positive_points'],row['negative_points']]
            if not isinstance(row,list) or len(row)!=4:raise ValueError('Invalid annotation row')
            identity,name,pos,neg=row
            if identity not in allowed or identity in used:raise ValueError('Invalid/duplicate identity')
            if not isinstance(name,str) or not pos:raise ValueError('Missing name/positive point')
            used.add(identity)
            for points in (pos,neg):
                if points and all(isinstance(x,(int,float)) for x in points):
                    if len(points)%2:raise ValueError('Odd flattened coordinate count')
                    points[:]=[points[j:j+2] for j in range(0,len(points),2)]
            for xy in pos+neg:
                if len(xy)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not 0<=v<=1000 for v in xy):
                    raise ValueError('Point outside normalized image')
            normalized.append(row)
        value[kind]=normalized
        # Existing evaluator converts 960x540 annotation coordinates to prepared size.
        for row in value[kind]:
            for points in row[2:]:
                for xy in points:xy[:]=[xy[0]*959/1000,xy[1]*539/1000]
    if not any(r[0] in (2,3) for r in value['segment']):raise ValueError('No visible hand initialized')
    if not any(r[0] in (1,4) for r in value['segment']):raise ValueError('No manipulated object initialized; requires review')
    if require_forearm and not value['sleeves']:raise ValueError('No observed forearm initialized; requires review')
    return value


def cross_target_negatives(value):
    """Use observed competing targets, not free-form background guesses."""
    value=copy.deepcopy(value)
    for kind in ('segment','sleeves'):
        for row in value[kind]:
            if kind=='segment':
                competitors=[r for r in value['segment'] if r[0]!=row[0]]
            else:
                competitors=value['segment']
            negative=[]
            for other in competitors:
                for point in other[2]:
                    if all((point[0]-p[0])**2+(point[1]-p[1])**2>16**2 for p in row[2]):
                        negative.append(point[:])
            row[3]=negative
    return value


def run(root,model_path,policy='original',allow_uninitialized_forearms=False):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.set_num_threads(4)
    audit=root/'automatic_prompt_audit';audit.mkdir(exist_ok=False)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,
        torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
    manifest=json.loads((root/'manifest.json').read_text());seeds={};reports=[]
    prompt=POSITIVE_PROMPT if policy=='cross-target' else PROMPT
    for i,item in enumerate(manifest['clips']):
        image_path=root/'source_review'/(item['key']+'.png')
        image=Image.open(image_path).convert('RGB')
        messages=[dict(role='user',content=[dict(type='image',image=image),dict(type='text',text=prompt)])]
        start=time.perf_counter()
        inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors='pt',enable_thinking=False).to(model.device)
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=1300,do_sample=False)
        raw=processor.decode(output[0,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
        (audit/(item['key']+'.txt')).write_text(raw)
        try:
            seeds[str(i)]=validate(raw,require_forearm=not allow_uninitialized_forearms)
            if policy=='cross-target':seeds[str(i)]=cross_target_negatives(seeds[str(i)])
        except Exception as error:
            (audit/'failure.json').write_text(json.dumps(dict(key=item['key'],error=repr(error))))
            raise
        reports.append(dict(key=item['key'],seconds=time.perf_counter()-start))
        print(item['key'],reports[-1]['seconds'],flush=True)
    with (root/'source_seeds.json').open('x') as f:json.dump(seeds,f,indent=2)
    (audit/'provenance.json').write_text(json.dumps(dict(model=str(model_path),prompt=prompt,policy=policy,
        local_only=True,no_manual_edits=True,frames='source frame zero only',
        sampling='greedy',reports=reports,allow_uninitialized_forearms=allow_uninitialized_forearms,
        missing_forearm_policy='No fabricated mask; initialize later from frozen wrist/palm using existing repair stage'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path(external_path('prompt_model')))
    p.add_argument('--policy',choices=['original','cross-target'],default='original')
    p.add_argument('--allow-uninitialized-forearms',action='store_true')
    a=p.parse_args();run(a.output.resolve(),a.model.resolve(),a.policy,a.allow_uninitialized_forearms)
