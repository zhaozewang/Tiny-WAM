"""Frozen task-disjoint retargeting evaluation; retain failures and stage timings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sample_rgb_retarget_batch import COMPARE,RGB,SAM,CACHE,VIDEO_DEPS,command_for,save
TIMING_LOCK=threading.Lock()


def execute(root,label,command):
    env=dict(os.environ,PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
             PYTHONPATH=os.pathsep.join(map(str,[ROOT/'src',CACHE/'objective-retargeting/src',VIDEO_DEPS])))
    path=root/'timings.json'
    command=list(map(str,command));record=dict(label=label,command=command,started=time.time(),status='running')
    with TIMING_LOCK:
        records=json.loads(path.read_text()) if path.exists() else []
        if any(r['label']==label for r in records):
            raise ValueError('Stage already attempted; preserve it: '+label)
        records.append(record);save(path,records)
    start=time.perf_counter()
    with (root/(label.replace('/','_')+'.log')).open('x') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(seconds=time.perf_counter()-start,returncode=result.returncode,
                  status='complete' if result.returncode==0 else 'failed')
    with TIMING_LOCK:
        records=json.loads(path.read_text());records=[record if r['label']==label else r for r in records]
        save(path,records)
    print(label,record['status'],round(record['seconds'],2),flush=True)
    return result.returncode==0


def freeze(root,tracking_v2=False,gpu_objective=False):
    manifest=json.loads((root/'manifest.json').read_text())
    protocol=dict(split=f"{len(manifest['clips'])} randomly sampled previously unused task families; one clip per task",
        seed=manifest['seed'],frames_per_clip=manifest['frames'],stride=2,render_size=[960,540],
        tracking_v2=tracking_v2,
        gpu_objective=gpu_objective,
        actual_frames_per_clip={item['key']:len(item['indices']) for item in manifest['clips']},
        prompts='One manual initialization on source frame zero only; no output-driven additions or fixes.',
        annotations='No EgoDex poses used for inference or mask prompting; not an independently labeled accuracy benchmark.',
        parameters=dict(wilor_detector_conf=.3,max_det=2,agnostic_nms=True,iou=.45,
                        refine_sweeps=[2,2,1,1,1,.5,.5,.25],arm_proximal_exit=True),
        failures='Retain every sampled clip and failed stage; no replacement samples.',
        speed='Exact-output optimizations tested on old development clips only; startup and per-stage wall time retained.',
        publication=False)
    save(root/'protocol.json',protocol)
    files={}
    for path in [*ROOT.glob('src/tinywam/retargeting/*.py'),*ROOT.glob('scripts/*rgb*.py'),
                 ROOT/'scripts/fit_arms_to_frozen_hands.py',ROOT/'scripts/repair_forearm_tracks.py']:
        dst=root/'source_snapshot'/path.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dst);files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(root/'code_hashes.json',files)


def sources(root,manifest):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    out=root/'source_review';out.mkdir(exist_ok=False)
    for row,item in enumerate(manifest['clips']):
        case=root/'clips'/item['key']
        with imageio.get_reader(case/'source/source.mp4') as reader:
            frame=Image.fromarray(reader.get_data(item['indices'][0])).resize((960,540))
        frame.save(out/(item['key']+'.png'))
        grid=frame.copy();d=ImageDraw.Draw(grid)
        for x in range(0,960,80):
            d.line([(x,0),(x,539)],fill=(140,140,140));d.text((x+2,2),str(x),fill='red')
        for y in range(0,540,60):
            d.line([(0,y),(959,y)],fill=(140,140,140));d.text((2,y+2),str(y),fill='red')
        grid.save(out/(item['key']+'_grid.png'))


def estimate(root,manifest):
    keys=[item['key'] for item in manifest['clips']]
    ok=execute(root,'estimate_all',[RGB,ROOT/'scripts/reestimate_egodex_rgb.py','--source-root',root/'clips',
               '--output',root/'estimate','--frames',str(manifest['frames']),'--clips',*keys])
    if not ok:raise RuntimeError('Estimation failed; inspect estimate_all.log')
    (root/'initial_predictions').mkdir()
    for key in keys:(root/'initial_predictions'/key).symlink_to(root/'estimate'/key,target_is_directory=True)


def prompts(root,manifest):
    seeds=json.loads((root/'source_seeds.json').read_text())
    for row,item in enumerate(manifest['clips']):
        case=root/'clips'/item['key'];seed=seeds[str(row)]
        for kind in ('segment','sleeves'):
            objects=[]
            for identity,name,positive,negative in seed[kind]:
                objects.append(dict(id=identity,name=name,role='object' if kind=='segment' and identity in (1,4) else 'hand',
                    annotations=[dict(frame=0,points=[[x*512/960,y*288/540] for x,y in positive+negative],
                                      labels=[1]*len(positive)+[0]*len(negative))]))
            save(case/(kind+'_prompts.json'),dict(size=[512,288],objects=objects,
                 annotation_source='One source-RGB frame-zero initialization before viewing outputs; no later corrections.'))


def run(root,manifest,stage,workers=1,tracking_v2=False,gpu_objective=False):
    if stage=='freeze':freeze(root,tracking_v2,gpu_objective)
    elif stage=='sources':sources(root,manifest)
    elif stage=='estimate':estimate(root,manifest)
    elif stage=='prompts':prompts(root,manifest)
    elif stage=='recover':
        execute(root,'recover_all',[RGB,Path(__file__),'--output',root,'--stage','recover-worker']+
                (['--tracking-v2'] if tracking_v2 else []))
    elif stage=='recover-worker':
        from repair_rgb_retarget_batch import recover
        recover(root,root,predictions_root=root/'initial_predictions',reconcile_identities=tracking_v2)
    elif stage in ('segment','sleeves'):
        for item in manifest['clips']:
            execute(root,item['key']+'/'+stage,command_for(stage,root,item))
    elif stage=='fit':
        (root/'hands').mkdir(exist_ok=True)
        def fit_case(item):
            key=item['key'];case=root/'clips'/key
            for name in ('hands_visible','refine','smooth','arms'):
                command=command_for(name,root,item)
                if name in ('refine','smooth'):command+=['--fast-mask-readback']
                if name in ('refine','smooth') and gpu_objective:command+=['--gpu-objective']
                if name=='arms':command+=['--proximal-exit','--fast-mask-readback']
                if name=='arms' and tracking_v2:command+=['--observed-forearm-fallback']
                if not execute(root,key+'/'+name,command):break
                if name=='hands_visible':(root/'hands'/key).symlink_to(case/'hand_fit_visible'/key,target_is_directory=True)
        save(root/'fit_schedule.json',dict(workers=workers,clip_independent=True,started=time.time()))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fit_case,manifest['clips']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['freeze','sources','estimate','prompts','segment','sleeves','fit','recover','recover-worker'],required=True)
    p.add_argument('--workers',type=int,default=1,choices=[1,2,4])
    p.add_argument('--tracking-v2',action='store_true')
    p.add_argument('--gpu-objective',action='store_true')
    a=p.parse_args();root=a.output.resolve()
    run(root,json.loads((root/'manifest.json').read_text()),a.stage,a.workers,a.tracking_v2,a.gpu_objective)
