"""Full-video production: preserve per-video failures without stopping the queue."""
from tinywam.paths import external_path
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
from sample_rgb_retarget_batch import ROOT,COMPARE,RGB,SAM,save,command_for
from evaluate_rgb_heldout import execute,sources
from evaluate_rgb_tracking_v3 import run as tracking
from initialize_rgb_prompts import cross_target_negatives
from plan_rgb_production import prepare

PROMPT_PY=Path(external_path('retarget', 'prompt-env/bin/python'))
QWEN=Path(external_path('prompt_model'))
DINO=Path(external_path('dino_model'))
VARIANT='automatic_grounded_v1'


def step(root,label,command,artifact):
    timings=json.loads((root/'timings.json').read_text()) if (root/'timings.json').exists() else []
    previous=[r for r in timings if r['label']==label]
    if previous:
        if previous[0]['status']!='complete' or not artifact.exists():
            raise RuntimeError('Previously failed/incomplete stage; diagnose without retry: '+label)
        return
    if not execute(root,label,command) or not artifact.exists():raise RuntimeError('Failed stage: '+label)


def write_prompts(root,item):
    raw=json.loads((root/'source_seeds.json').read_text())['0']
    value=cross_target_negatives(raw);case=root/'clips'/item['key']
    for kind in ('segment','sleeves'):
        objects=[]
        for identity,name,positive,negative in value[kind]:
            objects.append(dict(id=identity,name=name,role='object' if kind=='segment' and identity in (1,4) else 'hand',
                annotations=[dict(frame=0,points=[[x*512/960,y*288/540] for x,y in positive+negative],
                                  labels=[1]*len(positive)+[0]*len(negative))]))
        path=case/(kind+'_prompts.json')
        if not path.exists():save(path,dict(size=[512,288],objects=objects,annotation_source='Local VLM positive points with cross-target negatives. Objects replaced by periodic Grounded SAM. No manual points.'))


def run_case(collection,index):
    prepare(collection,index+1,only_index=index)
    manifest=json.loads((collection/'manifest.json').read_text());item=manifest['clips'][index];key=item['key']
    directory=collection/VARIANT/key;root=directory/'input';result=directory/'result'
    if not root.exists():
        root.mkdir(parents=True)
        subset=dict(manifest,clips=[item],count=1,frames=len(item['indices']))
        save(root/'manifest.json',subset)
        case=root/'clips'/key;case.mkdir(parents=True)
        (case/'source').symlink_to(collection/'clips'/key/'source',target_is_directory=True)
    subset=json.loads((root/'manifest.json').read_text());case=root/'clips'/key
    if subset['clips'][0]['indices']!=item['indices']:raise ValueError('Changed full-video frame ordering')
    if not (root/'source_review').exists():sources(root,subset)
    failure=root/'automatic_prompt_audit/failure.json'
    if failure.exists() and json.loads(failure.read_text()).get('error')=="ValueError('No observed forearm initialized; requires review')":
        step(root,'recover_missing_forearm_seed',[COMPARE,ROOT/'scripts/recover_rgb_initialization.py','--output',root,'--stage','seeds'],root/'source_seeds.json')
    else:
        step(root,'automatic_initialize',[PROMPT_PY,ROOT/'scripts/initialize_rgb_prompts.py','--output',root,'--allow-uninitialized-forearms'],root/'source_seeds.json')
    write_prompts(root,item)
    step(root,'vocabulary',[PROMPT_PY,ROOT/'scripts/ground_rgb_objects.py','--output',root,'--stage','vocabulary','--model',QWEN],root/'vocabulary'/(key+'.json'))
    step(root,'ground_objects',[PROMPT_PY,ROOT/'scripts/ground_rgb_objects.py','--output',root,'--stage','detect','--model',DINO],case/'grounded_objects/detections.json')
    step(root,'grounded_track',[SAM,ROOT/'scripts/track_grounded_rgb.py','--output',root,'--key',key,'--prompt-source',root],case/'segment/segmentation.json')
    if json.loads((case/'sleeves_prompts.json').read_text())['objects']:
        step(root,'sleeves',command_for('sleeves',root,item),case/'sleeves/segmentation.json')
    else:
        step(root,'defer_missing_sleeves',[COMPARE,ROOT/'scripts/recover_rgb_initialization.py','--output',root,'--stage','empty-sleeves','--key',key],case/'sleeves/segmentation.json')
    step(root,'estimate_all',[RGB,ROOT/'scripts/reestimate_egodex_rgb.py','--source-root',root/'clips','--output',root/'estimate',
         '--clips',key,'--frames',str(len(item['indices']))],root/'estimate'/key/'summary.json')
    (root/'initial_predictions').mkdir(exist_ok=True)
    link=root/'initial_predictions'/key
    if not link.exists():link.symlink_to(root/'estimate'/key,target_is_directory=True)
    if not result.exists():
        tracking(root,result,'prepare')
        save(result/'provenance.json',dict(source=str(root),split='random training source; complete original timeline',
            initialization='Local Qwen positive points, Grounding DINO + SAM2 periodic object refresh every 16 frames',
            task_category_used=True,manual_points=False,hand_arm_fitting_baseline='tracking_v3',
            changes=['automatic segmentation only'],queue_policy='Isolate sample failures; continue other selected videos',publication=False))
        tracking(root,result,'freeze')
        protocol=json.loads((result/'protocol.json').read_text())
        protocol.update(prompts='Automatic RGB prompts and periodic object refresh, then unchanged pose-guided forearm repair.',
            full_original_timeline=True,initialization_variant=VARIANT)
        save(result/'protocol.json',protocol)
    step(result,'recover_all',[RGB,ROOT/'scripts/evaluate_rgb_tracking_v3.py','--source',root,'--output',result,'--stage','recover-worker'],result/'predictions'/key/'summary.json')
    (result/'hands').mkdir(exist_ok=True);target=result/'clips'/key
    for stage,artifact in [('hands_visible',target/'hand_fit_visible'/key/'report.json'),('refine',target/'refine/report.json'),('smooth',target/'smooth/report.json')]:
        cmd=command_for(stage,result,item)
        if stage!='hands_visible':cmd+=['--gpu-objective']
        step(result,key+'/'+stage,cmd,artifact)
        if stage=='hands_visible' and not (result/'hands'/key).exists():
            (result/'hands'/key).symlink_to(target/'hand_fit_visible'/key,target_is_directory=True)
    step(result,key+'/repair_sleeves',[COMPARE,ROOT/'scripts/repair_forearm_tracks.py','--output',result,'--source',root,'--key',key],target/'sleeves/labels.npy')
    step(result,key+'/arms',command_for('arms',result,item)+['--proximal-exit','--fast-mask-readback','--validated-forearm'],target/'arms/report.json')
    step(result,'review',[COMPARE,ROOT/'scripts/review_rgb_retarget_batch.py','--output',result],result/'review/verification.json')
    verification=json.loads((result/'review/verification.json').read_text())
    rows=verification['samples'] if isinstance(verification,dict) else verification
    if len(rows)!=1 or not rows[0].get('processing_verified') or rows[0]['frames']!=len(item['indices']):
        raise RuntimeError('Full-duration output verification failed')
    record=dict(key=key,frames=len(item['indices']),complete=True,software_verified=True,
        visual_review='pending',result=str(result),finished=time.time())
    save(directory/'complete.json',record)
    return record


def run(collection,index,bulk):
    out=collection/VARIANT;out.mkdir(exist_ok=True)
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=json.loads((collection/'manifest.json').read_text())
        if bulk:
            release_path=out/'release_queue_recovery.json'
            if not release_path.exists():release_path=out/'release.json'
            release=json.loads(release_path.read_text())
            if not release.get('approved') or not release.get('full_duration_pilot_visual_pass'):
                raise ValueError('No accepted full-duration pilot release')
            if not release.get('code_hashes') or not release.get('pilots'):
                raise ValueError('Release lacks source hashes or verified pilots')
            for pilot in release['pilots']:
                record=json.loads((out/pilot/'complete.json').read_text())
                selected=next(item for item in manifest['clips'] if item['key']==pilot)
                if not record.get('software_verified') or record.get('visual_review')!='accepted' or record['frames']!=selected['frames_at_stride_two']:
                    raise ValueError('Pilot is not full-duration and accepted: '+pilot)
                video=Path(record['result'])/'review'/(pilot+'.mp4')
                if hashlib.sha256(video.read_bytes()).hexdigest()!=release['pilot_video_sha256'][pilot]:
                    raise ValueError('Pilot media differs from reviewed output: '+pilot)
            for name,sha in release['code_hashes'].items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Source changed after release: '+name)
        indices=range(len(manifest['clips'])) if bulk else [index]
        started=time.time();records=[];failed=[]
        for i in indices:
            key=manifest['clips'][i]['key'];complete=out/key/'complete.json'
            if complete.exists():records.append(json.loads(complete.read_text()));continue
            failure_path=out/key/'failed.json'
            if failure_path.exists():failed.append(json.loads(failure_path.read_text()));continue
            status=dict(state='running',variant=VARIANT,current=key,requested=len(manifest['clips']) if bulk else 1,
                        software_verified=len(records),visual_verified=sum(r.get('visual_review')=='accepted' for r in records),
                        failed=len(failed),failures=failed,started=started,bulk=bulk)
            save(out/'status.json',status)
            if shutil.disk_usage(collection).free<50*1024**3:
                save(out/'status.json',dict(status,state='storage_exhausted',error='Less than 50 GiB free; cannot safely write outputs'))
                raise RuntimeError('Output storage exhausted')
            try:
                records.append(run_case(collection,i))
            except Exception as error:
                entry=dict(key=key,error=repr(error),failed_at=time.time(),state='isolated_needs_repair',complete=False)
                failure_path.parent.mkdir(parents=True,exist_ok=True);save(failure_path,entry);failed.append(entry)
                print('ISOLATED',key,repr(error),'continuing remaining videos',flush=True)
                save(out/'status.json',dict(status,state='continuing_after_sample_failure',failed=len(failed),failures=failed,current=None))
                if not bulk:raise
        save(out/'status.json',dict(state='finished_with_failures' if failed else 'software_complete_visual_review_pending',bulk=bulk,
            software_verified=len(records),visual_verified=sum(r.get('visual_review')=='accepted' for r in records),
            failed=len(failed),failures=failed,records=records,seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--index',type=int,default=0);p.add_argument('--bulk',action='store_true')
    a=p.parse_args();run(a.collection.resolve(),a.index,a.bulk)
