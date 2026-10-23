"""Frozen random pilot of RGB retargeting; failures are retained, never replaced."""
from swam.paths import data_path, external_path
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(external_path('retarget'))
COMPARE = CACHE/'compare-env/bin/python'
RGB = CACHE/'rgb-env/bin/python'
SAM = Path(external_path('sam_env', 'bin/python'))
SAMROOT = Path(external_path('sam'))
VIDEO_DEPS = Path(external_path('video_deps'))


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def plan(args):
    import h5py
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    with zipfile.ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        members = sorted(n[:-4] for n in names if n.endswith('.mp4')
                         and not n.startswith('__MACOSX/') and n[:-4]+'.hdf5' in names)
        excluded=set(getattr(args,'exclude_tasks',[]))
        members=[m for m in members if m.split('/')[-2] not in excluded]
        distinct=getattr(args,'distinct_tasks',False)
        if distinct:
            tasks=sorted({m.split('/')[-2] for m in members})
            selected=[rng.choice([m for m in members if m.split('/')[-2]==task])
                      for task in rng.sample(tasks,args.count)]
        else:
            selected = rng.sample(members, args.count)
        manifest = dict(archive=str(args.archive.resolve()), seed=args.seed,
                        population_size=len(members), count=args.count,
                        selection='Uniform without replacement over paired test archive members, then uniform window start; no pose/quality filtering',
                        frames=args.frames, stride=2, clips=[],
                        scope='Short-window pilot, not full-sequence or dataset-wide validation',
                        original_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        if distinct:
            manifest['selection']='Uniform tasks without replacement, uniform paired clip within each task, then uniform time window; no output filtering'
        manifest['excluded_tasks']=sorted(excluded)
        for index, member in enumerate(selected):
            item = dict(member=member, key=f'{index:02d}_'+member.replace('/','_'), state='selected')
            manifest['clips'].append(item)
        save(output/'manifest.json', manifest)
        for item in manifest['clips']:
            source = output/'clips'/item['key']/'source'
            source.mkdir(parents=True)
            try:
                for suffix in ('mp4','hdf5'):
                    member = item['member']+'.'+suffix
                    item[suffix+'_crc'] = archive.getinfo(member).CRC
                    with archive.open(member) as src, (source/('source.'+suffix)).open('wb') as dst:
                        shutil.copyfileobj(src,dst)
                info = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                    '-show_entries','stream=width,height,r_frame_rate,nb_frames','-of','json',str(source/'source.mp4')]))['streams'][0]
                with h5py.File(source/'source.hdf5') as data:
                    total = len(data['transforms/camera'])
                if int(info['nb_frames']) != total:
                    raise ValueError('Video/annotation length mismatch')
                extent = 2*(args.frames-1)+1
                if total < extent:
                    raise ValueError('Source shorter than requested window; do not substitute another sample')
                start = rng.randrange(total-extent+1)
                indices = list(range(start,start+extent,2))
                num,den = map(int,info['r_frame_rate'].split('/'))
                config = dict(source_member=item['member'],indices=indices,
                              crop=[0,0,info['width'],info['height']],size=[512,288],fps=num/den/2,physics=False)
                save(source/'prepare.json',config)
                item.update(state='prepared',total_source_frames=total,window_start=start,
                            indices=indices,source_resolution=[info['width'],info['height']],fps=config['fps'])
            except Exception as error:
                item.update(state='failed_prepare',error=repr(error))
            save(output/'manifest.json',manifest)
            print(item,flush=True)


def command_for(stage, root, item):
    key=item['key']; case=root/'clips'/key
    source=case/'source'; predictions=root/'predictions'/key; hand=root/'hands'/key
    scripts=ROOT/'scripts'
    if stage=='estimate':
        # The estimator's root is new for each invocation, with one selected clip.
        return [RGB,scripts/'reestimate_egodex_rgb.py','--source-root',root/'clips','--clips',key,
                '--output',case/'estimate','--frames',str(len(item['indices']))]
    if stage in ('hands','hands_visible'):
        return [COMPARE,scripts/'render_rgb_robot_hands.py','--predictions',root/'predictions',
                '--output',case/('hand_fit' if stage=='hands' else 'hand_fit_visible'),
                '--clips',key,'--no-legacy-reference']+(['--allow-missing-hands'] if stage=='hands_visible' else [])
    if stage in ('segment','sleeves'):
        return [SAM,scripts/'sample_rgb_retarget_batch.py','--output',root,'--stage','track-one',
                '--key',key,'--mask-kind',stage]
    if stage in ('refine','smooth'):
        command=[COMPARE,scripts/'refine_rgb_hand_alignment.py','--source',source,'--base',hand,
                 '--segmentation',case/'segment','--predictions',predictions,'--all-frames',
                 '--output',case/stage]
        if stage=='refine': command += ['--sweeps','2','2','1','1','1','.5','.5','.25']
        else: command += ['--corrections-from',case/'refine']
        return command
    if stage=='arms':
        return [COMPARE,scripts/'fit_arms_to_frozen_hands.py','--source',source,'--base',hand,
                '--accepted',case/'smooth','--segmentation',case/'segment',
                '--sleeves',case/'sleeves','--output',case/'arms']
    raise ValueError(stage)


def prepare_short(args):
    import h5py
    root=args.output.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    for item in manifest['clips']:
        if item['state']!='failed_prepare' or 'Source shorter' not in item.get('error',''):
            continue
        source=root/'clips'/item['key']/'source'
        with h5py.File(source/'source.hdf5') as data:
            total=len(data['transforms/camera'])
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,r_frame_rate,nb_frames','-of','json',str(source/'source.mp4')]))['streams'][0]
        if int(info['nb_frames'])!=total or total<2:
            raise ValueError('Short source has invalid frame coverage')
        indices=list(range(0,total,2));num,den=map(int,info['r_frame_rate'].split('/'))
        save(source/'prepare.json',dict(source_member=item['member'],indices=indices,
             crop=[0,0,info['width'],info['height']],size=[512,288],fps=num/den/2,physics=False))
        item['preparation_recovery']=dict(original_error=item.pop('error'),
            policy='Keep the original random sample; process its entire shorter duration at unchanged stride two')
        item.update(state='prepared',total_source_frames=total,window_start=0,indices=indices,
                    source_resolution=[info['width'],info['height']],fps=num/den/2)
    save(root/'manifest.json',manifest)


def run(args):
    root=args.output.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    environment=dict(os.environ,PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    environment['PYTHONPATH']=os.pathsep.join(map(str,[ROOT/'src',CACHE/'objective-retargeting/src',VIDEO_DEPS]))
    for item in manifest['clips']:
        if args.key and item['key']!=args.key: continue
        if item['state']!='prepared': continue
        case=root/'clips'/item['key']; status=case/'status.json'
        record=json.loads(status.read_text()) if status.exists() else dict(member=item['member'],stages={})
        if args.stage in record['stages']:
            print('Already attempted; preserving',item['key'],args.stage,flush=True)
            continue
        command=list(map(str,command_for(args.stage,root,item)))
        entry=dict(state='running',command=command,started=time.time())
        record['stages'][args.stage]=entry;save(status,record)
        with (case/(args.stage+'.log')).open('x') as log:
            process=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        entry.update(state='complete' if process.returncode==0 else 'failed',returncode=process.returncode,finished=time.time())
        if process.returncode==0 and args.stage in ('estimate','hands','hands_visible'):
            destination=root/('predictions' if args.stage=='estimate' else 'hands')/item['key']
            destination.parent.mkdir(exist_ok=True)
            directory={'estimate':'estimate','hands':'hand_fit','hands_visible':'hand_fit_visible'}[args.stage]
            destination.symlink_to(case/directory/item['key'],target_is_directory=True)
        save(status,record)
        print(item['key'],args.stage,entry['state'],round(entry['finished']-entry['started'],1),'seconds',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--archive',type=Path,default=Path(data_path('egodex/test.zip')))
    p.add_argument('--stage',choices=['plan','prepare-short','estimate','hands','hands_visible','segment','sleeves','refine','smooth','arms','track-one'],required=True)
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261008)
    p.add_argument('--frames',type=int,default=33)
    p.add_argument('--key')
    p.add_argument('--mask-kind',choices=['segment','sleeves'])
    p.add_argument('--exclude-tasks',nargs='*',default=[])
    p.add_argument('--distinct-tasks',action='store_true')
    a=p.parse_args()
    if a.stage=='plan': plan(a)
    elif a.stage=='prepare-short': prepare_short(a)
    elif a.stage=='track-one':
        sys.path.insert(0,str(ROOT/'src'))
        from swam.retargeting.segmentation import track
        case=a.output/'clips'/a.key
        track(case/'source',case/a.mask_kind,case/(a.mask_kind+'_prompts.json'),
              sam_root=SAMROOT,native=True,output_size=(960,540))
    else: run(a)
