"""Immutable, resumable full-video selection from training archives only."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import subprocess
import zipfile
import h5py
from sample_rgb_retarget_batch import ROOT, save


def plan(root, archives, count, seed):
    population={}
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            names=set(archive.namelist())
            for name in sorted(names):
                if name.endswith('.mp4') and name[:-4]+'.hdf5' in names and not name.startswith('__MACOSX/'):
                    member=name[:-4]; task=member.split('/')[-2]
                    population.setdefault(task,[]).append((str(path.resolve()),member))
    if count>len(population): raise ValueError('Requested more distinct tasks than available')
    rng=random.Random(seed)
    chosen=[rng.choice(population[t]) for t in rng.sample(sorted(population),count)]
    root.mkdir(parents=True,exist_ok=False)
    rows=[]
    for i,(path,member) in enumerate(chosen):
        with zipfile.ZipFile(path) as archive:
            info=archive.getinfo(member+'.mp4'); meta=archive.getinfo(member+'.hdf5')
            with h5py.File(io.BytesIO(archive.read(meta))) as data:
                total=len(data['transforms/camera'])
            rows.append(dict(key=f'{i:03d}_'+member.replace('/','_'),archive=path,member=member,
                state='selected',source_frames=total,frames_at_stride_two=len(range(0,total,2)),
                source_bytes=info.file_size+meta.file_size,mp4_crc=info.CRC,hdf5_crc=meta.CRC))
    hashes={}
    for path in [*ROOT.glob('src/tinywam/retargeting/*.py'),*ROOT.glob('scripts/*rgb*.py'),
                 ROOT/'scripts/fit_arms_to_frozen_hands.py',ROOT/'scripts/repair_forearm_tracks.py']:
        relative=path.relative_to(ROOT); dst=root/'accepted_baseline_snapshot'/relative
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
        hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(root/'baseline_hashes.json',hashes)
    manifest=dict(seed=seed,count=count,clips=rows,frames=max(x['frames_at_stride_two'] for x in rows),
        selection='Uniform distinct training task labels without replacement, then uniform paired member in each task. No duration/quality filtering or replacement.',
        window='Complete original timeline, frame zero through last available stride-two frame',
        stride=2,render_size=[960,540],test_archive_used=False,
        baseline='rgb_tracking_v3_fresh_20261008, accepted by user',
        initialization='Automatic variant requires paired pilot and full-duration validation before bulk release',
        bulk_release=False)
    save(root/'manifest.json',manifest)
    save(root/'status.json',dict(state='planned_waiting_automatic_initialization_validation',
        requested=count,complete=0,verified=0,failed=0,
        projected_frames=sum(x['frames_at_stride_two'] for x in rows),
        estimated_source_bytes=sum(x['source_bytes'] for x in rows)))
    print(json.dumps(json.loads((root/'status.json').read_text()),indent=2))


def prepare(root,limit,only_index=None):
    manifest=json.loads((root/'manifest.json').read_text())
    items=manifest['clips'][:limit] if only_index is None else [manifest['clips'][only_index]]
    for item in items:
        if item['state']=='prepared': continue
        if item['state']!='selected': raise RuntimeError('Preserve failed preparation; inspect before retry')
        source=root/'clips'/item['key']/'source'; source.mkdir(parents=True,exist_ok=False)
        try:
            with zipfile.ZipFile(item['archive']) as archive:
                for suffix in ('mp4','hdf5'):
                    with archive.open(item['member']+'.'+suffix) as src, (source/('source.'+suffix)).open('xb') as dst:
                        shutil.copyfileobj(src,dst)
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=width,height,r_frame_rate,nb_frames','-of','json',str(source/'source.mp4')]))['streams'][0]
            if int(info['nb_frames'])!=item['source_frames']:raise ValueError('Video and metadata frame counts differ')
            num,den=map(int,info['r_frame_rate'].split('/'));indices=list(range(0,item['source_frames'],2))
            save(source/'prepare.json',dict(source_member=item['member'],indices=indices,
                crop=[0,0,info['width'],info['height']],size=[512,288],fps=num/den/2,physics=False))
            item.update(state='prepared',indices=indices,total_source_frames=item['source_frames'],window_start=0,
                        source_resolution=[info['width'],info['height']],fps=num/den/2)
        except Exception as error:
            item.update(state='failed_prepare',error=repr(error))
            save(root/'manifest.json',manifest);raise
        save(root/'manifest.json',manifest)
        print(item['key'],len(indices),'frames',len(indices)/item['fps'],'seconds',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['plan','prepare'],required=True)
    p.add_argument('--archives',type=Path,nargs='+',default=[])
    p.add_argument('--count',type=int,default=100);p.add_argument('--seed',type=int,default=6100804)
    p.add_argument('--limit',type=int,default=1)
    a=p.parse_args();root=a.output.resolve()
    if a.stage=='plan':plan(root,a.archives,a.count,a.seed)
    else:prepare(root,a.limit)
