"""Transactional RGB-lossless MP4 migration of completed dataset shards."""
from tinywam.paths import data_path
import argparse,fcntl,hashlib,io,json,os,shutil,subprocess,tarfile,time,traceback
from pathlib import Path
import numpy as np
from tinywam.shards import atomic_json,sha256,publish_episode

import threading
COMMIT_LOCK=threading.Lock()
VIEWS=('head','left','right')

def decoded_digest(path):
    p=subprocess.Popen(['ffmpeg','-v','error','-threads','2','-i',str(path),'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-threads','2','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    h=hashlib.sha256();size=0
    for chunk in iter(lambda:p.stdout.read(1024*1024),b''):h.update(chunk);size+=len(chunk)
    assert p.wait()==0, f'Decode failed: {path}'
    return h.hexdigest(),size

def refresh_index(root):
    rows=[json.loads(p.read_text()) for p in sorted((root/'shards').glob('episode_*.json')) if p.with_suffix('.tar').exists()]
    index=root/'index.json';old=json.loads(index.read_text()) if index.exists() else {}
    old['shards']=rows;atomic_json(index,old)

def _finish_transaction(source,transaction):
    record=json.loads(transaction.read_text());partial=Path(record['partial']);side=source.with_suffix('.json')
    if partial.exists():
        assert sha256(partial)==record['manifest']['sha256']
        os.replace(partial,source)  # Complete, verified replacement; old inode removed atomically.
    else:assert sha256(source)==record['manifest']['sha256']
    atomic_json(side,record['manifest']);refresh_index(source.parent.parent)
    transaction.unlink();shutil.rmtree(record['work'],ignore_errors=True)
    return record['manifest']

def finish_transaction(source,transaction):
    with COMMIT_LOCK:return _finish_transaction(source,transaction)


def convert(source,status):
    root=source.parent.parent;transaction=source.with_suffix('.mp4-transaction.json')
    if transaction.exists():return finish_transaction(source,transaction)
    manifest=json.loads(source.with_suffix('.json').read_text())
    if manifest.get('storage_format')=='mp4_rgb_lossless_v1':return manifest
    started=time.time();initial_stat=source.stat();work=root/'mp4_conversion'/source.stem;work.mkdir(parents=True,exist_ok=True)
    # An interrupted encode has no committed output; the raw source is intact.
    for p in work.iterdir():
        if p.is_file():p.unlink()
    writers=[];logs=[];digests=[hashlib.sha256() for _ in VIEWS];frames=0;previous=None
    try:
        with tarfile.open(source) as archive:
            blocks=sorted(m.name for m in archive if m.name.startswith('block_') and m.name.endswith('.npz'))
            assert len(blocks)==manifest['blocks']
            for member in archive:
                if member.name in blocks:continue
                assert Path(member.name).name==member.name and member.isfile(), member.name
                (work/member.name).write_bytes(archive.extractfile(member).read())
            for i,name in enumerate(blocks):
                with np.load(io.BytesIO(archive.extractfile(name).read())) as z:d={k:z[k] for k in z.files}
                rgb=d.pop('rgb');assert rgb.dtype==np.uint8 and rgb.ndim==5 and rgb.shape[1]==3 and rgb.shape[-1]==3
                assert len(rgb)==len(d['actions'])+1
                assert np.array_equal(d['frame_indices'],np.arange(i*manifest['block_actions'],i*manifest['block_actions']+len(rgb)))
                if previous is not None:
                    for key in ('qpos','qvel'):assert np.array_equal(previous[key],d[key][0])
                    assert np.array_equal(previous['rgb'],rgb[0])
                previous={key:d[key][-1].copy() for key in ('qpos','qvel')};previous['rgb']=rgb[-1].copy()
                if not writers:
                    height,width=rgb.shape[2:4]
                    for view in VIEWS:
                        log=(work/f'{view}.ffmpeg.log').open('wb');logs.append(log)
                        writers.append(subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(manifest['fps']),'-i','pipe:0','-an','-c:v','libx264rgb','-crf','0','-preset','veryfast','-threads','2','-g','24','-movflags','+faststart',str(work/f'{view}.mp4')],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log))
                fresh=rgb if i==0 else rgb[1:]
                for v,p in enumerate(writers):
                    raw=np.ascontiguousarray(fresh[:,v]).tobytes();digests[v].update(raw);p.stdin.write(raw)
                frames+=len(fresh)
                np.savez_compressed(work/name,**d)
                with np.load(work/name) as saved:
                    assert set(saved.files)==set(d)
                    for key,value in d.items():assert np.array_equal(saved[key],value),key
                atomic_json(status,dict(stage='encoding',source=str(source),block=i+1,blocks=len(blocks),elapsed_seconds=time.time()-started,updated=time.time()))
        for p in writers:p.stdin.close()
        for p in writers:assert p.wait()==0,'Video encoder failed; original shard retained'
        for log in logs:log.close()
        assert frames==manifest['observations']==manifest['actions']+1
        atomic_json(status,dict(stage='verifying_pixels',source=str(source),frames=frames,updated=time.time()))
        hashes={}
        for v,view in enumerate(VIEWS):
            digest,size=decoded_digest(work/f'{view}.mp4')
            assert size==frames*height*width*3 and digest==digests[v].hexdigest(), 'RGB roundtrip mismatch'
            hashes[view]=digest
        storage=dict(format='mp4_rgb_lossless_v1',codec='libx264rgb',crf=0,preset='veryfast',views=list(VIEWS),frames=frames,resolution=[height,width],fps=manifest['fps'],decoded_rgb_sha256=hashes,original_shard_sha256=manifest['sha256'],original_bytes=initial_stat.st_size,includes_terminal_observation=True)
        atomic_json(work/'storage.json',storage)
        partial=source.with_suffix('.mp4.tar.partial')
        with tarfile.open(partial,'w') as out:
            for p in sorted(work.iterdir()):
                if p.suffix!='.log':out.add(p,arcname=p.name,recursive=False)
        with partial.open('rb') as stream:os.fsync(stream.fileno())
        # Verify packaged payloads against the validated loose files before replacing anything.
        with tarfile.open(partial) as out:
            for member in out:
                digest=hashlib.sha256()
                stream=out.extractfile(member)
                for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
                assert digest.hexdigest()==sha256(work/member.name),member.name
        assert source.stat().st_size==initial_stat.st_size and source.stat().st_mtime_ns==initial_stat.st_mtime_ns
        new=dict(manifest,storage_format=storage['format'],original_sha256=manifest['sha256'],original_bytes=initial_stat.st_size,bytes=partial.stat().st_size,sha256=sha256(partial),rgb_verified_lossless=True,conversion_seconds=time.time()-started)
        backup=root/'mp4_conversion'/(source.stem+'.original_manifest.json');atomic_json(backup,manifest)
        atomic_json(transaction,dict(partial=str(partial),work=str(work),manifest=new))
        return finish_transaction(source,transaction)
    finally:
        for p in writers:
            if p.poll() is None:p.kill();p.wait()
        for log in logs:
            if not log.closed:log.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path(data_path('tinywam')));p.add_argument('--dataset');a=p.parse_args()
    status=a.base/'mp4_conversion_status.json';lock=(a.base/'.mp4_conversion.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    roots=[a.base/a.dataset] if a.dataset else sorted(r for r in a.base.iterdir() if r.is_dir())
    jobs=[];dataset_locks=[]
    for root in roots:
        # Only complete trajectories are eligible, including finished historical pilot data.
        if (root/'worker.lock').exists():
            l=(root/'worker.lock').open('a');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB);dataset_locks.append(l)
        for folder in sorted((root/'staging').glob('episode_*')):
            if not (folder/'complete.json').exists():continue
            meta=json.loads((folder/'metadata.json').read_text())
            if meta.get('physics_only'):continue
            if not (root/'shards'/f'{folder.name}.tar').exists():publish_episode(folder,root/'shards')
        jobs.extend(sorted((root/'shards').glob('episode_*.tar')))
    atomic_json(a.base/'mp4_conversion_plan.json',dict(shards=[str(j) for j in jobs],count=len(jobs),format='mp4_rgb_lossless_v1',created=time.time()))
    from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
    records=[];progress=a.base/'mp4_conversion_progress';progress.mkdir(exist_ok=True)
    # Serialize final manifest/index publication; encoding may run concurrently.
    jobs.sort(key=lambda p:(p.parent.parent.name!='r1_random_10min_v5',str(p)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(convert,source,progress/(source.parent.parent.name+'_'+source.stem+'.json')):source for source in jobs}
        try:
            while pending:
                done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for task in done:
                    source=pending.pop(task);result=task.result();records.append(dict(source=str(source),**result))
                    atomic_json(a.base/'mp4_conversion_results.json',dict(completed=len(records),target=len(jobs),shards=records,updated=time.time()))
                    print(json.dumps(dict(source=str(source),bytes_before=result['original_bytes'],bytes_after=result['bytes'],seconds=result['conversion_seconds'])),flush=True)
                workers=[]
                for path in progress.glob('*.json'):
                    row=json.loads(path.read_text())
                    if row.get('source') in {str(p) for p in pending.values()}:workers.append(row)
                atomic_json(status,dict(stage='converting',completed=len(records),target=len(jobs),workers=workers,updated=time.time()))
        except BaseException:
            for task in pending:task.cancel()
            raise
    atomic_json(status,dict(stage='complete',completed=len(records),target=len(jobs),updated=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException:
        error=traceback.format_exc();print(error,flush=True)
        # Preserve both the old shard and any verified pending transaction for recovery.
        base=Path(data_path('tinywam'));atomic_json(base/'mp4_conversion_status.json',dict(stage='failed',error=error,updated=time.time()))
        raise
