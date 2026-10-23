"""Supervise bounded, restartable continuous-trajectory collection on the HDD."""
from tinywam.paths import data_path
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from tinywam.shards import atomic_json, publish_episode, sha256


def stop_worker(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(data_path('tinywam/r1_random_10min_512_solver64_v1')))
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--parallel', type=int, default=10)
    parser.add_argument('--assets', type=Path, default=Path(__file__).resolve().parents[1] / 'results/original_random_5s_512')
    parser.add_argument('--storage-root', type=Path, help='Require this mounted filesystem for dataset storage')
    args = parser.parse_args()
    assert 1 <= args.parallel <= 16 and args.episodes > 0
    assert args.seconds % 5 == 0
    args.root.mkdir(parents=True, exist_ok=True)
    if args.storage_root is not None:
        assert args.storage_root.is_mount(), 'Requested storage filesystem is not mounted'
        assert args.root.stat().st_dev == args.storage_root.stat().st_dev
    for name in ('shards','staging','failed','logs','provenance'):
        (args.root/name).mkdir(exist_ok=True)
    lock = (args.root/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = dict(episodes=args.episodes,duration_s=args.seconds,fps=24,views=3,resolution=[512,512],
        block_actions=120,parallel=args.parallel,solver_iterations=[64,8],hand_damping=.1,visual_partitions=True,protocol="original_actions_solver64_v1",split='episode_id modulo 14: 0..11 train, 12 validation, 13 test',
        encoding='lossless RGB uint8 NPZ blocks inside uncompressed TAR; no VAE dependency',
        restart='resume completed shards; restart interrupted trajectories from beginning, never splice resets',
        disk_free_floor_bytes=500*1024**3,policy='original_smoothstep_1s; arm +/-0.15rad; hand inner80%; no holds; seed100+id',
        reset='one reference pose reset and 2s physics warm-up per trajectory, no internal resets')
    config_path=args.root/'dataset.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Existing dataset protocol differs'
    else:
        atomic_json(config_path,config)
    source=Path(__file__).resolve().parents[1]
    snapshots={name:source/'src/tinywam'/name for name in
        ('collect_continuous512.py','parallel_scene.py','collection_contract.py','shards.py','r1.py','workcell.py','collect.py','preview.py')}
    snapshots.update({'generate_continuous512.py':Path(__file__).resolve(),
        'review_long_batch.py':source/'scripts/review_long_batch.py',
        'arm_pose.json':source/'assets/arm_pose.json',
        'r1_isyhand.urdf':args.assets/'r1_isyhand.urdf'})
    hashes={}
    for name,path in snapshots.items():
        destination=args.root/'provenance'/name
        digest=sha256(path)
        if destination.exists() and sha256(destination)!=digest:
            atomic_json(args.root/'status.json',dict(stage='failed',reason='Source provenance changed',file=name))
            raise SystemExit(78)
        if not destination.exists():
            shutil.copy2(path,destination)
        hashes[name]=digest
    atomic_json(args.root/'provenance/sha256.json',hashes)
    attempts_path=args.root/'attempts.json'
    attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else {}
    started=time.time()
    while True:
        # Finish committed trajectories left behind by an interrupted supervisor.
        for folder in sorted((args.root/'staging').glob('episode_*')):
            if (folder/'complete.json').exists():
                try:
                    atomic_json(args.root/'status.json',dict(stage='publishing',episode=folder.name,updated=time.time()))
                    publish_episode(folder,args.root/'shards')
                except Exception as error:
                    atomic_json(args.root/'status.json',dict(stage='failed',reason='Shard publication rejected',
                        episode=folder.name,error=repr(error),updated=time.time()))
                    raise SystemExit(78) from error
        manifests=[json.loads(p.read_text()) for p in sorted((args.root/'shards').glob('episode_*.json'))]
        committed={item['episode'] for item in manifests}
        atomic_json(args.root/'index.json',dict(shards=manifests))
        pending=[i for i in range(args.episodes) if i not in committed]
        state=dict(stage='complete' if not pending else 'collecting',completed=len(committed),target=args.episodes,
            bytes=sum(m['bytes'] for m in manifests),updated=time.time(),started=started)
        atomic_json(args.root/'status.json',state)
        if not pending:
            print('DATASET_COMPLETE',flush=True)
            return
        if shutil.disk_usage(args.root).free < config['disk_free_floor_bytes'] + args.parallel*35*1024**3:
            atomic_json(args.root/'status.json',dict(state,stage='paused_disk_reserve'))
            time.sleep(30)
            continue
        ids=pending[:args.parallel]
        for identifier in ids:
            if attempts.get(str(identifier),0) >= 3:
                atomic_json(args.root/'status.json',dict(state,stage='failed',reason='Three failed attempts',episode=identifier,worker=json.loads((args.root/'worker_status.json').read_text())))
                raise SystemExit(78)
            folder=args.root/'staging'/f'episode_{identifier:05d}'
            if folder.exists():
                folder.rename(args.root/'failed'/f'{folder.name}_{time.time_ns()}')
            attempts[str(identifier)]=attempts.get(str(identifier),0)+1
        atomic_json(attempts_path,attempts)
        batch_name=f'batch_{ids[0]:05d}_{time.time_ns()}'
        command=[sys.executable,'-u','-m','tinywam.collect_continuous512','--root',str(args.root),
                 '--assets',str(args.assets),'--ids',*map(str,ids),'--seconds',str(args.seconds),'--device','cuda:0']
        log=args.root/'logs'/f'{batch_name}.log'
        with log.open('w') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            launched=time.time()
            failure=None
            try:
                while process.poll() is None:
                    done=all((args.root/'staging'/f'episode_{i:05d}'/'complete.json').exists() for i in ids)
                    if done:
                        break  # Isaac shutdown can hang; data is already closed and fsynced.
                    progress_path=args.root/'worker_status.json'
                    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
                    if progress.get('stage') == 'failed' and progress.get('updated',0) >= launched:
                        failure=progress
                        print('COLLECTOR_FAILED',progress.get('error'),flush=True)
                        break
                    stamp=max(launched,progress.get('updated',0))
                    limit=1200 if progress.get('stage')=='initializing' else 300
                    if time.time()-stamp > limit:
                        raise TimeoutError('Collector heartbeat stalled')
                    atomic_json(args.root/'status.json',dict(state,active_ids=ids,worker=progress,
                        log=str(log),updated=time.time()))
                    time.sleep(5)
            except TimeoutError as error:
                failure=dict(stage='failed',error=str(error),updated=time.time())
                print(str(error),flush=True)
            finally:
                stop_worker(process)
        print('BATCH_STOPPED',ids,'exit',process.returncode,flush=True)
        latest=json.loads((args.root/'worker_status.json').read_text())
        if latest.get('stage')=='failed' and latest.get('updated',0)>=launched:
            failure=latest
        if not failure and not all((args.root/'staging'/f'episode_{i:05d}'/'complete.json').exists() for i in ids):
            failure=dict(stage='failed',error=f'Worker exited {process.returncode} without a complete batch',updated=time.time())
        if failure:
            atomic_json(args.root/'status.json',dict(state,stage='failed',active_ids=ids,worker=failure,
                reason='Collector rejected data; identical seeds will not be blindly retried',updated=time.time()))
            raise SystemExit(78)


if __name__ == '__main__':
    main()
