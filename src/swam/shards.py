"""Verified HDD shards, leased SSD caching, and frame-aligned window reads."""
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile

import numpy as np


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def publish_episode(episode, shard_root):
    episode, shard_root = Path(episode), Path(shard_root)
    metadata = json.loads((episode / 'complete.json').read_text())
    protocol = json.loads((episode / 'metadata.json').read_text())
    assert not protocol.get('physics_only',False), 'Physics diagnostic is not an image dataset'
    blocks = sorted(episode.glob('block_*.npz'))
    assert len(blocks) == metadata['blocks']
    assert metadata['actions'] == metadata['fps'] * metadata['duration_s']
    # Verify each compressed member and all cross-block alignment before publishing.
    previous = None
    actions = 0
    for block in blocks:
        with np.load(block) as source:
            # Decode once; all members are read and ZIP CRCs verified.
            data = {key: source[key] for key in source.files}
            assert len(data['qpos']) == len(data['actions']) + 1
            assert len(data['rgb']) == len(data['qpos'])
            if previous is not None:
                for key, value in previous.items():
                    assert np.array_equal(value, data[key][0]), f'Discontinuous {key}'
                assert data['frame_indices'][0] == actions
            assert np.array_equal(data['frame_indices'], np.arange(actions, actions + len(data['qpos'])))
            previous = {key: data[key][-1].copy() for key in ('qpos', 'qvel', 'rgb')}
            actions += len(data['actions'])
    assert actions == metadata['actions']
    shard_root.mkdir(parents=True, exist_ok=True)
    target = shard_root / (episode.name + '.tar')
    partial = target.with_suffix('.tar.partial')
    with tarfile.open(partial, 'w') as archive:
        for source in sorted(episode.iterdir()):
            if source.is_file() and source.suffix != '.tmp':
                archive.add(source, arcname=source.name, recursive=False)
    with partial.open('rb') as stream:
        os.fsync(stream.fileno())
    checksum = sha256(partial)
    with tarfile.open(partial) as archive:
        assert len([m for m in archive if m.name.startswith('block_')]) == metadata['blocks']
    partial.replace(target)
    manifest = dict(metadata, file=target.name, bytes=target.stat().st_size, sha256=checksum)
    atomic_json(target.with_suffix('.json'), manifest)
    # Only remove the redundant staging copy after the final HDD shard is committed.
    shutil.rmtree(episode)
    return manifest


@contextmanager
def cached_shard(source, cache_root):
    source = Path(source).resolve()
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    if source.parent == cache_root or source.is_relative_to(cache_root):
        raise ValueError('Cache must be separate from the HDD source')
    marker = cache_root / '.swam_cache.json'
    if not marker.exists():
        if any(cache_root.iterdir()):
            raise ValueError('Use an empty dedicated SWAM cache directory')
        atomic_json(marker, dict(owner='swam-shard-cache', schema=1))
    assert json.loads(marker.read_text())['owner'] == 'swam-shard-cache'
    if cache_root.stat().st_dev == source.stat().st_dev:
        raise ValueError('SSD cache and HDD source must be on different filesystems')
    manifest = json.loads(source.with_suffix('.json').read_text())
    target = cache_root / (manifest['sha256'][:16] + '_' + source.name)
    with (cache_root / (target.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists() and sha256(target) != manifest['sha256']:
            target.unlink()  # This is exclusively our leased SSD cache copy.
        if not target.exists():
            if shutil.disk_usage(cache_root).free < manifest['bytes'] + 10 * 1024**3:
                raise RuntimeError('Insufficient SSD cache space')
            temporary = target.with_suffix('.tar.partial')
            shutil.copyfile(source, temporary)
            if sha256(temporary) != manifest['sha256']:
                temporary.unlink()
                raise IOError('SSD copy checksum mismatch')
            temporary.replace(target)
        try:
            yield target
        finally:
            target.unlink(missing_ok=True)


def _read_mp4_rgb(archive, start, frames):
    """Read an exact frame window from a lossless RGB MP4 shard."""
    import subprocess
    import tempfile
    storage=json.load(archive.extractfile('storage.json'))
    height,width=storage['resolution'];fps=storage['fps']
    whole_seconds,remainder=divmod(start,fps)
    images=[]
    with tempfile.TemporaryDirectory(prefix='swam-mp4-') as directory:
        for view in storage['views']:
            path=Path(directory)/(view+'.mp4')
            with path.open('wb') as output:
                shutil.copyfileobj(archive.extractfile(view+'.mp4'),output)
            raw=subprocess.check_output(['ffmpeg','-v','error','-threads','2',
                '-ss',str(whole_seconds),'-i',str(path),'-vf',
                f'trim=start_frame={remainder}:end_frame={remainder+frames}',
                '-frames:v',str(frames),'-vsync','0','-f','rawvideo',
                '-pix_fmt','rgb24','-threads','2','pipe:1'])
            assert len(raw)==frames*height*width*3, 'MP4 frame count mismatch'
            images.append(np.frombuffer(raw,dtype=np.uint8).reshape(frames,height,width,3))
    return np.stack(images,axis=1)


def read_window(shard, start, frames):
    """Return frames observations and frames-1 causal actions, possibly across blocks."""
    with tarfile.open(shard) as archive:
        meta = json.load(archive.extractfile('complete.json'))
        if not (0 <= start and 2 <= frames and start + frames <= meta['actions'] + 1):
            raise ValueError('Requested window is outside this continuous trajectory')
        first = start // meta['block_actions']
        last = (start + frames - 2) // meta['block_actions']
        video_storage='storage.json' in archive.getnames()
        arrays = {}
        for index in range(first, last + 1):
            member = archive.extractfile(f'block_{index:05d}.npz')
            with np.load(io.BytesIO(member.read())) as block:
                for key in ('rgb', 'qpos', 'qvel', 'actions', 'frame_indices'):
                    if key=='rgb' and video_storage:continue
                    value = block[key]
                    if index > first and key != 'actions':
                        value = value[1:]
                    arrays.setdefault(key, []).append(value)
        offset = start - first * meta['block_actions']
        result = {}
        for key, parts in arrays.items():
            joined = np.concatenate(parts)
            count = frames - 1 if key == 'actions' else frames
            result[key] = joined[offset:offset + count]
        assert np.array_equal(result['frame_indices'], np.arange(start, start + frames))
        if video_storage:result['rgb']=_read_mp4_rgb(archive,start,frames)
        result['timestamps'] = result['frame_indices'] / meta['fps']
        return result
