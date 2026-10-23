"""Resume a fixed manifest of complete EgoDex geometry fits."""
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import zipfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE_FILES = ['scripts/retarget_egodex.py', 'src/tinywam/retargeting/cli.py',
                'src/tinywam/retargeting/geometry.py', 'src/tinywam/retargeting/batch.py',
                'src/tinywam/retargeting/connected.py', 'src/tinywam/r1.py',
                'src/tinywam/retargeting/segmentation.py']


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def annotation_info(blob):
    """Check required annotations without filtering on fit quality."""
    with h5py.File(io.BytesIO(blob)) as data:
        frames = len(data['transforms/camera'])
        if frames < 5 or data['camera/intrinsic'].shape != (3, 3):
            raise ValueError('Insufficient frames or unsupported camera intrinsics')
        coverage = {}
        for side in ('left', 'right'):
            labels = [side + 'Hand']
            for finger in ('IndexFinger', 'MiddleFinger', 'RingFinger', 'LittleFinger', 'Thumb'):
                labels.extend(side + finger + part for part in
                              ('Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip'))
            for label in ['camera', side + 'Arm', side + 'Forearm', *labels]:
                values = data['transforms/' + label]
                if values.shape != (frames, 4, 4) or not np.isfinite(values[:]).all():
                    raise ValueError('Invalid transform: ' + label)
            confidence = np.stack([data['confidences/' + label][:] for label in labels], axis=1)
            if confidence.shape != (frames, 21) or not np.isfinite(confidence).all():
                raise ValueError('Invalid hand confidence: ' + side)
            coverage[side] = float((confidence > 0).mean())
        return dict(source_frames=frames, positive_confidence_fraction=coverage)


def plan_batch(archive_path, output, count):
    """Freeze a category-balanced selection before any solver runs."""
    if (output / 'manifest.json').exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['archive'] != str(archive_path.resolve()) or manifest['count'] != count:
            raise ValueError('Existing manifest belongs to a different collection')
        return manifest

    # Shuffle within categories, then traverse every category before its next clip.
    rng = random.Random(42)
    with zipfile.ZipFile(archive_path) as archive:
        groups = {}
        members = set(archive.namelist())
        for name in sorted(members):
            if name.endswith('.mp4') and not name.startswith('__MACOSX/'):
                stem = name[:-4]
                groups.setdefault(str(Path(stem).parent), []).append(stem)
        categories = sorted(groups)
        rng.shuffle(categories)
        for clips in groups.values():
            rng.shuffle(clips)
        candidates = ['test/screw_unscrew_bottle_cap/0']
        for index in range(max(map(len, groups.values()))):
            candidates.extend(groups[key][index] for key in categories if index < len(groups[key]))

        # Keep every exclusion and every accepted identity; failed fits are not replaced.
        selected, excluded, seen = [], [], set()
        for stem in candidates:
            if stem in seen:
                continue
            seen.add(stem)
            try:
                info = annotation_info(archive.read(stem + '.hdf5'))
                video = archive.getinfo(stem + '.mp4')
                annotation = archive.getinfo(stem + '.hdf5')
            except (KeyError, ValueError, OSError) as error:
                excluded.append(dict(member=stem, reason=str(error)))
                continue
            info.update(member=stem, video_crc=video.CRC, annotation_crc=annotation.CRC)
            selected.append(info)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)} clips have the required schema')

    # Freeze executable source beside the collection for reproducible resumes.
    hashes = {}
    for name in SOURCE_FILES:
        destination = output / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        hashes[name] = digest(destination)
    manifest = dict(archive=str(archive_path.resolve()), count=count, seed=42,
                    selection='Reviewed bottle-cap clip, then shuffled category round-robin; schema-only exclusions',
                    stride=2, size=[512, 288], whole_sequence=True, internal_resets=False,
                    hand_reference_scale=0.7455261749407607, connected_hand_scale=.73,
                    wrist_ratio=.4, wrist_spacer=0, iterations=650, source_hashes=hashes,
                    clips=selected, excluded=excluded, created=time.time())
    save_json(output / 'manifest.json', manifest)
    return manifest


def probe_video(path):
    command = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate,duration',
               '-of', 'json', str(path)]
    result = subprocess.check_output(command)
    return json.loads(result)['streams'][0]


def extract_clip(archive_path, item, source):
    source.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for suffix in ('mp4', 'hdf5'):
            name = item['member'] + '.' + suffix
            expected = item['video_crc' if suffix == 'mp4' else 'annotation_crc']
            if archive.getinfo(name).CRC != expected:
                raise ValueError('Archive member changed: ' + name)
            with archive.open(name) as reader, (source / ('source.' + suffix)).open('wb') as writer:
                shutil.copyfileobj(reader, writer)
    video = probe_video(source / 'source.mp4')
    if int(video['nb_read_frames']) != item['source_frames']:
        raise ValueError('Video and annotation frame counts differ')
    return video


def check_sources(hashes):
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Source changed since planning: ' + name)


def run_stage(case, record, name, command, environment, hashes):
    if record['stages'].get(name, {}).get('state') == 'complete':
        return
    entry = dict(state='running', command=command, started=time.time())
    record['stages'][name] = entry
    save_json(case / 'status.json', record)
    # Serialize SAM state allocation; independent CPU fits can proceed together.
    lock = (ROOT / 'results/retargeting_sam.lock').open('w') if name == 'segment' else nullcontext()
    with lock as handle:
        if handle is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        check_sources(hashes)
        with (case / (name + '.log')).open('a') as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    entry.update(finished=time.time(), returncode=result.returncode)
    entry['state'] = 'complete' if result.returncode == 0 else 'failed'
    save_json(case / 'status.json', record)
    if result.returncode:
        raise RuntimeError(f'{name} exited with code {result.returncode}')


def verify_clip(case, item):
    """Verify full coverage and saved numeric state; processing success is not fit quality."""
    config = json.loads((case / 'source/prepare.json').read_text())
    expected = list(range(0, item['source_frames'], 2))
    if config['indices'] != expected:
        raise ValueError('Prepared video does not cover the full source at stride two')
    parameters = np.load(case / 'connected/parameters.npz')
    for side in ('left', 'right'):
        if parameters[side].shape != (len(expected), 28):
            raise ValueError('Incomplete joint trajectory: ' + side)
    if not all(np.isfinite(parameters[key]).all() for key in parameters.files):
        raise ValueError('Nonfinite fitted parameters')
    metrics = json.loads((case / 'connected/metrics.json').read_text())
    if not all(row['limits_valid'] for row in metrics['hands'].values()):
        raise ValueError('Joint-limit violation')
    for name in ('source/original.mp4', 'hand/alignment.mp4', 'connected/comparison.mp4'):
        info = probe_video(case / name)
        if int(info['nb_read_frames']) != len(expected):
            raise ValueError('Incomplete rendered video: ' + name)
    optimization = json.loads((case / 'connected/optimization.json').read_text())
    review = dict(frames=len(expected), duration_seconds=len(expected) / config['fps'],
                  complete_source_coverage=True, finite_parameters=True, joint_limits_valid=True,
                  optimizer_converged=optimization['success'], positive_confidence_fraction=item['positive_confidence_fraction'],
                  geometry_quality='Needs visual review; numerical validity does not establish alignment or contact',
                  metrics=metrics)
    save_json(case / 'quality.json', review)
    return review


def run_clip(archive_path, output, item, fit_python, segment_python, hashes):
    case = output / 'clips' / item['member']
    case.mkdir(parents=True, exist_ok=True)
    status = case / 'status.json'
    record = json.loads(status.read_text()) if status.exists() else dict(member=item['member'], stages={})
    if record.get('state') in ('processed', 'failed'):
        return record
    record.update(state='running', started=record.get('started', time.time()))
    save_json(status, record)
    environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    environment['PYOPENGL_PLATFORM'] = 'egl'
    source, hand, connected = case / 'source', case / 'hand', case / 'connected'
    command = [str(fit_python), str(ROOT / 'scripts/retarget_egodex.py')]

    try:
        if 'video' not in record:
            record['video'] = extract_clip(archive_path, item, source)
            save_json(status, record)
        video = record['video']
        crop = f"0,0,{video['width']},{video['height']}"
        run_stage(case, record, 'prepare', command + ['prepare-paired', '--output', str(source),
                  '--indices', f"0:{item['source_frames']}:2", '--crop', crop, '--member', item['member']], environment, hashes)
        segment = [str(segment_python), *command[1:], 'segment-egodex', '--output', str(source), '--offload-state']
        run_stage(case, record, 'segment', segment, environment, hashes)
        run_stage(case, record, 'hand', command + ['align-sharpa', '--source', str(source), '--output', str(hand),
                  '--hand-scale', '0.7455261749407607', '--silhouette'], environment, hashes)
        run_stage(case, record, 'connected', command + ['align-arms', '--source', str(source), '--reference', str(hand),
                  '--output', str(connected), '--separate-scales', '--wrist-ratio', '.4', '--wrist-spacer', '0', '--screen-fit'], environment, hashes)
        record['quality'] = verify_clip(case, item)
        record.update(state='processed', finished=time.time())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        record.update(state='failed', error=str(error), finished=time.time())
    save_json(status, record)
    return record


def run_batch(archive, output, count, fit_python, segment_python, plan_only=False, max_clips=None, workers=1):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'batch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = plan_batch(archive, output, count)
        check_sources(manifest['source_hashes'])
        if plan_only:
            print(f"Selected {count} complete videos; {len(manifest['excluded'])} schema exclusions", flush=True)
            return
        selected = manifest['clips'] if max_clips is None else manifest['clips'][:max_clips]
        records = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for item in selected:
                future = pool.submit(run_clip, archive, output, item, fit_python, segment_python, manifest['source_hashes'])
                futures.append(future)
            for future in as_completed(futures):
                record = future.result()
                records[record['member']] = record
                processed = sum(row['state'] == 'processed' for row in records.values())
                failed = sum(row['state'] == 'failed' for row in records.values())
                progress = dict(total=count, visited=len(records), processed=processed, failed=failed,
                                workers=workers, last_member=record['member'], last_state=record['state'], updated=time.time())
                save_json(output / 'progress.json', progress)
                print(record['state'], record['member'], record.get('error', ''), flush=True)
