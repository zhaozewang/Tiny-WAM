"""Explicitly prompted, jointly tracked hand/object masks using official SAM 2."""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def assign_labels(logits, identities):
    """A positive maximum logit owns each pixel; zero means background."""
    logits = np.asarray(logits)
    if logits.ndim != 3 or len(identities) != len(logits):
        raise ValueError('Expected one HxW logit image per object identity')
    winner = logits.argmax(axis=0)
    return np.where(logits.max(axis=0) > 0, np.asarray(identities)[winner], 0).astype(np.uint16)


def load_frames(source, native):
    config = json.loads((source / 'prepare.json').read_text())
    if not native:
        return np.load(source / 'source_frames.npy'), config
    frames = []
    with imageio.get_reader(source / 'source.mp4') as reader:
        for index in config['indices']:
            frames.append(np.asarray(Image.fromarray(reader.get_data(index)).crop(config['crop'])))
    return np.stack(frames), config


def track(source, output, prompt_path, sam_root=None, checkpoint=None, native=False,
          device='cuda', config_name='configs/sam2.1/sam2.1_hiera_l.yaml', output_size=None):
    import torch
    sam_root = sam_root or os.environ.get('SWAM_SAM_ROOT')
    if sam_root:
        sys.path.insert(0, str(Path(sam_root).resolve()))
    from sam2.build_sam import build_sam2_video_predictor
    checkpoint = checkpoint or (Path(sam_root) / 'sam2.1_hiera_large.pt' if sam_root else None)
    if checkpoint is None or not Path(checkpoint).is_file():
        raise ValueError('Provide a local SAM checkpoint; no automatic download')
    prompts = json.loads(prompt_path.read_text())
    objects = prompts['objects']
    ids = [o['id'] for o in objects]
    if not ids or len(set(ids)) != len(ids) or any(not 0 < i < 65536 for i in ids):
        raise ValueError('Prompt identities must be unique positive uint16 values')
    if any(o['role'] not in ('hand', 'object') for o in objects):
        raise ValueError('Each prompt role must be hand or object')
    frames, config = load_frames(source, native)
    width, height = config['size']
    if prompts['size'] != [width, height]:
        raise ValueError('Prompt coordinates must use prepared-frame dimensions')
    output_width, output_height = output_size or (width, height)
    if any(not isinstance(v, int) or v <= 0 for v in (output_width, output_height)):
        raise ValueError('Output dimensions must be positive integers')
    if any(not 0 <= a['frame'] < len(frames) for o in objects for a in o['annotations']):
        raise ValueError('Prompt frame outside source sequence')
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    predictor = build_sam2_video_predictor(config_name, str(checkpoint), device=device,
        hydra_overrides_extra=['++model.non_overlap_masks=true'])
    scale = np.array([frames.shape[2] / width, frames.shape[1] / height])
    labels = np.zeros((len(frames), output_height, output_width), dtype=np.uint16)
    covered = np.zeros(len(frames), dtype=bool)
    context = torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    with tempfile.TemporaryDirectory(prefix='sam-frames-', dir=output) as folder:
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(Path(folder) / f'{i:05d}.jpg', quality=95)
        with torch.inference_mode(), context:
            state = predictor.init_state(folder, offload_video_to_cpu=True, offload_state_to_cpu=True)
            for obj in objects:
                for annotation in obj['annotations']:
                    points = np.asarray(annotation['points'], dtype=np.float32)
                    point_labels = np.asarray(annotation['labels'], dtype=np.int32)
                    if points.shape != (len(point_labels), 2) or not np.isin(point_labels, [0, 1]).all():
                        raise ValueError('Invalid point/label arrays')
                    predictor.add_new_points_or_box(state, frame_idx=annotation['frame'], obj_id=obj['id'],
                        points=points*scale, labels=point_labels)
            first = min(a['frame'] for o in objects for a in o['annotations'])
            for reverse in ([False, True] if first > 0 else [False]):
                for index, identities, logits in predictor.propagate_in_video(state, start_frame_idx=first, reverse=reverse):
                    scores = logits[:, 0].float().cpu().numpy()
                    # Downsample logits, not integer labels or binary masks.
                    scores = np.stack([np.asarray(Image.fromarray(s).resize((output_width, output_height), Image.Resampling.BILINEAR)) for s in scores])
                    labels[index] = assign_labels(scores, identities)
                    covered[index] = True
    if not covered.all():
        raise RuntimeError('SAM did not produce every source frame')
    object_ids = [o['id'] for o in objects if o['role'] == 'object']
    hand_ids = [o['id'] for o in objects if o['role'] == 'hand']
    np.save(output / 'labels.npy', labels)
    np.save(output / 'object_masks.npy', np.isin(labels, object_ids))
    np.save(output / 'human_masks.npy', np.isin(labels, hand_ids))
    (output / 'prompts.json').write_text(json.dumps(prompts, indent=2) + '\n')
    metadata = dict(source=str(source.resolve()), source_indices=config['indices'],
                    source_prepare_sha256=hashlib.sha256((source / 'prepare.json').read_bytes()).hexdigest(),
                    model_config=config_name, checkpoint=str(Path(checkpoint).resolve()),
                    input_size=[frames.shape[2], frames.shape[1]], output_size=[output_width, output_height],
                    native_input=native, non_overlap=True, seconds=time.perf_counter()-start,
                    objects=objects, semantics='visible-instance labels; no metric depth',
                    areas={str(i): (labels == i).sum(axis=(1, 2)).tolist() for i in ids})
    (output / 'segmentation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: metadata[k] for k in ('seconds', 'input_size', 'output_size')}), flush=True)


def segment_hands(source, sam_root=None):
    """Compatibility for fresh batch preparation; outputs are never overwritten."""
    config = json.loads((source / 'prepare.json').read_text())
    points = json.loads((source / 'prompts.json').read_text())
    width, height = config['size']
    objects = []
    for identity, (side, values) in enumerate(points.items(), 1):
        values = [p for p in values if 0 <= p[0] < width and 0 <= p[1] < height]
        if values:
            objects.append(dict(id=identity, role='hand', name=side,
                                annotations=[dict(frame=0, points=values, labels=[1]*len(values))]))
    prompt = source / 'hand_tracking_prompts.json'
    if prompt.exists() or (source / 'human_masks.npy').exists():
        raise FileExistsError('Hand segmentation already exists; preserve previous results')
    prompt.write_text(json.dumps(dict(size=[width, height], objects=objects), indent=2))
    destination = source / 'hand_segmentation'
    track(source, destination, prompt, sam_root)
    import shutil
    shutil.copy2(destination / 'human_masks.npy', source / 'human_masks.npy')
