"""Track the same first-frame labels with official Cutie, without fitting weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--seed', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cutie-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.cutie_root.resolve()))
    from hydra import initialize_config_dir, compose
    from omegaconf import open_dict
    from cutie.model.cutie import CUTIE
    from cutie.inference.inference_core import InferenceCore
    from cutie.inference.utils.args_utils import get_dataset_cfg

    digest = hashlib.md5(a.checkpoint.read_bytes()).hexdigest()
    if digest != 'a6071de6136982e396851903ab4c083a':
        raise ValueError('Checkpoint does not match official Cutie release checksum')
    frames = np.load(a.source / 'source_frames.npy')
    initial = np.load(a.seed / 'labels.npy')[0].astype(np.int64)
    metadata = json.loads((a.seed / 'segmentation.json').read_text())
    config = json.loads((a.source / 'prepare.json').read_text())
    if metadata['source_indices'] != config['indices']:
        raise ValueError('Seed labels and source frames do not match')
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with initialize_config_dir(version_base='1.3.2', config_dir=str((a.cutie_root / 'cutie/config').resolve())):
        cfg = compose(config_name='eval_config')
    get_dataset_cfg(cfg)
    model = CUTIE(cfg).cuda().eval()
    model.load_weights(torch.load(a.checkpoint, map_location='cpu', weights_only=True))
    processor = InferenceCore(model, cfg=cfg)
    processor.max_internal_size = 480
    identities = [int(i) for i in np.unique(initial) if i]
    labels = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for i, frame in enumerate(frames):
            image = torch.from_numpy(frame.copy()).permute(2, 0, 1).cuda().float()/255
            if i == 0:
                probs = processor.step(image, torch.from_numpy(initial).cuda(), objects=identities)
            else:
                probs = processor.step(image)
            labels.append(processor.output_prob_to_mask(probs).cpu().numpy().astype(np.uint16))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    labels = np.stack(labels)
    object_ids = [o['id'] for o in metadata['objects'] if o['role'] == 'object']
    hand_ids = [o['id'] for o in metadata['objects'] if o['role'] == 'hand']
    np.save(a.output / 'labels.npy', labels)
    np.save(a.output / 'object_masks.npy', np.isin(labels, object_ids))
    np.save(a.output / 'human_masks.npy', np.isin(labels, hand_ids))
    metadata.update(model='Cutie base mega', checkpoint=str(a.checkpoint.resolve()),
                    seed_labels=str((a.seed / 'labels.npy').resolve()), seconds=seconds,
                    input_size=[frames.shape[2], frames.shape[1]], native_input=False,
                    areas={str(i): (labels == i).sum(axis=(1, 2)).tolist() for i in identities})
    metadata.pop('model_config', None)
    (a.output / 'segmentation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(seconds=seconds, frames=len(frames))), flush=True)


if __name__ == '__main__':
    main()
