"""Sample windows from atomically published latent blocks of verified shards."""
import json
import random
from functools import lru_cache
from pathlib import Path

import torch


class GrowingWindows:
    def __init__(self, root, seed=42, frames=241):
        self.root = Path(root)
        self.rng = random.Random(seed)
        self.frames = frames
        self.norm = torch.load(self.root / 'normalization.pt', map_location='cpu', weights_only=False)
        self.refresh()

    def refresh(self):
        index = json.loads((self.root / 'index.json').read_text())
        if index['tokenizer_sha256'] != self.norm['tokenizer_sha256']:
            raise ValueError('Published data and normalization use different tokenizers')
        self.episodes = [row for row in index['episodes'].values()
                         if row['blocks'] * 120 >= self.frames - 1]

    @lru_cache(maxsize=48)
    def block(self, episode, index):
        path = self.root / f'episode_{episode:05d}' / f'block_{index:05d}.pt'
        return torch.load(path, map_location='cpu', weights_only=False)

    def read(self, episode, start, frames=241):
        # Adjacent blocks share an observation but never share an action.
        first = start // 120
        last = (start + frames - 2) // 120
        parts = {}
        for index in range(first, last + 1):
            block = self.block(episode, index)
            for name in ('images', 'proprio', 'actions', 'qpos', 'frames'):
                value = block[name]
                if index != first and name != 'actions':
                    value = value[1:]
                parts.setdefault(name, []).append(value)

        # Crop after stitching, preserving the final observation/action offset.
        offset = start - first * 120
        window = {}
        for name, values in parts.items():
            joined = torch.cat(values)
            length = frames - 1 if name == 'actions' else frames
            window[name] = joined[offset:offset + length]
        if not torch.equal(window['frames'], torch.arange(start, start + frames)):
            raise ValueError('Latent blocks contain non-contiguous frame indices')
        if len(window['actions']) != frames - 1:
            raise ValueError('Window must have one more observation than command')
        return self.normalize(window)

    def normalize(self, window):
        # State/action codes are normalized when prepared; images and goals here.
        images = window['images'].float()
        window['images'] = (images - self.norm['image_mean']) / self.norm['image_scale']
        window['proprio'] = window['proprio'].float()
        window['actions'] = window['actions'].float()
        joints = window.pop('qpos')
        window['joint_positions'] = (joints - self.norm['goal_mean']) / self.norm['goal_scale']
        frame_indices = window.pop('frames').float()
        window['times'] = frame_indices / 24
        return window

    def batch(self, n, split='train', fixed=False):
        eligible = [row for row in self.episodes if row['split'] == split
                    and (split == 'train' or row['episode'] == 12)]
        if not eligible:
            raise RuntimeError(f'No complete {split} windows')

        # Preserve the choice/start RNG order and the historical validation episode.
        windows = []
        metadata = []
        for _ in range(n):
            row = eligible[0] if fixed else self.rng.choice(eligible)
            count = row['blocks'] * 120 - (self.frames - 1) + 1
            start = 0 if fixed else self.rng.randrange(count)
            windows.append(self.read(row['episode'], start, self.frames))
            metadata.append((row['episode'], start))
        batch = {}
        for name in windows[0]:
            values = [window[name] for window in windows]
            batch[name] = torch.stack(values)
        return batch, metadata
