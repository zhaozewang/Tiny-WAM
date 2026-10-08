"""Maintained geometry entrypoint; no training or generative video model launch."""
import argparse
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare-paired', 'segment-egodex', 'align-sharpa',
                                   'align-arms', 'batch-egodex', 'segment-objects', 'composite'])
    p.add_argument('--output', type=Path, required=True)
    for name in ('source', 'reference', 'initial', 'archive', 'prompts', 'segmentation', 'sam-root', 'checkpoint'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--frames', default='all')
    p.add_argument('--indices', default='20:86:2')
    p.add_argument('--crop', default='640,360,1920,1080')
    p.add_argument('--member')
    p.add_argument('--hand-scale', type=float, default=1.)
    p.add_argument('--wrist-ratio', type=float, default=1.)
    p.add_argument('--wrist-spacer', type=float, default=.012)
    for name in ('silhouette', 'separate-scales', 'screen-fit', 'offload-state', 'native-input', 'plan-only'):
        p.add_argument('--' + name, action='store_true')
    p.add_argument('--count', type=int, default=200)
    p.add_argument('--max-clips', type=int)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--fit-python', type=Path, default=Path(sys.executable))
    p.add_argument('--segment-python', type=Path, default=Path(sys.executable))
    a = p.parse_args()
    if a.mode == 'segment-objects':
        if not (a.source and a.prompts):
            p.error('segment-objects requires --source and --prompts')
        from .segmentation import track
        return track(a.source, a.output, a.prompts, a.sam_root, a.checkpoint, a.native_input)
    if a.mode == 'composite':
        if not (a.source and a.reference and a.segmentation):
            p.error('composite requires --source, --reference and --segmentation')
        from .occlusion import compare
        return compare(a.source, a.reference, a.segmentation, a.output)
    if a.mode == 'segment-egodex':
        from .segmentation import segment_hands
        return segment_hands(a.output, a.sam_root)
    if a.mode == 'batch-egodex':
        from .batch import run_batch
        if a.archive is None or a.count < 1 or a.workers < 1:
            p.error('batch-egodex requires --archive and positive --count/--workers')
        return run_batch(a.archive, a.output, a.count, a.fit_python, a.segment_python,
                         a.plan_only, a.max_clips, a.workers)
    from . import geometry
    if a.mode == 'prepare-paired':
        indices = list(range(*map(int, a.indices.split(':'))))
        if len(indices) < 2 or not a.member:
            p.error('prepare-paired requires two frames and --member')
        a.output.mkdir(parents=True, exist_ok=True)
        return geometry.prepare_paired(a.output, indices, tuple(map(int, a.crop.split(','))), a.member)
    if a.source is None:
        p.error('Alignment requires --source')
    if a.output.exists():
        p.error('Alignment output already exists; use a new directory')
    if a.mode == 'align-sharpa':
        import numpy as np
        selected = range(len(np.load(a.source / 'source_frames.npy'))) if a.frames == 'all' else list(map(int, a.frames.split(',')))
        return geometry.prepare_sharpa(a.source, a.output, selected, a.hand_scale, a.silhouette)
    if a.reference is None or (a.screen_fit and not a.separate_scales):
        p.error('align-arms requires --reference; --screen-fit requires --separate-scales')
    from .connected import run
    hands = {s: geometry.SharpaHand(s, geometry.pyrender.Scene()) for s in ('left', 'right')}
    return run(a.source, a.reference, a.output, hands, a.separate_scales, a.initial,
               a.wrist_ratio, a.wrist_spacer, a.screen_fit)
