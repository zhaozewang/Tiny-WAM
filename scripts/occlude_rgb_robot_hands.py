"""Transfer observed object visibility to the frozen, approved RGB robot render."""
from swam.paths import external_path
import argparse
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from swam.retargeting.occlusion import composite
from swam.retargeting.segmentation import load_frames, track


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panels(images, titles):
    width, height = images[0].shape[1], images[0].shape[0]
    canvas = Image.new('RGB', (width*len(images), height+28), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (im, title) in enumerate(zip(images, titles)):
        canvas.paste(Image.fromarray(im), (width*i, 28))
        draw.text((width*i+8, 7), title, fill='black')
    return canvas


def run(args):
    source = ROOT/'results/retargeting_import_20261007'/args.clip/'source'
    frozen = ROOT/'results/rgb_robot_hands_v1'/args.clip
    frames, config = load_frames(source, True)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    if args.inspect:
        for i in (0, 8, 16, 24, 32):
            Image.fromarray(frames[i]).save(output/f'source_{i:03d}.png')
        return
    render_path = frozen/'robot_render.npz'
    pose_path = frozen/'qpos.npz'
    hashes = {p.name: sha256(p) for p in (render_path, pose_path)}
    with np.load(render_path) as data:
        rgb, masks = data['rgb'], data['mask'].astype(bool)
    with np.load(pose_path) as data:
        if not np.array_equal(data['source_indices'], config['indices']):
            raise ValueError('Frozen robot render does not match source frame indices')
    height, width = masks.shape[1:]
    frames = np.stack([np.asarray(Image.fromarray(f).resize((width, height), Image.Resampling.LANCZOS)) for f in frames])
    if args.segmentation:
        segmentation = args.segmentation
    else:
        segmentation = output/'segmentation'
        track(source, segmentation, args.prompts, native=True,
              sam_root=external_path('sam', ''), output_size=(width, height))
    metadata = json.loads((segmentation/'segmentation.json').read_text())
    if metadata['source_indices'] != config['indices']:
        raise ValueError('Object segmentation is not aligned with RGB poses')
    labels = np.load(segmentation/'labels.npy')
    objects = np.load(segmentation/'object_masks.npy')
    if frames.shape != rgb.shape or objects.shape != masks.shape:
        raise ValueError('Source, robot and masks must have identical resolution and frame counts')
    counts = []
    # Fixed crops are in render pixels and remain unchanged throughout each clip.
    crop = (610, 205, 960, 435) if args.clip == 'use_chopsticks' else (335, 190, 850, 535)
    colors = np.array([[0, 0, 0], [30, 220, 70], [60, 150, 255], [230, 60, 210],
                       [240, 150, 30], [170, 60, 255]], dtype=np.uint8)
    visible_masks = []
    with imageio.get_writer(output/'robot_hands.mp4', fps=config['fps'], macro_block_size=1, quality=9) as video, \
         imageio.get_writer(output/'comparison.mp4', fps=config['fps'], macro_block_size=1, quality=9) as comparison, \
         imageio.get_writer(output/'detail.mp4', fps=config['fps'], macro_block_size=1, quality=9) as detail:
        for i, frame in enumerate(frames):
            before = np.where(masks[i, ..., None], rgb[i], frame)
            after, visible = composite(frame, rgb[i], masks[i], objects[i])
            changed = masks[i] & objects[i]
            assert np.array_equal(after[~changed], before[~changed])
            assert np.array_equal(after[changed], frame[changed])
            counts.append(int(changed.sum()))
            visible_masks.append(visible)
            diagnostic = frame.copy()
            active = labels[i] > 0
            diagnostic[active] = (.6*frame[active] + .4*colors[labels[i][active] % len(colors)]).astype(np.uint8)
            row = panels([before, after], ['Before', 'Observed object occlusion'])
            close = panels([np.asarray(Image.fromarray(im).crop(crop).resize((720, 480)))
                            for im in (frame, before, after)], ['Source', 'Before', 'After'])
            video.append_data(after)
            comparison.append_data(np.asarray(row))
            detail.append_data(np.asarray(close))
            if i in (0, 8, 16, 24, len(frames)-1):
                row.save(output/f'comparison_{i:03d}.png')
                close.save(output/f'detail_{i:03d}.png')
                panels([frame, diagnostic, after], ['Source', 'Joint visible-instance masks', 'After']).save(output/f'masks_{i:03d}.png')
    assert hashes == {p.name: sha256(p) for p in (render_path, pose_path)}
    np.save(output/'visible_robot_masks.npy', visible_masks)
    report = dict(source=str(source), source_indices=config['indices'], frozen_render=str(frozen),
                  frozen_sha256=hashes, segmentation=str(segmentation), frames=len(frames), fps=config['fps'],
                  restored_pixels=counts, unchanged_outside_intersection=True, pose_modified=False,
                  limitation='Observed RGB visibility transfer, not metric 3D occlusion or physical contact. '
                             'Human pixels outside the robot remain. Hidden object surfaces are not reconstructed.')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clip', choices=['use_chopsticks', 'screw_unscrew_bottle_cap'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--prompts', type=Path)
    parser.add_argument('--segmentation', type=Path)
    run(parser.parse_args())
