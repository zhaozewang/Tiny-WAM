"""Observed-view occlusion mattes, not inferred metric scene depth."""
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def composite(frame, robot_rgb, robot_mask, object_mask):
    """Preserve visible source-object pixels; do not hide an entire robot link."""
    if frame.shape != robot_rgb.shape or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError('RGB shapes must agree and have three channels')
    if robot_mask.shape != frame.shape[:2] or object_mask.shape != robot_mask.shape:
        raise ValueError('Mask and image dimensions must agree')
    visible = robot_mask.astype(bool) & ~object_mask.astype(bool)
    return np.where(visible[..., None], robot_rgb, frame), visible


def compare(source, render_dir, segmentation, output):
    output.mkdir(parents=True, exist_ok=False)
    frames = np.load(source / 'source_frames.npy')
    config = json.loads((source / 'prepare.json').read_text())
    objects = np.load(segmentation / 'object_masks.npy').astype(bool)
    render_path = render_dir / 'render.npz'
    if not render_path.exists():
        render_path = render_dir / 'robot_render.npz'
    with np.load(render_path) as render:
        rgb, masks = render['rgb'], render['mask'].astype(bool)
    if frames.shape != rgb.shape or objects.shape != masks.shape:
        raise ValueError('Source, render and segmentation must cover identical frames')
    provenance = json.loads((segmentation / 'segmentation.json').read_text())
    if provenance['source_indices'] != config['indices']:
        raise ValueError('Segmentation uses different source frames')
    counts, visibility = [], []
    with imageio.get_writer(output / 'comparison.mp4', fps=config['fps'], macro_block_size=1) as writer, \
            imageio.get_writer(output / 'corrected.mp4', fps=config['fps'], macro_block_size=1) as corrected:
        for index, frame in enumerate(frames):
            before = np.where(masks[index, ..., None], rgb[index], frame)
            after, visible = composite(frame, rgb[index], masks[index], objects[index])
            visibility.append(visible)
            counts.append(int((masks[index] & objects[index]).sum()))
            assert np.array_equal(after[objects[index]], frame[objects[index]])
            diagnostic = frame.copy()
            diagnostic[objects[index]] = (frame[objects[index]] * .4 + np.array([0, 150, 0])).astype(np.uint8)
            panels = [frame, before, diagnostic, after]
            canvas = Image.fromarray(np.concatenate(panels, axis=1))
            labeled = Image.new('RGB', (canvas.width, canvas.height + 24), 'white')
            labeled.paste(canvas, (0, 24))
            draw = ImageDraw.Draw(labeled)
            for col, label in enumerate(['Source', 'Original overlay', 'Visible object mask', 'Occlusion-aware overlay']):
                draw.text((col * frame.shape[1] + 5, 5), label, fill='black')
            writer.append_data(np.asarray(labeled))
            corrected.append_data(after)
            if index in (0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1):
                labeled.save(output / f'comparison_{index:03d}.png')
    np.save(output / 'visible_robot_masks.npy', visibility)
    report = dict(source=str(source.resolve()), render=str(render_path.resolve()),
                  segmentation=str(segmentation.resolve()), frames=len(frames), fps=config['fps'],
                  restored_object_pixels_per_frame=counts, trajectory_modified=False,
                  source_object_pixels_preserved=True,
                  limitation='Transfers observed image visibility, not 3D depth. New robot geometry, hidden object surfaces and residual human pixels remain unresolved.')
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
