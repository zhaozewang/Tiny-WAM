"""Check input annotations against RGB independently of retargeter fit metrics."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = json.loads((args.output/'report.json').read_text())
    targets = np.load(args.output/'targets.npz')
    solutions = np.load(args.output/'solutions.npz')
    config = json.loads((Path(report['source'])/'prepare.json').read_text())
    intrinsic = np.asarray(config['intrinsic'])
    intrinsic[:2] *= 2
    reader = imageio.get_reader(args.output/'source.mp4')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    colors = [(255, 50, 50), (50, 210, 255), (80, 255, 90), (255, 180, 30), (220, 90, 255)]
    with imageio.get_writer(args.output/'annotation_check.mp4', fps=report['fps'], macro_block_size=1, quality=8) as writer:
        for i, camera in enumerate(solutions['camera']):
            frame = Image.fromarray(reader.get_data(i))
            draw = ImageDraw.Draw(frame)
            for side in ('left', 'right'):
                p = targets[side+'_hand'][i] @ camera[:3, :3].T + camera[:3, 3]
                uv = p @ intrinsic.T
                uv = uv[:, :2]/uv[:, 2:]
                for k, indices in enumerate([[0, 17, 18, 19, 20], [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16]]):
                    draw.line([tuple(x) for x in uv[indices]], fill=colors[k], width=2)
                    x, y = uv[indices[-1]]
                    draw.ellipse((x-4, y-4, x+4, y+4), outline=colors[k], width=2)
            draw.text((10, 10), 'Original EgoDex landmarks (not robot predictions)', fill='white', font=font, stroke_width=1, stroke_fill='black')
            writer.append_data(np.asarray(frame))
            if i in [0, len(solutions['camera'])//2, len(solutions['camera'])-1]:
                frame.save(args.output/f'annotation_{i:03d}.png')
    reader.close()


if __name__ == '__main__':
    main()
