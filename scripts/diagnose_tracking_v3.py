"""Inspect observed masks and detector crops before changing tracking policy."""
import json
from pathlib import Path
import sys
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw
import pyrender

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tinywam.retargeting.geometry import SharpaHand, project

source = ROOT / 'results/rgb_tracking_v2_fresh_20261008'
output = ROOT / 'results/rgb_tracking_v3_diagnostics_20261008'
output.mkdir(exist_ok=True)
manifest = json.loads((source / 'manifest.json').read_text())
for number, frame, side in [(0, 64, 'left'), (2, 3, 'right'), (3, 13, 'left')]:
    item = manifest['clips'][number]; key = item['key']; case = source / 'clips' / key
    with imageio.get_reader(case / 'source/source.mp4') as reader:
        rgb = np.array(Image.fromarray(reader.get_data(item['indices'][frame])).resize((960,540)))
    labels = np.load(case / 'segment/labels.npy')[frame]
    sleeves = np.load(case / 'sleeves/labels.npy')[frame]
    pose = np.load(case / 'smooth/native_parameters.npz')[side][frame]
    k = np.load(source / 'hands' / key / 'qpos.npz')['intrinsic'].copy(); k[:2] /= 2
    hand = SharpaHand(side, pyrender.Scene()); points, transforms = hand.forward(pose)
    wrist = project(transforms[hand.root][None,:3,3], k)[0]
    palm = project(points[[1,5,9,13]], k).mean(0)
    panels = []
    for name, mask in [('source', np.zeros_like(labels, bool)),
                       ('hand', labels == (2 if side == 'left' else 3)),
                       ('sleeve', sleeves == (1 if side == 'left' else 2))]:
        pic = rgb.copy(); pic[mask] = (.5*pic[mask]+.5*np.array([255,40,40])).astype('uint8')
        pic = Image.fromarray(pic); draw = ImageDraw.Draw(pic)
        draw.text((10,10), name, fill='white')
        draw.line([tuple(wrist), tuple(palm)], fill='lime', width=4)
        draw.ellipse([*(wrist-5),*(wrist+5)], fill='cyan')
        records = json.loads((source/'initial_predictions'/key/'predictions.json').read_text())
        for pred in records[frame]['wilor']:
            box = np.array(pred['hand_bbox']) / 2
            draw.rectangle(box.tolist(), outline='yellow', width=2)
            draw.text(tuple(box[:2]), str(pred['is_right']), fill='yellow')
        panels.append(pic)
    sheet = Image.new('RGB',(960,540*3))
    for i,pic in enumerate(panels): sheet.paste(pic,(0,i*540))
    sheet.save(output / f'{number}_{frame}.jpg')
    print(key, frame, 'wrist', wrist, 'palm', palm,
          'boxes', [(p['is_right'],p['hand_bbox']) for p in records[frame]['wilor']])
