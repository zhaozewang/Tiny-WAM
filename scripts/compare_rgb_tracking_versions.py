"""Full-duration source/before/after review; no selected frames removed."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def run(before, after):
    out=after/'comparison'; out.mkdir(exist_ok=False)
    manifest=json.loads((after/'manifest.json').read_text())
    stats=[]
    with imageio.get_writer(out/'all_samples.mp4',fps=15,quality=8,macro_block_size=1) as writer:
        for item in manifest['clips']:
            key=item['key']; case=after/'clips'/key; old=before/'clips'/key
            with imageio.get_reader(case/'source/source.mp4') as src, \
                 imageio.get_reader(old/'arms/robot_arms.mp4') as a, \
                 imageio.get_reader(case/'arms/robot_arms.mp4') as b:
                selected={0,len(item['indices'])//2,len(item['indices'])-1,3,11,13,26}
                for i,index in enumerate(item['indices']):
                    sheet=Image.new('RGB',(1920,384),'white'); draw=ImageDraw.Draw(sheet)
                    for j,(frame,title) in enumerate(zip([src.get_data(index),a.get_data(i),b.get_data(i)],
                                                        ['Source RGB','Before','After'])):
                        sheet.paste(Image.fromarray(frame).resize((640,360)),(640*j,24))
                        draw.text((640*j+10,6),title,fill='black')
                    writer.append_data(np.asarray(sheet))
                    if i in selected: sheet.save(out/f'{key}_{i:03d}.jpg')
            p=np.load(old/'smooth/native_parameters.npz'); q=np.load(case/'smooth/native_parameters.npz')
            rows={}
            for side in ('left','right'):
                common=p[side+'_valid'] & q[side+'_valid']
                delta=np.max(np.abs(p[side][common]-q[side][common]),axis=1)
                rows[side]=dict(before=int(p[side+'_valid'].sum()),after=int(q[side+'_valid'].sum()),
                    unchanged_common=int((delta==0).sum()),changed_common=int((delta!=0).sum()),
                    max_parameter_delta=float(delta.max()) if len(delta) else None)
            stats.append(dict(key=key,hands=rows))
    (out/'parameter_audit.json').write_text(json.dumps(stats,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    a=p.parse_args();run(a.before.resolve(),a.after.resolve())
