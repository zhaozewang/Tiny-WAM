"""Fixed-pose clip-plane ablation: no fitting, material or geometry changes."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
import pyrender
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tinywam.retargeting.geometry import SharpaHand


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    poses=np.load(args.case/'smooth/native_parameters.npz')
    k=np.load(args.case/'hand_fit_visible'/args.case.name/'qpos.npz')['intrinsic'].copy();k[:2]/=2
    canvas=Image.new('RGB',(1920,1140),'white');stats=[]
    for j,near in enumerate((.01,.1,1.,8.)):
        scene=pyrender.Scene(bg_color=[.15,.15,.15,1],ambient_light=[.65]*3)
        for side in ('left','right'):
            if not poses[side+'_valid'][args.frame]:continue
            hand=SharpaHand(side,scene);hand.show(scene,poses[side][args.frame])
        scene.add(pyrender.IntrinsicsCamera(k[0,0],k[1,1],k[0,2],k[1,2],znear=near,zfar=100),pose=np.diag([1.,-1.,-1.,1.]))
        scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=np.diag([1.,-1.,-1.,1.]))
        renderer=pyrender.OffscreenRenderer(960,540)
        rgb,depth=renderer.render(scene);renderer.delete()
        Image.fromarray(rgb).save(args.output/f'near_{near}.png')
        stats.append(dict(near=near,dark_mesh_pixels=int(((rgb.max(-1)<60)&(depth>0)).sum()),coverage=int((depth>0).sum())))
        canvas.paste(Image.fromarray(rgb),(j%2*960,j//2*570+30))
        ImageDraw.Draw(canvas).text((j%2*960+8,j//2*570+8),str(stats[-1]),fill='black')
    canvas.save(args.output/'comparison.png')
    (args.output/'report.json').write_text(json.dumps(stats,indent=2))
    print(stats)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--frame',type=int,default=32);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
