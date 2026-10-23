"""Paired old-code/current-code benchmark on development images only."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    result.ROOT=ROOT
    return result


def run(output,repeats):
    output.mkdir(parents=True,exist_ok=False)
    snapshot=ROOT/'results/rgb_random_repair_v1/review/source_snapshot'
    old=module(snapshot/'scripts/refine_rgb_hand_alignment.py','baseline_refine')
    geo=module(snapshot/'src/swam/retargeting/geometry.py','baseline_geometry')
    old.SharpaHand=geo.SharpaHand
    new=module(ROOT/'scripts/refine_rgb_hand_alignment.py','fast_refine')
    pilot=ROOT/'results/rgb_random_pilot_20261007';key='03_test_charge_uncharge_device_16'
    args=dict(source=pilot/'clips'/key/'source',base=pilot/'hands'/key,
              segmentation=pilot/'clips'/key/'segment',predictions=pilot/'predictions'/key,
              frames=[0,13,25],all_frames=False,corrections_from=None,sweeps=[2,2,1,1,1,.5,.5,.25])
    rows=[]
    for repeat in range(repeats):
        for name,impl in ([('baseline',old),('fast',new)] if repeat%2==0 else [('fast',new),('baseline',old)]):
            destination=output/f'{name}_{repeat}';start=time.perf_counter()
            with (output/f'{name}_{repeat}.log').open('x') as log,contextlib.redirect_stdout(log):
                impl.run(SimpleNamespace(**args,output=destination,fast_mask_readback=name=='fast'))
            rows.append(dict(mode=name,repeat=repeat,wall_seconds=time.perf_counter()-start))
            print(rows[-1],flush=True)
    reference=np.load(output/'baseline_0/native_parameters.npz')
    reference_render=np.load(output/'baseline_0/robot_render.npz')
    for row in rows:
        destination=output/f'{row["mode"]}_{row["repeat"]}'
        params=np.load(destination/'native_parameters.npz');render=np.load(destination/'robot_render.npz')
        row['poses_identical']=all(np.array_equal(reference[k],params[k],equal_nan=True) for k in reference.files)
        row['masks_identical']=np.array_equal(reference_render['mask'],render['mask'])
        delta=render['rgb'].astype(float)-reference_render['rgb'].astype(float)
        row['rgb_mae']=float(np.abs(delta).mean())
        row['rgb_psnr']=float(10*np.log10(255**2/max(float((delta**2).mean()),1e-12)))
        assert row['poses_identical'] and row['masks_identical']
    medians={name:float(np.median([r['wall_seconds'] for r in rows if r['mode']==name])) for name in ('baseline','fast')}
    result=dict(rows=rows,median_seconds=medians,speedup=medians['baseline']/medians['fast'],
                frames_per_run=3,development_only=True,profiling_enabled=False,
                rgb_note='Unordered coplanar draw ties may differ between baseline repetitions; report rather than claiming pixel-exact RGB.')
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();run(a.output.resolve(),a.repeats)
