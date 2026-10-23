"""Paired real-data fitting test; no profiler and no reduced fitting settings."""
from swam.paths import external_path
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path(external_path('retarget', 'compare-env/bin/python'))


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'results/rgb_random_pilot_20261007'
    key='03_test_charge_uncharge_device_16';case=source/'clips'/key
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    rows=[];reference=None
    for repeat in range(3):
        for mode in (['cpu','gpu'] if repeat%2==0 else ['gpu','cpu']):
            dst=output/f'{mode}_{repeat}'
            cmd=[PYTHON,ROOT/'scripts/refine_rgb_hand_alignment.py','--source',case/'source',
                 '--base',source/'hands'/key,'--segmentation',case/'segment','--predictions',source/'predictions'/key,
                 '--frames','0','13','25','--sweeps','2','2','1','1','1','.5','.5','.25',
                 '--fast-mask-readback','--output',dst]+(['--gpu-objective'] if mode=='gpu' else [])
            start=time.perf_counter()
            with (output/f'{mode}_{repeat}.log').open('x') as log:
                subprocess.run(list(map(str,cmd)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            elapsed=time.perf_counter()-start
            if reference is None:reference=dst
            a=np.load(reference/'native_parameters.npz');b=np.load(dst/'native_parameters.npz')
            same=all(np.array_equal(a[n],b[n],equal_nan=True) for n in a.files)
            ar=np.load(reference/'robot_render.npz');br=np.load(dst/'robot_render.npz')
            rows.append(dict(mode=mode,repeat=repeat,seconds=elapsed,poses_identical=same,
                 mask_disagreement_pixels=int(np.count_nonzero(ar['mask']!=br['mask'])),
                 max_parameter_difference=max(float(np.nanmax(np.abs(a[s]-b[s]))) for s in ('left','right'))))
            print(rows[-1],flush=True)
    medians={m:float(np.median([r['seconds'] for r in rows if r['mode']==m])) for m in ('cpu','gpu')}
    result=dict(rows=rows,median_seconds=medians,speedup=medians['cpu']/medians['gpu'],
                frames_per_run=3,development_only=True,
                scope='Queues the two +/- coordinate candidates; compute-reduces objective on GPU and reads 64 bytes. Triangle draw calls remain sequential.',
                retain_only_if_quality_verified=True)
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output.resolve())
