"""Measure serial versus two-process fitting on the same six development frames."""
from tinywam.paths import external_path
import argparse
from concurrent.futures import ThreadPoolExecutor
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
    root=ROOT/'results/rgb_random_pilot_20261007'
    keys=['01_test_assemble_disassemble_furniture_bench_drawer_11','03_test_charge_uncharge_device_16']
    results=[]
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for workers in [1,2]:
        def fit(key):
            destination=output/f'workers{workers}'/key;destination.parent.mkdir(exist_ok=True)
            case=root/'clips'/key
            cmd=[PYTHON,ROOT/'scripts/refine_rgb_hand_alignment.py','--source',case/'source','--base',root/'hands'/key,
                 '--segmentation',case/'segment','--predictions',root/'predictions'/key,'--frames','0','13','25',
                 '--sweeps','2','2','1','1','1','.5','.5','.25','--fast-mask-readback','--output',destination]
            with (destination.parent/(key+'.log')).open('x') as log:
                subprocess.run(list(map(str,cmd)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(fit,keys))
        results.append(dict(workers=workers,wall_seconds=time.perf_counter()-start,frames=6))
        print(results[-1],flush=True)
    for key in keys:
        for file in ('native_parameters.npz','robot_render.npz'):
            a=np.load(output/'workers1'/key/file);b=np.load(output/'workers2'/key/file)
            for field in a.files:
                if field!='rgb':np.testing.assert_array_equal(a[field],b[field])
    result=dict(results=results,speedup=results[0]['wall_seconds']/results[1]['wall_seconds'],
                pose_and_mask_exact_match=True,development_only=True,
                caveat='Six development frames, one paired trial; not a full-dataset scaling guarantee.')
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output.resolve())
