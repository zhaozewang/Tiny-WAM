"""Full-clip GPU scoring regression against existing CPU fits, without tuning."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from sample_rgb_retarget_batch import COMPARE,ROOT


def run(source,output,keys):
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for key in keys:
        case=source/'clips'/key;dst=output/key
        command=[COMPARE,ROOT/'scripts/refine_rgb_hand_alignment.py','--source',case/'source',
                 '--base',source/'hands'/key,'--segmentation',case/'segment','--predictions',source/'predictions'/key,
                 '--all-frames','--sweeps','2','2','1','1','1','.5','.5','.25',
                 '--fast-mask-readback','--gpu-objective','--output',dst]
        start=time.perf_counter()
        with (output/(key+'.log')).open('x') as log:
            subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        elapsed=time.perf_counter()-start
        a=np.load(case/'refine/native_parameters.npz');b=np.load(dst/'native_parameters.npz')
        ar=np.load(case/'refine/robot_render.npz');br=np.load(dst/'robot_render.npz')
        row=dict(key=key,frames=len(a['indices']),seconds=elapsed,
                 poses_identical=all(np.array_equal(a[n],b[n],equal_nan=True) for n in a.files),
                 mask_disagreement_pixels=int(np.count_nonzero(ar['mask']!=br['mask'])),
                 max_parameter_difference=max(float(np.nanmax(np.abs(a[s]-b[s]))) for s in ('left','right')))
        rows.append(row);print(row,flush=True)
    (output/'report.json').write_text(json.dumps(dict(samples=rows,development_regression=True,
        gate_passed=all(r['poses_identical'] and not r['mask_disagreement_pixels'] for r in rows)),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--clips',nargs='+',required=True)
    a=p.parse_args();run(a.source.resolve(),a.output.resolve(),a.clips)
