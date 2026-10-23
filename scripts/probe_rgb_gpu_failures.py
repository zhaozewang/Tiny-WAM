"""Compare fresh-test failure frames against CPU without changing inference."""
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from sample_rgb_retarget_batch import ROOT,COMPARE


def run():
    source=ROOT/'results/rgb_tracking_v2_fresh_20261008'
    output=ROOT/'results/rgb_tracking_v2_fresh_cpu_control_20261008'
    output.mkdir(exist_ok=False)
    cases={'00_test_insert_remove_tennis_ball_11':[64],
           '02_test_assemble_disassemble_furniture_bench_lamp_21':[3],
           '03_test_stack_unstack_plates_47':[13,26]}
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    rows=[]
    for key,indices in cases.items():
        case=source/'clips'/key;dst=output/key
        command=[COMPARE,ROOT/'scripts/refine_rgb_hand_alignment.py','--source',case/'source','--base',source/'hands'/key,
                 '--segmentation',case/'segment','--predictions',source/'predictions'/key,
                 '--frames',*indices,'--sweeps','2','2','1','1','1','.5','.5','.25','--fast-mask-readback','--output',dst]
        start=time.perf_counter()
        with (output/(key+'.log')).open('x') as log:
            subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        a=np.load(case/'refine/native_parameters.npz');b=np.load(dst/'native_parameters.npz')
        ar=np.load(case/'refine/robot_render.npz');br=np.load(dst/'robot_render.npz')
        row=dict(key=key,indices=indices,seconds=time.perf_counter()-start,
                 poses_identical=all(np.array_equal(a[n][indices],b[n],equal_nan=True) for n in a.files),
                 mask_disagreement_pixels=int(np.count_nonzero(ar['mask'][indices]!=br['mask'])))
        rows.append(row);print(row,flush=True)
    (output/'report.json').write_text(json.dumps(dict(samples=rows,
         scope='Diagnostic CPU control on fresh-test failures, not parameter tuning or another held-out split'),indent=2)+'\n')


if __name__=='__main__':run()
