"""Aggregate coverage and elapsed stage cost without treating mask IoU as accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def run(root):
    manifest=json.loads((root/'manifest.json').read_text())
    verification=json.loads((root/'review/verification.json').read_text())
    timings=json.loads((root/'timings.json').read_text())
    completed=[r for r in timings if r['status']=='complete']
    fits=[r for r in completed if r['label'].rsplit('/',1)[-1] in ('hands_visible','refine','smooth','arms')]
    upstream=[r for r in completed if r not in fits]
    fit_wall=max(r['started']+r['seconds'] for r in fits)-min(r['started'] for r in fits)
    upstream_seconds=sum(r['seconds'] for r in upstream)
    frames=sum(len(item['indices']) for item in manifest['clips'])
    stages={}
    for row in completed:
        name=row['label'].rsplit('/',1)[-1]
        stages[name]=stages.get(name,0.)+row['seconds']
    recovery=[]
    for item in manifest['clips']:
        key=item['key']
        initial=json.loads((root/'initial_predictions'/key/'predictions.json').read_text())
        predicted=json.loads((root/'predictions'/key/'predictions.json').read_text())
        summary=json.loads((root/'predictions'/key/'summary.json').read_text())
        recovery.append(dict(key=key,recovered=len(summary['recovery']),
            before={s:sum(sum(int(p['is_right'])==side for p in r['wilor'])==1 for r in initial) for s,side in [('left',0),('right',1)]},
            after={s:sum(sum(int(p['is_right'])==side for p in r['wilor'])==1 for r in predicted) for s,side in [('left',0),('right',1)]}))
    hashes=json.loads((root/'code_hashes.json').read_text())
    algorithm_files=[p for p in hashes if p.startswith('src/tinywam/retargeting/') or Path(p).name in
                     ('reestimate_egodex_rgb.py','render_rgb_robot_hands.py','refine_rgb_hand_alignment.py','fit_arms_to_frozen_hands.py','repair_rgb_retarget_batch.py')]
    unchanged={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==hashes[p] for p in algorithm_files}
    assert all(unchanged.values()),'Algorithm changed during heldout run'
    result=dict(clips=len(manifest['clips']),frames=frames,
        processing_verified=sum(s['processing_verified'] for s in verification['samples']),
        fit_wall_seconds=fit_wall,upstream_active_seconds=upstream_seconds,
        active_pipeline_seconds=fit_wall+upstream_seconds,
        active_pipeline_input_fps=frames/(fit_wall+upstream_seconds),
        fit_input_fps=frames/fit_wall,stage_process_seconds=stages,
        failed_stages=[r for r in timings if r['status']=='failed'],
        recovery=recovery,algorithm_hashes_unchanged=unchanged,
        timing_scope='Upstream serial stage wall times + concurrent two-worker fitting wall span. Excludes annotation, idle gaps, extraction, benchmark experiments and review-media export.',
        caveats=['Task-disjoint by archive label, not a held-out subject/environment guarantee.',
                 'Held out from our engineering/tuning, not a statement about pretrained model training data.',
                 'First-frame manual SAM prompts remain necessary. Not zero-annotation automation.',
                 'Detection coverage and fitted mask IoU are not pose/contact ground-truth accuracy.'])
    (root/'evaluation_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    contact=Image.open(root/'review/contact.png')
    for start in range(0,len(manifest['clips']),4):
        contact.crop((0,start*320,1920,min(start+4,len(manifest['clips']))*320)).save(root/f'review/contact_{start//4}.png')
    print(json.dumps({k:v for k,v in result.items() if k not in ('recovery','algorithm_hashes_unchanged')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output.resolve())
