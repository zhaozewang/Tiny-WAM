"""Bind a manually inspected full pilot to immutable code/media release hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from sample_rgb_retarget_batch import ROOT,save
from run_rgb_full_production import VARIANT


def release(collection,pilot,queue_recovery=False):
    out=collection/VARIANT;directory=out/pilot
    release_path=out/('release_queue_recovery.json' if queue_recovery else 'release.json')
    if release_path.exists():raise FileExistsError('Release already exists')
    previous=json.loads((out/'release.json').read_text()) if queue_recovery else None
    visual=json.loads((directory/'visual_review.json').read_text())
    done=json.loads((directory/'complete.json').read_text())
    manifest=json.loads((collection/'manifest.json').read_text())
    item=next(x for x in manifest['clips'] if x['key']==pilot)
    result=Path(done['result']);verification=json.loads((result/'review/verification.json').read_text())['samples'][0]
    if visual['decision']!='accepted_for_current_visual_retargeting_baseline' or not verification['processing_verified']:
        raise ValueError('Pilot not accepted')
    if done['frames']!=item['frames_at_stride_two'] or done['frames']!=verification['frames']:
        raise ValueError('Pilot is not a full original video')
    core=json.loads((collection/'baseline_hashes.json').read_text())
    protected=[p for p in core if p.startswith('src/swam/retargeting/') or Path(p).name in
        ('render_rgb_robot_hands.py','reestimate_egodex_rgb.py','refine_rgb_hand_alignment.py','fit_arms_to_frozen_hands.py','repair_forearm_tracks.py')]
    for name in protected:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=core[name]:
            raise ValueError('Accepted fitting baseline changed: '+name)
    files=[*ROOT.glob('src/swam/retargeting/*.py'),*ROOT.glob('scripts/*rgb*.py'),
           ROOT/'scripts/fit_arms_to_frozen_hands.py',ROOT/'scripts/repair_forearm_tracks.py',ROOT/'scripts/initialize_rgb_prompts.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    video=result/'review'/(pilot+'.mp4')
    done['visual_review']='accepted';save(directory/'complete.json',done)
    save(release_path,dict(approved=True,full_duration_pilot_visual_pass=True,pilots=[pilot],
        pilot_video_sha256={pilot:hashlib.sha256(video.read_bytes()).hexdigest()},code_hashes=hashes,
        baseline_core_unchanged=True,manual_prompt_points=False,frames_per_second=15,stride=2,
        selection_unchanged=True,worker_count=1,failures='Isolate and record sample failures; continue remaining original selection without substitution',
        previous_release_sha256=hashlib.sha256((out/'release.json').read_bytes()).hexdigest() if previous else None,
        queue_recovery=queue_recovery,
        caveat='Pilot acceptance authorizes processing; remaining outputs still need individual visual review.'))
    print('Released',pilot,'for',len(manifest['clips']),'full original videos')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--pilot',required=True)
    p.add_argument('--queue-recovery',action='store_true')
    a=p.parse_args();release(a.collection.resolve(),a.pilot,a.queue_recovery)
