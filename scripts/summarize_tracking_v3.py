"""Summarize frozen v3 execution without double-counting nested SAM work."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def run(root,preprocess=None):
    manifest=json.loads((root/'manifest.json').read_text())
    verification=json.loads((root/'review/verification.json').read_text())
    times=json.loads((root/'timings.json').read_text())
    assert all(t['status']=='complete' for t in times), 'Incomplete or failed stages retained'
    fits=[t for t in times if t['label']!='recover_all']
    fit_span=max(t['started']+t['seconds'] for t in fits)-min(t['started'] for t in fits)
    recovery=sum(t['seconds'] for t in times if t['label']=='recover_all')
    preprocessing=0.
    if preprocess:
        upstream=json.loads((preprocess/'timings.json').read_text())
        assert all(t['status']=='complete' for t in upstream)
        preprocessing=sum(t['seconds'] for t in upstream)
    hashes=json.loads((root/'code_hashes.json').read_text()) if (root/'code_hashes.json').exists() else {}
    unchanged={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==value for p,value in hashes.items()}
    assert all(unchanged.values()), 'Frozen source changed during evaluation'
    frames=sum(len(c['indices']) for c in manifest['clips'])
    clips=[]
    for sample in verification['samples']:
        key=sample['key'];case=root/'clips'/key
        predictions=json.loads((root/'predictions'/key/'summary.json').read_text())
        repairs=json.loads((case/'sleeves/repair.json').read_text())
        clips.append(dict(key=key,processing_verified=sample['processing_verified'],
            hands=sample.get('hand_valid_frames'),arms=sample.get('arm_valid_frames'),
            wrist_error=sample.get('wrist_transform_error'),
            quarantined=sum(e['tracked_side'] is None for e in predictions['identity_reconciliation']),
            corrected_identities=sum(e['tracked_side'] is not None for e in predictions['identity_reconciliation']),
            roi_recoveries=len(predictions['recovery']),
            automatic_forearm_attempts=len(repairs),
            accepted_automatic_forearms=sum(r['accepted_pixels']>0 for r in repairs)))
    total=preprocessing+recovery+fit_span
    report=dict(frames=frames,clips=clips,preprocessing_seconds=preprocessing,
        recovery_seconds=recovery,fit_wall_seconds=fit_span,total_active_seconds=total,
        operational_input_fps=frames/total,failed_stages=[],
        all_frozen_files_unchanged=bool(hashes) and all(unchanged.values()),
        timing_scope='Serial source estimate/segmentation + recovery + two-worker fit wall span. Nested forearm SAM is already inside fit span, not added twice. Excludes annotation, extraction, idle gaps and review export. Operational measurement, not a matched speed benchmark.',
        visual_acceptance='See separate manual_review.json; software verification is not a visual/contact pass')
    (root/'evaluation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preprocess',type=Path)
    a=p.parse_args();run(a.output.resolve(),a.preprocess.resolve() if a.preprocess else None)
