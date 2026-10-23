"""Explicit development repair; old held-out data is now regression evidence."""
import argparse
import json
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
from sample_rgb_retarget_batch import COMPARE, RGB, ROOT, command_for, save
from evaluate_rgb_heldout import execute


def run(source,root,stage):
    if stage=='prepare':
        root.mkdir(parents=True,exist_ok=False)
        shutil.copy2(source/'manifest.json',root/'manifest.json')
        save(root/'provenance.json',dict(source=str(source),split='development regression; previously inspected held-out',
             changes=['persistent SAM identity reconciliation','observed proximal hand-track fallback'],
             unchanged=['resolution','sweeps','manual prompts','native geometry','immutable hand-to-arm wrist constraint']))
        for item in json.loads((root/'manifest.json').read_text())['clips']:
            case=root/'clips'/item['key'];case.mkdir(parents=True)
            for name in ('source','segment','sleeves'):
                (case/name).symlink_to(source/'clips'/item['key']/name,target_is_directory=True)
        return
    if stage=='recover':
        if not execute(root,'recover_all',[RGB,__file__,'--source',source,'--output',root,'--stage','recover-worker']):
            raise RuntimeError('Recovery failed')
        return
    if stage=='recover-worker':
        from repair_rgb_retarget_batch import recover
        recover(source,root,predictions_root=source/'initial_predictions',reconcile_identities=True)
        return
    (root/'hands').mkdir()
    def fit(item):
        key=item['key'];case=root/'clips'/key
        summary=json.loads((root/'predictions'/key/'summary.json').read_text())
        if summary['identity_reconciliation']:
            for name in ('hands_visible','refine','smooth'):
                command=command_for(name,root,item)
                if name!='hands_visible':command+=['--fast-mask-readback']
                if not execute(root,key+'/'+name,command):return
                if name=='hands_visible':(root/'hands'/key).symlink_to(case/'hand_fit_visible'/key,target_is_directory=True)
        else:
            (root/'hands'/key).symlink_to(source/'hands'/key,target_is_directory=True)
            (case/'smooth').symlink_to(source/'clips'/key/'smooth',target_is_directory=True)
        execute(root,key+'/arms',command_for('arms',root,item)+
                ['--proximal-exit','--fast-mask-readback','--observed-forearm-fallback'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fit,json.loads((root/'manifest.json').read_text())['clips']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','recover','recover-worker','fit'],required=True)
    a=p.parse_args();run(a.source.resolve(),a.output.resolve(),a.stage)
