"""Versioned repairs with immutable source masks, prediction audit and videos."""
import argparse
import json
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
from sample_rgb_retarget_batch import COMPARE, RGB, ROOT, command_for, save
from evaluate_rgb_heldout import execute, freeze


def run(source,root,stage,fresh=False):
    if stage=='prepare':
        root.mkdir(parents=True,exist_ok=False)
        shutil.copy2(source/'manifest.json',root/'manifest.json')
        save(root/'provenance.json',dict(source=str(source),split='fresh task-disjoint test' if fresh else 'development regression',
            changes=['quarantine unsupported detections, including potentially real untracked new hands',
                     're-infer corrected handedness on original detector crop',
                     'reject short/disconnected forearm masks; automatic geometry-guided SAM re-prompt'],
            unchanged=['source frames','manual prompts','hand/arm dimensions','frozen wrist constraint',
                       'render resolution','refinement sweeps']))
        (root/'initial_predictions').symlink_to(source/'initial_predictions',target_is_directory=True)
        for item in json.loads((root/'manifest.json').read_text())['clips']:
            case=root/'clips'/item['key'];case.mkdir(parents=True)
            for name in ('source','segment'):
                (case/name).symlink_to(source/'clips'/item['key']/name,target_is_directory=True)
        return
    if stage=='freeze':
        if (root/'protocol.json').exists(): raise FileExistsError('Protocol already frozen')
        freeze(root,tracking_v2=True,gpu_objective=True)
        protocol=json.loads((root/'protocol.json').read_text())
        protocol.update(tracking_v3=True,split=json.loads((root/'provenance.json').read_text())['split'],
            prompts='One manual source-frame-zero initialization, plus automatic pose-guided forearm re-prompts at least 16 frames apart. No manual output corrections.',
            untracked_hands='Quarantined, including potentially real newly entering hands; requires track initialization.')
        save(root/'protocol.json',protocol)
        return
    if stage=='recover':
        if not execute(root,'recover_all',[RGB,__file__,'--source',source,'--output',root,'--stage','recover-worker']):
            raise RuntimeError('Recovery failed')
        return
    if stage=='recover-worker':
        from repair_rgb_retarget_batch import recover
        recover(source,root,predictions_root=source/'initial_predictions',reconcile_identities=True,tracking_v3=True)
        return
    (root/'hands').mkdir()
    def fit(item):
        key=item['key'];case=root/'clips'/key
        for name in ('hands_visible','refine','smooth'):
            command=command_for(name,root,item)
            if name!='hands_visible':command+=['--gpu-objective']
            if not execute(root,key+'/'+name,command):raise RuntimeError(key+'/'+name)
            if name=='hands_visible':(root/'hands'/key).symlink_to(case/'hand_fit_visible'/key,target_is_directory=True)
        if not execute(root,key+'/repair_sleeves',[COMPARE,ROOT/'scripts/repair_forearm_tracks.py',
                '--output',root,'--source',source,'--key',key]):raise RuntimeError('Forearm repair failed')
        if not execute(root,key+'/arms',command_for('arms',root,item)+
                ['--proximal-exit','--fast-mask-readback','--validated-forearm']):raise RuntimeError('Arms failed')
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fit,json.loads((root/'manifest.json').read_text())['clips']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','freeze','recover','recover-worker','fit'],required=True)
    p.add_argument('--fresh',action='store_true')
    a=p.parse_args();run(a.source.resolve(),a.output.resolve(),a.stage,a.fresh)
