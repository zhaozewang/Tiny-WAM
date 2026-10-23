"""Replay cached automatic positives with one versioned negative-point policy."""
import argparse
import json
import shutil
from pathlib import Path
from evaluate_auto_initialization import run
from evaluate_rgb_heldout import prompts
from initialize_rgb_prompts import cross_target_negatives
from sample_rgb_retarget_batch import save


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--automatic',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    source=a.source.resolve();root=a.output.resolve()
    run(source,root,'prepare')
    seeds=json.loads((a.automatic/'source_seeds.json').read_text())
    save(root/'source_seeds.json',{k:cross_target_negatives(v) for k,v in seeds.items()})
    shutil.copytree(a.automatic/'automatic_prompt_audit',root/'automatic_prompt_audit')
    save(root/'negative_policy.json',dict(version='cross_target_v1',source=str(a.automatic.resolve()),
        changes='Replace generated negatives with other targets positives, excluding points within 16px of own positives.',
        no_manual_points=True,retargeting_changes=False,bulk_release=False))
    prompts(root,json.loads((root/'manifest.json').read_text()))
    for stage in ('segment','sleeves','review'):run(source,root,stage)
