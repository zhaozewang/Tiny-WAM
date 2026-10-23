"""Recover missing forearm seeds using the existing frozen-wrist repair path."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from initialize_rgb_prompts import validate
from sample_rgb_retarget_batch import save


def recover_seeds(root):
    manifest=json.loads((root/'manifest.json').read_text());seeds={}
    failure=json.loads((root/'automatic_prompt_audit/failure.json').read_text())
    if failure['error']!="ValueError('No observed forearm initialized; requires review')":
        raise ValueError('Not a missing-forearm failure; preserve for separate diagnosis')
    for i,item in enumerate(manifest['clips']):
        raw=(root/'automatic_prompt_audit'/(item['key']+'.txt')).read_text()
        seeds[str(i)]=validate(raw,require_forearm=False)
    if (root/'source_seeds.json').exists():raise FileExistsError('Seeds already exist')
    save(root/'source_seeds.json',seeds)
    save(root/'initialization_recovery.json',dict(method='Reuse original valid hand/object seeds; defer missing forearms to existing pose-guided repair',
         original_failure=failure,manual_points=False,output_validation_unchanged=True))


def empty_sleeves(root,key):
    case=root/'clips'/key;source=case/'source'
    prompts=json.loads((case/'sleeves_prompts.json').read_text())
    if prompts['objects']:raise ValueError('Do not replace observed sleeve prompts with empty masks')
    config=json.loads((source/'prepare.json').read_text());out=case/'sleeves';out.mkdir(exist_ok=False)
    labels=np.lib.format.open_memmap(out/'labels.npy',mode='w+',dtype=np.uint16,shape=(len(config['indices']),540,960))
    labels[:]=0;labels.flush()
    save(out/'segmentation.json',dict(source=str(source.resolve()),source_indices=config['indices'],
        source_prepare_sha256=hashlib.sha256((source/'prepare.json').read_bytes()).hexdigest(),
        output_size=[960,540],objects=[],observed=False,
        method='No initial forearm observation; frozen-wrist geometry-guided SAM repair remains mandatory',
        semantics='Unknown, not a claim that no forearm exists'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['seeds','empty-sleeves'],required=True);p.add_argument('--key')
    a=p.parse_args()
    if a.stage=='seeds':recover_seeds(a.output.resolve())
    else:empty_sleeves(a.output.resolve(),a.key)
