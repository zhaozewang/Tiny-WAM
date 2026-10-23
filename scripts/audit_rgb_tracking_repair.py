"""Record what a tracking repair changed without calling coverage accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np


def run(before,after):
    rows=[]
    for item in json.loads((before/'manifest.json').read_text())['clips']:
        key=item['key'];a=before/'clips'/key;b=after/'clips'/key
        old=np.load(a/'smooth/native_parameters.npz');new=np.load(b/'smooth/native_parameters.npz')
        oa=np.load(a/'arms/arms.npz');na=np.load(b/'arms/arms.npz')
        sides={}
        for side in ('left','right'):
            common=old[side+'_valid']&new[side+'_valid']
            np.testing.assert_array_equal(old[side][common],new[side][common])
            sides[side]=dict(old_hands=int(old[side+'_valid'].sum()),new_hands=int(new[side+'_valid'].sum()),
                             common_hand_poses_identical=True,old_arms=int(oa[side+'_valid'].sum()),
                             new_arms=int(na[side+'_valid'].sum()))
        events=json.loads((after/'predictions'/key/'summary.json').read_text())['identity_reconciliation']
        report=json.loads((b/'arms/report.json').read_text())
        assert report['visible_hand_pixels_identical'] and not report['hand_parameters_modified']
        rows.append(dict(key=key,sides=sides,identity_events=events,
                         hand_pixels_preserved_during_arm_fit=True))
    result=dict(samples=rows,scope='Development regression; no independent visual/contact accuracy claim')
    (after/'regression_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    a=p.parse_args();run(a.before,a.after)
