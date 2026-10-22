"""Record local reproduction dependencies without copying private media or weights."""
from swam.paths import external_path
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(external_path('retarget'))
PACKAGES = ['numpy','scipy','torch','torchvision','pin','pinocchio','nlopt','dex-retargeting',
            'retargeting','pyrender','trimesh','imageio','imageio-ffmpeg','Pillow','h5py',
            'PyOpenGL','opencv-python','opencv-contrib-python','mediapipe','protobuf',
            'wilor_mini','ultralytics','smplx','chumpy','timm','einops','hydra-core',
            'iopath','sam-2','sam2','pytest']


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def command(args):
    return subprocess.check_output(list(map(str,args)),text=True).strip()


def capture(output):
    if output.exists():
        raise FileExistsError('Keep earlier environment snapshots; use a new output path')
    environments = dict(compare=CACHE/'compare-env/bin/python',rgb=CACHE/'rgb-env/bin/python',
                        segmentation=Path(external_path('sam_env', 'bin/python')))
    probe = '''import importlib.metadata as m,json,sys
names=json.loads(sys.argv[1]); packages={}
for name in names:
    try: packages[name]=m.version(name)
    except m.PackageNotFoundError: pass
print(json.dumps(dict(python=sys.version,executable=sys.executable,packages=packages)))
'''
    report = dict(scope='Local environment snapshot; not a portable dependency lock or redistribution approval',
                  environments={},upstream={},inputs={},weights={},sharpa_assets={})
    for name,python in environments.items():
        report['environments'][name] = json.loads(command([python,'-c',probe,json.dumps(PACKAGES)]))
    for name,path in dict(wilor=CACHE/'WiLoR',wilor_mini=CACHE/'WiLoR-mini',
                          dex_retargeting=CACHE/'dex-retargeting',objectives=CACHE/'objective-retargeting',
                          cutie=CACHE/'Cutie',sam2=Path(external_path('sam')),sharpa=Path(external_path('sharpa'))).items():
        report['upstream'][name] = dict(path=str(path),commit=command(['git','-C',path,'rev-parse','HEAD']),
                                       tracked_diff_sha256=hashlib.sha256(command(['git','-C',path,'diff','HEAD']).encode()).hexdigest())
    for clip in ['use_chopsticks','screw_unscrew_bottle_cap']:
        source = ROOT/'results/retargeting_import_20261007'/clip/'source'
        report['inputs'][clip] = dict(source=str(source),prepare=json.loads((source/'prepare.json').read_text()),
                                      sha256={name:digest(source/name) for name in ['source.mp4','source.hdf5','prepare.json']})
    checkpoints = [Path(external_path('sam', 'sam2.1_hiera_large.pt')),CACHE/'cutie-base-mega.pth']
    checkpoints += [p for p in (CACHE/'wilor-weights').rglob('*') if p.is_file() and p.suffix in
                    ('.pt','.pth','.ckpt','.pkl','.npz','.onnx','.yaml','.yml')]
    for p in checkpoints:
        report['weights'][str(p)] = dict(bytes=p.stat().st_size,sha256=digest(p))
    for p in sorted((ROOT/'assets/sharpa').rglob('*')):
        if p.is_file() and '.git' not in p.relative_to(ROOT).parts:
            report['sharpa_assets'][str(p.relative_to(ROOT))] = digest(p)
    report['ffmpeg'] = command(['ffmpeg','-version']).splitlines()[0]
    report['gpu'] = command(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    capture(parser.parse_args().output)
