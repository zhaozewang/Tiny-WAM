"""Recover missed detections in a frozen pilot using its manually seeded SAM tracks."""
from swam.paths import external_path
import argparse
import json
from pathlib import Path
import shutil
import sys
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / 'manifest.json', output / 'manifest.json')
    manifest = json.loads((output / 'manifest.json').read_text())
    for item in manifest['clips']:
        key = item['key']
        case = output / 'clips' / key
        case.mkdir(parents=True)
        for name in ('source', 'sleeves'):
            (case / name).symlink_to(source / 'clips' / key / name, target_is_directory=True)
        if 'load_dispense_ice' not in key:
            (case / 'segment').symlink_to(source / 'clips' / key / 'segment', target_is_directory=True)
        else:
            prompt = json.loads((source / 'clips' / key / 'segment_prompts.json').read_text())
            # Correct a bad positive click on skin in the last frame. Coordinates
            # below were annotated on the 960x540 source RGB, not robot renders.
            annotation = prompt['objects'][0]['annotations'][-1]
            annotation['points'] = [[x * 512 / 960, y * 288 / 540] for x, y in
                                    [(463, 280), (463, 325), (525, 395), (555, 355), (450, 405)]]
            annotation['labels'] = [1, 1, 0, 0, 0]
            prompt['repair'] = 'Last-frame handle prompt was on skin; reannotated source RGB. No blanket mask erosion.'
            (case / 'segment_prompts.json').write_text(json.dumps(prompt, indent=2) + '\n')
    (output / 'repair_provenance.json').write_text(json.dumps(dict(
        original=str(source), selection_unchanged=True,
        policy='Keep existing unambiguous estimates; rerun WiLoR on observed tracked ROIs for missing/duplicate identities.'), indent=2))


def recover(source, output, predictions_root=None, reconcile_identities=False, tracking_v3=False):
    # Reuse trusted-checkpoint compatibility and exactly the original pose model.
    from reestimate_egodex_rgb import serializable, torch, cv2, imageio, np
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    from swam.retargeting.tracked_roi import mask_box, select_tracked_detection, reconcile_tracked_identities
    torch.set_num_threads(4)
    pipeline = WiLorHandPose3dEstimationPipeline(device=torch.device('cuda'), dtype=torch.float32,
        verbose=False, wilor_pretrained_dir=str(Path(external_path('retarget', 'wilor-weights'))))
    manifest = json.loads((output / 'manifest.json').read_text())
    for item in manifest['clips']:
        key = item['key']; case = output / 'clips' / key
        original = (predictions_root or source / 'predictions') / key
        records = json.loads((original / 'predictions.json').read_text())
        summary = json.loads((original / 'summary.json').read_text())
        labels = np.load(case / 'segment/labels.npy')
        metadata = json.loads((case / 'segment/segmentation.json').read_text())
        if metadata['source_indices'] != [r['source_frame'] for r in records]:
            raise ValueError('Mask/pose frame ordering mismatch')
        edits = []; identity_events = []; previous_boxes = {}
        with imageio.get_reader(case / 'source/source.mp4') as reader:
            for i, record in enumerate(records):
                updated = []; rgb = None
                detections = record['wilor']
                rejected = []
                if reconcile_identities:
                    detections, rejected = reconcile_tracked_identities(
                        detections, labels[i], summary['input_resolution'],previous_boxes,
                        require_observed_track=tracking_v3)
                    identity_events.extend(dict(frame=i, **event) for event in rejected)
                    assigned = {event['detection']:event['tracked_side'] for event in rejected}
                    boxes = {side:[] for side in (0,1)}
                    for index, prediction in enumerate(record['wilor']):
                        assigned_side = assigned.get(index,int(prediction['is_right']))
                        if assigned_side is not None:
                            boxes[assigned_side].append(prediction['hand_bbox'])
                    previous_boxes = {side:values[0] for side,values in boxes.items() if len(values)==1}
                for side, identity in [('left', 2), ('right', 3)]:
                    is_right = int(side == 'right')
                    candidates = [p for p in detections if int(p['is_right']) == is_right]
                    if len(candidates) == 1:
                        updated.append(candidates[0]); continue
                    box = mask_box(labels[i] == identity, summary['input_resolution'])
                    if box is None:
                        # No observed track: keep the original absence/ambiguity.
                        updated.extend(candidates); continue
                    selected = select_tracked_detection(candidates, box)
                    event = dict(frame=i, side=side, old_count=len(candidates), box=box.tolist())
                    if selected is not None:
                        updated.append(selected); event['method'] = 'identity_selected_by_track_overlap'
                    else:
                        if tracking_v3:
                            corrected = [record['wilor'][e['detection']] for e in rejected
                                         if e['tracked_side'] == is_right]
                            original_crop = select_tracked_detection(corrected, box)
                            if original_crop is not None:
                                box = np.asarray(original_crop['hand_bbox'], dtype=float)
                                event.update(box=box.tolist(), crop_source='identity_corrected_detector_box')
                        if rgb is None:
                            rgb = reader.get_data(record['source_frame'])
                        prediction = pipeline.predict_with_bboxes(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            bboxes=np.array([box], dtype=np.float32), is_rights=np.array([is_right]))
                        if len(prediction) != 1:
                            raise RuntimeError('ROI inference did not return one pose')
                        values = prediction[0]['wilor_preds']
                        if not all(np.isfinite(values[n]).all() for n in
                                   ('pred_keypoints_3d', 'pred_keypoints_2d', 'pred_cam_t_full')):
                            raise RuntimeError('Nonfinite recovered pose')
                        updated.extend(serializable(prediction))
                        event['method'] = ('WiLoR_RGB_inference_on_identity_corrected_detector_box'
                                           if event.get('crop_source') else 'WiLoR_RGB_inference_on_tracked_mask_box')
                    w, h = summary['input_resolution']
                    event['track_touches_image_edge'] = bool(box[0] <= 0 or box[1] <= 0 or box[2] >= w or box[3] >= h)
                    edits.append(event)
                record['wilor'] = updated
        destination = output / 'predictions' / key
        destination.mkdir(parents=True)
        (destination / 'predictions.json').write_text(json.dumps(records))
        summary.update(source=str(case / 'source'), recovery=edits,
            identity_reconciliation=identity_events, reconcile_identities=reconcile_identities,
            tracking_v3=tracking_v3,
            recovery_limitations=['Manual SAM identity seeds remain required.',
                                 'Hidden or clipped fingers are model predictions, not observed ground truth.'])
        (destination / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(key, json.dumps(edits), flush=True)


def merge_accepted(old, candidate, output, camera):
    """Keep all previously accepted poses; add only newly observed valid hands."""
    os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    import numpy as np
    import pyrender
    from swam.retargeting.geometry import SharpaHand
    output.mkdir(parents=True,exist_ok=False)
    previous=np.load(old/'native_parameters.npz');new=np.load(candidate/'native_parameters.npz')
    merged={name:new[name].copy() for name in new.files}
    retained={}
    for side in ('left','right'):
        keep=previous[side+'_valid']
        merged[side][keep]=previous[side][keep]
        merged[side+'_valid'] |= keep
        np.testing.assert_array_equal(merged[side][keep],previous[side][keep])
        retained[side]=int(keep.sum())
    k=np.load(camera/'qpos.npz')['intrinsic'].copy();k[:2]/=2
    scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.65]*3)
    hands={s:SharpaHand(s,scene) for s in ('left','right')}
    scene.add(pyrender.IntrinsicsCamera(k[0,0],k[1,1],k[0,2],k[1,2],znear=.01,zfar=100),pose=np.diag([1.,-1.,-1.,1.]))
    scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=np.diag([1.,-1.,-1.,1.]))
    renderer=pyrender.OffscreenRenderer(960,540);rgbs=[];masks=[]
    for i in merged['indices']:
        for side,hand in hands.items():
            valid=bool(merged[side+'_valid'][i])
            for node in hand.nodes.values():node.mesh.is_visible=valid
            if valid:hand.show(scene,merged[side][i])
        rgb,depth=renderer.render(scene);rgbs.append(rgb);masks.append(depth>0)
    renderer.delete()
    np.savez_compressed(output/'native_parameters.npz',**merged)
    np.savez_compressed(output/'robot_render.npz',rgb=rgbs,mask=masks,indices=merged['indices'])
    (output/'report.json').write_text(json.dumps(dict(
        original=str(old),candidate=str(candidate),preserved_valid_poses=retained,
        policy='Original accepted parameters retained exactly; only formerly invalid hands added.'),indent=2))


def process(source, output):
    from sample_rgb_retarget_batch import COMPARE, CACHE, command_for
    environment=dict(os.environ,PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     PYTHONPATH=os.pathsep.join(map(str,[ROOT/'src',CACHE/'objective-retargeting/src'])))
    manifest=json.loads((output/'manifest.json').read_text())
    for item in manifest['clips']:
        key=item['key'];case=output/'clips'/key
        summary=json.loads((output/'predictions'/key/'summary.json').read_text())
        alias=output/'hands'/key;alias.parent.mkdir(exist_ok=True)
        def execute(stage,command):
            command=list(map(str,command))
            (case/(stage+'_command.json')).write_text(json.dumps(command,indent=2))
            with (case/(stage+'.log')).open('x') as log:
                subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(key,stage,'complete',flush=True)
        if summary['recovery']:
            execute('hands_visible',command_for('hands_visible',output,item))
            alias.symlink_to(case/'hand_fit_visible'/key,target_is_directory=True)
            execute('refine',command_for('refine',output,item))
            cmd=command_for('smooth',output,item)
            cmd[cmd.index('--output')+1]=case/'smooth_candidate'
            execute('smooth_candidate',cmd)
            merge_accepted(source/'clips'/key/'smooth',case/'smooth_candidate',case/'smooth',alias)
        else:
            alias.symlink_to(source/'hands'/key,target_is_directory=True)
            (case/'smooth').symlink_to(source/'clips'/key/'smooth',target_is_directory=True)
        execute('arms',command_for('arms',output,item)+['--proximal-exit'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'recover', 'process'], required=True)
    args = parser.parse_args()
    {'prepare':prepare,'recover':recover,'process':process}[args.stage](args.source.resolve(), args.output.resolve())
