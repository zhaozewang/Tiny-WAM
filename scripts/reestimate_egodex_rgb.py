"""Independent RGB-only hand estimation, before robot retargeting or smoothing."""
from tinywam.paths import external_path
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

# The published trusted WiLoR/YOLO checkpoints predate torch's weights-only default.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(external_path('retarget', 'yolo-config'))))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('MPLCONFIGDIR', str(Path(external_path('retarget', 'matplotlib'))))
import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import torch

# chumpy is used only to deserialize the released MANO data, not for inference.
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
for name, value in [('bool', bool), ('int', int), ('float', float), ('complex', complex),
                    ('object', object), ('unicode', str), ('str', str)]:
    if name not in np.__dict__:
        setattr(np, name, value)

ROOT = Path(__file__).resolve().parents[1]
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
          [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
COLORS = [(255, 65, 65), (45, 200, 255), (65, 240, 100), (255, 175, 45), (220, 90, 255)]


def draw_joints(frame, hands):
    frame = frame.copy()
    for points in hands:
        points = np.asarray(points)
        if points.shape != (21, 2) or not np.isfinite(points).all():
            continue
        for chain, color in zip(CHAINS, COLORS):
            uv = np.round(points[chain]).astype(np.int32)
            cv2.polylines(frame, [uv], False, color, 3, cv2.LINE_AA)
            for x, y in uv[1:]:
                cv2.circle(frame, (int(x), int(y)), 5, color, 2, cv2.LINE_AA)
    return frame


def egodex_points(data, index):
    camera = np.linalg.inv(data['transforms/camera'][index])
    k = data['camera/intrinsic'][:]
    hands = []
    for side in ('left', 'right'):
        names = [side+'Hand']
        for finger in ['Thumb', 'IndexFinger', 'MiddleFinger', 'RingFinger', 'LittleFinger']:
            names += [side+finger+p for p in ['Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip']]
        p = np.stack([(camera @ data['transforms/'+n][index])[:3, 3] for n in names])
        uv = p @ k.T
        hands.append(uv[:, :2]/uv[:, 2:])
    return hands


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


def run_clip(pipeline, mp, clip, output, limit, source=None):
    source = source or ROOT/'results/retargeting_import_20261007'/clip/'source'
    config = json.loads((source/'prepare.json').read_text())
    indices = config['indices'][:limit]
    output.mkdir(parents=True, exist_ok=False)
    reader = imageio.get_reader(source/'source.mp4')
    records, frames, overlays = [], [], []
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1,
                                 min_detection_confidence=.5) as hands, h5py.File(source/'source.hdf5') as data:
        for i, index in enumerate(indices):
            rgb = reader.get_data(index)
            torch.cuda.synchronize()
            start = time.perf_counter()
            predictions = pipeline.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), hand_conf=.3)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            start = time.perf_counter()
            landmarks = hands.process(rgb)
            mp_time = time.perf_counter()-start
            mp_hands = []
            if landmarks.multi_hand_landmarks:
                for hand, world, handedness in zip(landmarks.multi_hand_landmarks, landmarks.multi_hand_world_landmarks,
                                                    landmarks.multi_handedness):
                    cls = handedness.classification[0]
                    mp_hands.append(dict(keypoints_2d=np.array([[p.x*rgb.shape[1], p.y*rgb.shape[0]] for p in hand.landmark]),
                        normalized_landmarks=np.array([[p.x, p.y, p.z] for p in hand.landmark]),
                        world_landmarks=np.array([[p.x, p.y, p.z] for p in world.landmark]),
                        raw_selfie_handedness=cls.label, classification_score=cls.score))
            ego = egodex_points(data, index)
            predicted_uv = [p['wilor_preds']['pred_keypoints_2d'][0] for p in predictions]
            mp_uv = [p['keypoints_2d'] for p in mp_hands]
            record = dict(source_frame=index, wilor_seconds=elapsed, mediapipe_seconds=mp_time,
                          wilor=predictions, mediapipe=mp_hands, egodex_2d=ego)
            # MANO vertices stay in the local private artifact; not published or added to git.
            records.append(serializable(record))
            panels = [draw_joints(rgb, ego), draw_joints(rgb, predicted_uv), draw_joints(rgb, mp_uv)]
            frames.append(rgb)
            overlays.append(panels)
            if i % 8 == 0:
                print(clip, i+1, '/', len(indices), 'hands', len(predictions), len(mp_hands),
                      'WiLoR seconds', round(elapsed, 3), flush=True)
    reader.close()
    (output/'predictions.json').write_text(json.dumps(records))
    summary = dict(clip=clip, frames=len(indices), source_frame_indices=indices, fps=config['fps'],
        source=str(source), input_resolution=[frames[0].shape[1], frames[0].shape[0]],
        input='RGB only; EgoDex annotations used solely for the comparison panel', smoothing=False,
        wilor_wrapper='WiLoR-mini, linked by official WiLoR README as demo-only simplified implementation',
        precision='float32', focal='Unmodified WiLoR default virtual focal; NOT calibrated metric depth',
        detector=dict(max_det=2, agnostic_nms=True, iou=.45, conf=.3,
                      rationale='Known single-wearer clips; retain at most two highest-scored hands. Does not guarantee correct identity.'),
        mediapipe='static-image mode, complexity 1, max 2 hands, detection confidence .5',
        median_wilor_seconds=float(np.median([r['wilor_seconds'] for r in records[1:] or records])),
        wilor_detection_counts=[len(r['wilor']) for r in records],
        mediapipe_detection_counts=[len(r['mediapipe']) for r in records],
        limitations=['No hand-object contact validation', 'No metric 3D ground truth',
                     'Monocular absolute depth and scale remain ambiguous', 'No temporal identity tracking or smoothing',
                     'Visible alignment must be checked against RGB, not EgoDex pseudo-ground-truth'],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        weights=str(Path(external_path('retarget', 'wilor-weights'))),
        license_scope='Local research only; WiLoR, MANO, Ultralytics and EgoDex have separate licenses; no redistribution')
    for name in ['WiLoR', 'WiLoR-mini']:
        summary[name+'_commit'] = subprocess.check_output(['git', '-C', str(Path(external_path('retarget'))/name),
                                                         'rev-parse', 'HEAD'], text=True).strip()
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    labels = ['EgoDex projection', 'WiLoR: RGB only', 'MediaPipe: RGB only']
    with imageio.get_writer(output/'comparison.mp4', fps=config['fps'], macro_block_size=1, quality=8) as writer:
        for i, panels in enumerate(overlays):
            canvas = np.full((400, 1920, 3), 245, dtype=np.uint8)
            for j, panel in enumerate(panels):
                canvas[40:, j*640:(j+1)*640] = cv2.resize(panel, (640, 360), interpolation=cv2.INTER_AREA)
                cv2.putText(canvas, labels[j], (j*640+12, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (25,25,25), 1, cv2.LINE_AA)
            writer.append_data(canvas)
            if i in [0, len(indices)//2, len(indices)-1]:
                imageio.imwrite(output/f'comparison_{i:03d}.png', canvas)
                imageio.imwrite(output/f'wilor_{i:03d}.png', panels[1])
    for j, method in enumerate(['egodex', 'wilor', 'mediapipe']):
        imageio.mimwrite(output/(method+'.mp4'), [cv2.resize(p[j], (1280,720)) for p in overlays],
                        fps=config['fps'], macro_block_size=1, quality=8)
    # One fixed crop for the active right hand, derived from RGB detector boxes only.
    boxes = [p['hand_bbox'] for r in records for p in r['wilor'] if p['is_right'] == 1]
    if boxes:
        boxes = np.asarray(boxes)
        x0, y0 = np.maximum(boxes[:, :2].min(0)-30, [0,0]).astype(int)
        x1, y1 = np.minimum(boxes[:, 2:].max(0)+30, [frames[0].shape[1],frames[0].shape[0]]).astype(int)
        w, h = x1-x0, y1-y0
        scale = min(600/w, 600/h)
        cw, ch = int(w*scale), int(h*scale)
        with imageio.get_writer(output/'right_hand_detail.mp4', fps=config['fps'], macro_block_size=1, quality=8) as writer:
            for i, panels in enumerate(overlays):
                canvas = np.full((640, 1800, 3), 245, dtype=np.uint8)
                for j, panel in enumerate(panels):
                    x, y = j*600+(600-cw)//2, 40+(600-ch)//2
                    canvas[y:y+ch, x:x+cw] = cv2.resize(panel[y0:y1,x0:x1], (cw,ch))
                    cv2.putText(canvas, labels[j], (j*600+12, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (25,25,25), 1, cv2.LINE_AA)
                writer.append_data(canvas)
                if i in [0, len(indices)//2, len(indices)-1]:
                    imageio.imwrite(output/f'detail_{i:03d}.png', canvas)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clips', nargs='+', default=['use_chopsticks', 'screw_unscrew_bottle_cap'])
    parser.add_argument('--frames', type=int, default=33)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import mediapipe as mp
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    torch.set_num_threads(4)
    pipeline = WiLorHandPose3dEstimationPipeline(device=torch.device('cuda'), dtype=torch.float32, verbose=False,
        wilor_pretrained_dir=str(Path(external_path('retarget', 'wilor-weights'))))
    pipeline.hand_detector.overrides.update(max_det=2, agnostic_nms=True, iou=.45)
    for clip in args.clips:
        source = args.source_root/clip/'source' if args.source_root else None
        run_clip(pipeline, mp, clip, args.output/clip, args.frames, source)


if __name__ == '__main__':
    main()
