# RGB Hand Retargeting and Wrist Anchored Arms

Research record and reproduction procedure, 2026-10-07. This covers the EgoDex
retargeting work migrated into SWAM, the methods we tried, the accepted RGB hand
replacement, object occlusion, bottle alignment refinement, and arm attachment.
It does not describe or modify SWAM world-model training, physical trajectory
collection, or the earlier indoor-scene generation project.

**Current result:** two short RGB-driven Sharpa hand-replacement previews;
object-visibility correction for both; an accepted alignment refinement and
wrist-anchored R1 arms for the bottle clip. These are counterfactual video
overlays, not reconstructed physical scenes or executable robot demonstrations.
The final arm stage leaves the accepted hands unchanged.

## Current Results

All paths below are relative to `${TINY_WAM_ROOT}`. Generated results are
private local files under `results/`, not included in this commit.

| Result | Directory | Status |
| --- | --- | --- |
| RGB-only hand estimates, both clips | `results/rgb_hands_wilor_v2/<clip>/` | Selected instead of EgoDex projected poses |
| Native-size robot hands, both clips | `results/rgb_robot_hands_v1/<clip>/` | Accepted visual starting point |
| Visible object occlusion, both clips | `results/rgb_robot_occlusion_v1/<clip>/` | Bottle, cap and chopstick masks; not 3D depth |
| Bottle silhouette refinement | `results/rgb_robot_bottle_alignment_v3/` | Latest accepted hand baseline; freeze this for arms |
| Bottle with arms | `results/rgb_robot_bottle_arms_v1/` | Latest delivered result |

The clip names are `use_chopsticks` and `screw_unscrew_bottle_cap`.

Final bottle files:

- `robot_arms.mp4`: full 960x540 frame, 33 frames at 15 fps, 2.2 seconds.
- `closeup.mp4`: display-only crop, 960x330, same timing.
- `comparison.mp4`: accepted hands versus added arms.
- `comparison_000.png`, `comparison_016.png`, `comparison_032.png`: stills.
- `arms.npz`: six arm joints and inferred arm-base transforms per side/frame.
- `arm_render.npz`: arm RGB and compositing masks.
- `report.json`: frame metrics, hard-constraint checks and input hashes.

The arm stage is implemented only for the accepted bottle clip. Do not describe
the chopsticks clip as having the same final silhouette/arm treatment.

## What Changed

| Stage | Implementation and observation | Decision |
| --- | --- | --- |
| Migration | Moved geometry, connected-hand fitting, batch plumbing and segmentation into `src/swam/retargeting/`; added SWAM entrypoints | Keep this work separate from training |
| Early connected fit | EgoDex targets, adjustable arm/hand sizes and wrist proportions; approximately 0.564 arm scale, 0.730 hand scale in one reviewed preview | Preserve as an archive, not current geometry |
| Joint segmentation | Compared SAM2 joint hand/object masks at prepared and native input sizes; tried Cutie propagation | Native-input SAM chosen for current local masks; no universal tracker win established |
| Published retargeting objectives | Ran Position, Vector, DexPilot and two author-provided objective settings on native R1/Sharpa geometry | Better annotation fit did not fix RGB alignment |
| Annotation audit | Reprojected the original EgoDex labels directly into the source RGB | Found misalignment before any robot fitting; changing the optimizer alone was insufficient |
| RGB re-estimation | WiLoR-mini and independent MediaPipe Hands on original RGB | WiLoR selected visually; hidden fingers remain uncertain |
| Robot replacement | Author objective initialization, then custom RGB reprojection refinement | Better visible placement, but no object occlusion yet |
| Object occlusion | Joint visible-instance labels for hands, bottle, cap and chopsticks | Restore observed object pixels only where the robot overlay covered them |
| Bottle alignment | Custom visible-silhouette fit, then temporal filtering of added corrections | User accepted v3; retain all earlier trials |
| Arms | Reverse native R1 kinematics with the accepted wrist transform fixed exactly | Fit arms without moving or refitting hands |

The historical batch route still contains the old scaled-fit settings. Its
successful completion checks concern coverage and file integrity, not grasp
quality. It is **not** the approved RGB pipeline and must not be used to silently
scale the latest method to hundreds of clips.

## Frameworks and Attribution

| Dependency | What we use | What we added |
| --- | --- | --- |
| [EgoDex](https://github.com/apple/ml-egodex) | Paired RGB, transforms, intrinsics and confidence fields | Extraction/alignment checks and diagnostic projections |
| [dex-retargeting](https://github.com/dexsuite/dex-retargeting) | Position, Vector and DexPilot implementations | R1/Sharpa URDF and landmark adapters |
| [Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation](https://github.com/Mingrui-Yu/retargeting) | Published optimizer; full setting and ablation option 2 | Robot configuration, joint weights and floating-wrist adaptation |
| [WiLoR](https://github.com/rolpotamias/WiLoR) through [WiLoR-mini](https://github.com/warmshao/WiLoR-mini) | RGB detector, MANO hand reconstruction and virtual camera estimate | Single-wearer detector settings, output serialization and robot fitting |
| MediaPipe Hands | Independent RGB landmark baseline | Comparison rendering; not blended into the selected target |
| [SAM2](https://github.com/facebookresearch/sam2) | SAM2.1 Hiera Large video segmentation | Joint hand/object prompts, exclusive labels, source-index checks and full-size outputs |
| [Cutie](https://github.com/hkchengrex/Cutie) | Alternative mask propagation from a labeled first frame | A comparison adapter; not used in the final output |
| [Sharpa assets](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml) | Vendor hand URDF, meshes and limits | Frame mapping and wrist mounting |
| [Galaxea URDF](https://github.com/userguide-galaxea/URDF) | Native R1 arm links and limits | Hard wrist-anchored reverse-chain fitting |
| Pinocchio, SciPy, PyTorch | FK/Jacobians, bounded optimizers and differentiation | Experiment objectives and validation |
| pyrender, trimesh, EGL, imageio, FFmpeg | Mesh loading, rasterization and media | Camera-aligned overlays and diagnostic videos |

Exact local upstream revisions, selected package versions, model checksums and
input hashes are recorded in [environment-20261007.json](environment-20261007.json).
The paper's project README points to [arXiv 2506.09384](https://arxiv.org/abs/2506.09384).
Our silhouette loss, pixel-refinement weights, prompt selection, temporal
correction and wrist-anchored arm fit are **custom engineering choices**, not
claimed reproductions of a paper's complete architecture or protocol.

WiLoR-mini is a simplified demo wrapper, not a full original-paper reproduction.
These previews do not use Isaac Sim physics, train SAM/WiLoR, or invoke a video
generation/inpainting model. The photographed environment is the original RGB.

## Data and Camera Contract

Sources on Remi:

```text
${TINY_WAM_ROOT}/results/retargeting_import_20261007/
  use_chopsticks/source/
  screw_unscrew_bottle_cap/source/
```

Each source contains `source.mp4`, `source.hdf5`, `prepare.json` and, for the
legacy route, prepared low-resolution arrays. Archive members are respectively
`test/use_chopsticks/0` and `test/screw_unscrew_bottle_cap/0`.

- Decode original frames `20, 22, ..., 84`, not video frames 0 through 32.
- The source is 1920x1080 at 30 fps; selection yields 33 frames at 15 fps.
- The crop is the entire image: `[0, 0, 1920, 1080]`.
- `prepare.json` has a historical 512x288 prepared size. SAM prompt coordinates
  are in that size and are scaled to the native image before inference.
- Current robot RGB/masks are 960x540. Resize original selected RGB using PIL
  LANCZOS; do not upscale the old 512x288 prepared images for these renders.
- Source indexing must agree across RGB, poses, segmentation and compositing.

Two distinct camera models appear in this history:

1. The annotation audit uses the EgoDex calibrated intrinsics and inverse
   camera-to-world transform to project annotation positions.
2. The accepted RGB retarget uses WiLoR's virtual intrinsics, with focal length
   37500 pixels and principal point `(960, 540)` at native resolution. Divide the
   first two rows of K by two for the 960x540 renderer.

WiLoR translation includes large virtual depths, approximately 12 to 22 in
these artifacts. They are **not measured physical distances in the room**. Do
not compare them directly with metric monocular scene depth or send them to a
robot. Camera coordinates use positive Z forward; pyrender's camera transform
is `diag(1, -1, -1, 1)`.

## RGB Hand Estimation

`scripts/reestimate_egodex_rgb.py` feeds RGB to WiLoR and MediaPipe. EgoDex points
are rendered as a diagnostic only; they do not supply model boxes or poses.

WiLoR uses float32. The first unfiltered experiment sometimes produced three
boxes, including duplicate opposite-handedness detections. The selected v2
sets `max_det=2`, `agnostic_nms=True`, `iou=0.45`, `conf=0.3` for these known
single-wearer clips. Both clips have one left and one right label in all selected
frames, which verifies the expected count, not anatomical accuracy.

MediaPipe uses static-image mode, complexity 1, up to two hands and detection
confidence 0.5. It sometimes misses the active right hand. No automatic ensemble
or additional hand-pose smoothing is applied at this stage.

`predictions.json` stores detector boxes, handedness, MANO joints/vertices,
virtual camera translation, projected joints, and separate MediaPipe/EgoDex
diagnostics. WiLoR's joint order is wrist, thumb, index, middle, ring, little.
The Sharpa mapping is `[0] + [5..20] + [1..4]`.

## Robot Hand Fitting

`scripts/render_rgb_robot_hands.py` uses native-size Sharpa meshes and limits,
with six virtual floating-wrist DOFs and 22 finger DOFs per hand. The author
retargeter with ablation option 2 initializes the solution while retaining
thumb-to-finger distances instead of contracting them.

The subsequent custom L-BFGS-B objective contains:

- Pixel reprojection with smooth L1 at an 8-native-pixel scale. Weights: tips
  4, distal direction bases 1.5, other joints 0.6, wrist 0.25.
- Distal direction loss, weight 0.35.
- Wrist-relative 3D shape loss, weight 0.05, normalized by 0.03.
- Joint-posture regularization, weight 0.002, and previous rotation/finger
  regularization, weight 0.005.
- Native finger limits and explicit virtual-root bounds; 250 iterations.

All 132 side/frame fits in the original two-clip pilot reached the iteration
budget rather than the convergence tolerance. They were finite bounded visual
solutions, not proven optima. Mean native-resolution fingertip residuals against
WiLoR were 1.62/1.98 pixels for chopsticks and 2.52/2.74 for bottle (left/right).
These are **pseudo-target fit residuals**, not errors against human ground truth.

`qpos.npz` uses `[translation XYZ, Euler XYZ, finger joints]`. Later alignment
outputs use a different format: `native_parameters.npz` contains
`[rotation vector XYZ, translation XYZ, finger joints]`. They are not interchangeable.

The renderer uses ambient `[0.65]*3`, a white directional light of intensity 2,
vendor hand material colors, metallic factor 0.15 and roughness 0.65. No human
pixels are removed outside the robot silhouette.

## Object Visibility

`src/swam/retargeting/segmentation.py` jointly tracks hand and object identities
with `non_overlap_masks=True`. Native-image inference is followed by bilinear
resizing of logits, then a positive-maximum label assignment at output size.
The current prompts are versioned in [configs](configs).

Bottle labels: 1 bottle, 2 left hand, 3 right hand, 4 cap. Chopsticks labels:
1 chopsticks, 2 left hand, 3 right hand. Cap corrections at later frames and
chopstick endpoint annotations are explicit manual prompts, not automatic
ground-truth labels. The sleeve tracker for arms has its own labels 1/2 and is
not substituted for the accepted bottle/cap segmentation.

Let R be the robot mask and O the observed object mask. Visible robot pixels are
`R & ~O`. At their intersection, restore original RGB object pixels. Outside
`R & O`, the overlay is unchanged. Joint hand/object segmentation reduces the
risk of putting the whole object on top of fingers actually visible in front.

This transfers **observed image visibility**. It does not infer hidden object
surfaces or establish which novel robot surface is physically in front. Source
hand shape and robot hand shape differ. Thin chopsticks, motion blur and contact
edges remain difficult. The local SAM build lacks the compiled hole-filling
extension; inference worked with that postprocessing step skipped.

Cutie was tried with the SAM first-frame labels. Its measured pilot time included
initial weight loading/download and is not a fair throughput comparison. It was
not selected as a demonstrated quality improvement over the native SAM result.

## Bottle Alignment Refinement

The first bottle frame exposed a weakness in fingertip-only reporting: all-joint
residuals against WiLoR were about 15.28 and 14.45 native pixels even when tips
matched within a few pixels. Robot base and anatomical human wrist landmarks
are also not equivalent. Matching estimated fingertips did not constrain the
visible palm adequately.

`scripts/refine_rgb_hand_alignment.py` adds a custom coordinate-descent stage:

- Match the visible SAM hand silhouette, excluding bottle/cap pixels and the
  forearm side of a wrist-based region.
- Penalize outside-mask distance, incomplete mask overlap and weak landmark
  disagreement; do not treat hidden fingertip estimates as strong observations.
- Keep native geometry and finger limits. Optimize wrist and fingers here only;
  these parameters are frozen once v3 is accepted.
- Objective: `20*(1-IoU) + clipped_outside_distance + 0.3*landmark_loss + 0.5*joint_prior`.
- V2 uses coordinate-descent factors `[2,2,1,1,1,0.5,0.5,0.25]`.
- V3 Gaussian-filters only the added parameter corrections with sigma 1 frame.
  The approved baseline motion is not itself filtered. Accept a proposal only
  if the combined per-frame objective improves.

Observed v3 mask overlap:

| Measure | Before | After |
| --- | ---: | ---: |
| First-frame left IoU | 0.6637 | 0.7344 |
| First-frame right IoU | 0.7352 | 0.8382 |
| Whole-clip mean left IoU | 0.7004 | 0.7697 |
| Whole-clip mean right IoU | 0.7296 | 0.7731 |

These are the same masks optimized against, not an independent evaluation.
Two right-hand frames regress in IoU despite an improved combined objective.
Temporal filtering reduces the extra variation introduced by independent fits,
but does not prove zero jitter: mean projected-joint acceleration changes from
5.90 to 6.27 left and 14.36 to 14.47 right pixels/frame squared. That measure also
contains true motion. Residual human pixels and imperfect grasps remain visible.

## Arms With Frozen Hands

`src/swam/retargeting/anchored_arm.py` anchors the terminal frame algebraically.
Given accepted hand transform H and constant flange-to-hand mount M:

```text
T_link6 = H @ inverse(M)
T_parent = T_child @ inverse(T_joint_origin @ R_joint(q))
```

The arm optimizer sees only six arm joint angles, never H or the finger angles.
Backward chaining infers the arm base. Forward chaining from that base must
recover H. Both R1 arms have the same joint-limit signs despite mirrored link
origins; an incorrect mirrored initialization was caught in the first static
pilot and corrected before the full sequence.

Native R1 dimensions are retained. The flange face is at x=0.03425, followed by
a 0.012 spacer and a +90-degree Y mount rotation. Sleeve masks supply a visible
forearm axis. Arm-only least-squares initialization and silhouette coordinate
descent fit the arm behind the wrist. Sleeve endpoints are temporally filtered;
arm steps are limited to 0.3 rad per selected frame. The source hand motion is
not changed to meet that arm constraint.

Rendering uses a proximal arm layer: exclude accepted hand pixels and observed
object pixels from the added arm mask. This explicitly preserves the approved
hand image, rather than claiming a complete depth-tested physical interaction.
The maximum round-trip wrist transform error was `3.55e-15`; maximum recovered
hand-joint position error `5.38e-15`, in virtual-camera units. Joint-limit excess
was zero. Accepted visible hand pixels were identical **before video encoding**.

Important limits:

- The two inferred arm bases can move independently. There is no common fixed
  torso or calibrated shoulder reconstruction.
- No self-collision, hand/arm penetration, object contact or dynamics validation.
- The accepted right wrist is near/below the image edge in some frames, so much
  of that forearm remains outside view. We do not move the hand to expose it.
- Original sleeves remain outside the robot coverage; no inpainting was done.

## Reproduce on Remi

### Existing accepted artifacts

This is the shortest path to repeat the final arm fitting without rerunning hand
estimation. Use a new output directory; do not delete or overwrite prior results.

```bash
cd "${TINY_WAM_ROOT:?Set TINY_WAM_ROOT to your checkout}"
COMPARE=${SWAM_RETARGET_ROOT}/compare-env/bin/python
export PYOPENGL_PLATFORM=egl
"$COMPARE" scripts/fit_arms_to_frozen_hands.py \
  --output results/rgb_robot_bottle_arms_repro_01
```

This requires the accepted v3 hand arrays, original prepared sources, accepted
object masks, `results/rgb_bottle_sleeves_v1/labels.npy`, and the original
`results/rgb_robot_hands_v1/screw_unscrew_bottle_cap/qpos.npz` camera intrinsics.
The script intentionally
uses those frozen directories. A different baseline requires an explicit code
or interface change, not silently replacing files at the accepted paths.

### Environment setup

Use the existing separate environments; do not pip-install over production
Isaac/learning environments:

```bash
CACHE=${SWAM_RETARGET_ROOT}
COMPARE="$CACHE/compare-env/bin/python"
RGB="$CACHE/rgb-env/bin/python"
SAM=${SWAM_SAM_ENV_ROOT}/bin/python
SAMROOT=${SWAM_SAM_ROOT}
VIDEO_DEPS=${SWAM_VIDEO_DEPS_ROOT}
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PWD/src:$CACHE/objective-retargeting/src${PYTHONPATH:+:$PYTHONPATH}"
```

Compare environment: Python 3.12, CPU PyTorch, Pinocchio, NLopt, pyrender/EGL.
RGB and SAM environments: Python 3.11 and CUDA PyTorch 2.7. The RGB environment
overlays NumPy 1.26.4, protobuf 4.25.8 and MediaPipe 0.10.21. SAM additionally uses
the separate `video-deps` path for dependencies such as iopath. CPU PyTorch in
the fitting environment does not mean rendering is CPU-only: EGL uses the GPU.

The environment JSON is a measured dependency snapshot, **not a tested
clean-machine installer**. On a new machine, recreate isolated environments,
check out the recorded upstream revisions, install their compatible dependencies,
obtain the permitted pretrained/MANO assets, and verify checksums before running.
Current pilots retain Remi-specific cache paths. An unmodified portable package
or automatic model download workflow is not promised.

Required local model locations:

```text
${SWAM_RETARGET_ROOT}/wilor-weights/pretrained_models/
${SWAM_SAM_ROOT}/sam2.1_hiera_large.pt
${SWAM_RETARGET_ROOT}/objective-retargeting/
${SWAM_RETARGET_ROOT}/dex-retargeting/
```

`ffmpeg`, `ffprobe`, NVIDIA/EGL libraries and DejaVu Sans are also used. Trusted
legacy WiLoR checkpoints require the script's weights-only compatibility setting;
do not use it for unknown downloaded pickle/checkpoint files.

### Full pipeline in a fresh local checkout

Several research entrypoints intentionally use fixed experiment paths. A fresh
local checkout is the simplest way to rerun every stage without overwriting the
accepted artifacts. Run this **after the documentation/code commit exists**:

```bash
ORIGINAL=${TINY_WAM_ROOT}
REPRO=${TINY_WAM_REPRO_ROOT}
test ! -e "$REPRO" || { printf 'Choose a new reproduction directory\n'; exit 1; }
git clone --no-hardlinks "$ORIGINAL" "$REPRO"
cd "$REPRO"

for clip in use_chopsticks screw_unscrew_bottle_cap; do
  src="$ORIGINAL/results/retargeting_import_20261007/$clip/source"
  dst="results/retargeting_import_20261007/$clip/source"
  mkdir -p "$dst"
  cp "$src/source.mp4" "$src/source.hdf5" "$src/prepare.json" "$dst/"
done
```

This clones from local disk only; it does not push or publish. Copying inputs is
not bypassing estimation: these are the original videos, paired annotations and
frame-selection metadata. Verify their SHA256 values against the environment
snapshot. Reset the environment variables above after changing to the new checkout
so `PYTHONPATH` starts with its `src` directory.

1. Estimate RGB hands for both clips:

```bash
"$RGB" scripts/reestimate_egodex_rgb.py \
  --output results/rgb_hands_wilor_v2 \
  --clips use_chopsticks screw_unscrew_bottle_cap --frames 33
```

2. Fit and render native robot hands:

```bash
"$COMPARE" scripts/render_rgb_robot_hands.py \
  --predictions results/rgb_hands_wilor_v2 \
  --output results/rgb_robot_hands_v1 \
  --no-legacy-reference
```

The new flag only disables the optional archived video comparison. Without it,
the script additionally requires `paper_actual_distance.mp4` under the two
`results/retarget_papers_*_v2` directories. It does not change the solver, hand
geometry or raw robot render.

3. Segment visible objects and composite both clips:

```bash
env PYTHONPATH="$VIDEO_DEPS${PYTHONPATH:+:$PYTHONPATH}" "$SAM" \
  scripts/occlude_rgb_robot_hands.py screw_unscrew_bottle_cap \
  --prompts docs/retargeting/configs/rgb_occlusion_bottle_prompts_v1.json \
  --output results/rgb_robot_occlusion_v1/screw_unscrew_bottle_cap

env PYTHONPATH="$VIDEO_DEPS${PYTHONPATH:+:$PYTHONPATH}" "$SAM" \
  scripts/occlude_rgb_robot_hands.py use_chopsticks \
  --prompts docs/retargeting/configs/rgb_occlusion_chopsticks_prompts_v1.json \
  --output results/rgb_robot_occlusion_v1/use_chopsticks
```

4. Refit the bottle silhouette, then smooth only the added corrections:

```bash
"$COMPARE" scripts/refine_rgb_hand_alignment.py \
  --output results/rgb_robot_bottle_alignment_v2 --all-frames \
  --sweeps 2 2 1 1 1 .5 .5 .25

"$COMPARE" scripts/refine_rgb_hand_alignment.py \
  --output results/rgb_robot_bottle_alignment_v3 --all-frames \
  --corrections-from results/rgb_robot_bottle_alignment_v2

"$COMPARE" scripts/audit_rgb_hand_alignment.py \
  results/rgb_robot_bottle_alignment_v3
```

5. Track sleeves independently; keep the approved object masks unchanged:

```bash
env PYTHONPATH="$PWD/src:$VIDEO_DEPS${PYTHONPATH:+:$PYTHONPATH}" "$SAM" - <<'PY'
from pathlib import Path
import os
from swam.retargeting.segmentation import track
track(
    Path('results/retargeting_import_20261007/screw_unscrew_bottle_cap/source'),
    Path('results/rgb_bottle_sleeves_v1'),
    Path('docs/retargeting/configs/rgb_bottle_sleeves_prompts_v1.json'),
    sam_root=os.environ['SWAM_SAM_ROOT'],
    native=True, output_size=(960, 540),
)
PY
```

6. Freeze the accepted hand arrays and fit arms:

```bash
"$COMPARE" scripts/fit_arms_to_frozen_hands.py \
  --output results/rgb_robot_bottle_arms_v1

ffmpeg -v error -i results/rgb_robot_bottle_arms_v1/robot_arms.mp4 \
  -vf 'crop=960:330:0:210' -c:v libx264 -crf 18 -pix_fmt yuv420p \
  results/rgb_robot_bottle_arms_v1/closeup.mp4

ffprobe -v error -show_entries stream=width,height,nb_frames,r_frame_rate \
  -show_entries format=duration -of json \
  results/rgb_robot_bottle_arms_v1/robot_arms.mp4
```

Expected timing is 33 frames / 15 fps / 2.2 seconds, not a long newly generated
trajectory. The original arm fit took about 0.4-0.7 seconds per selected frame;
the v2 silhouette fit took about 3.7 seconds/frame. These are local pilot timings,
not guaranteed throughput or end-to-end cold-start benchmarks. Full regeneration
can vary numerically with model/driver versions and is not promised bit-exact.

### Tests and acceptance

```bash
"$COMPARE" -m pytest -q \
  tests/test_retarget_occlusion.py \
  tests/test_retarget_paper_adapter.py \
  tests/test_rgb_robot_hands.py \
  tests/test_wrist_anchored_arm.py

"$RGB" -m pytest -q tests/test_rgb_hand_estimation.py
```

Check actual passed/skipped counts: optional imports can skip integration tests.
The compare suite has 16 tests; the RGB-estimation suite has 3. Inspect the actual
videos and source/mask comparisons in addition to tests. Required invariants:

- The original selected source frame indices remain unchanged.
- Native finger and arm limits are respected.
- FK adapters agree with their source URDF convention.
- Object compositing only changes robot/object intersections.
- Arm fitting leaves accepted hand files and visible hand pixels unchanged.
- Arm forward/backward round trips recover the locked wrist and hand joints.

These checks do not establish physical grasp quality or exact RGB pose accuracy.

### Documentation-time verification

The [verification record](verification-20261007.json) records checks run while
preparing this commit, not just the earlier research claims. From an independent
export of the staged code and assets, using the existing Remi environments:

- All 16 compare tests and 3 RGB tests passed, with no skipped tests.
- Repeated the full 33-frame bottle arm stage using frozen private inputs.
  Arm joints, inferred bases and masks matched the original arrays exactly;
  accepted hands stayed unchanged. Arm RGB differed in a tiny fraction of
  channels by at most one 8-bit intensity unit.
- Repeated the full 33-frame bottle hand solver with `--no-legacy-reference`.
  Joint parameters, intrinsics, indices and robot masks matched exactly. Raw
  RGB was not bit-identical: mean absolute difference was 0.00471 on the 0-255
  scale, with 0.0381 percent of channels differing and maximum difference 134.
  The cause of those sparse render differences was not isolated in this check.
- Confirmed the arm video is 960x540, 33 frames, 15 fps, 2.2 seconds.

Outputs are preserved locally in
`results/retarget_documentation_verification_20261007/`. This check reused frozen
WiLoR predictions, accepted masks and the existing environments; it did not
reinstall dependencies or rerun WiLoR/SAM from raw footage. The complete recipe
above is documented, but a clean-machine end-to-end reproduction was not tested.

## Reproduce Earlier Experiments

The archived inputs are already under `results/retargeting_import_20261007` on
Remi. Paper comparisons additionally need the archived `connected/parameters.npz`
and `connected/render.npz` beside each `source/`, plus its `source_frames.npy`.
Do not substitute the new floating-hand arrays into these old formats.

```bash
"$COMPARE" scripts/compare_egodex_retargeting.py --clip use_chopsticks \
  --output results/retarget_papers_chopsticks_repro_01 --frames 33
"$COMPARE" scripts/compare_egodex_retargeting.py --clip screw_unscrew_bottle_cap \
  --output results/retarget_papers_bottle_repro_01 --frames 33
"$COMPARE" scripts/inspect_retarget_annotations.py \
  results/retarget_papers_bottle_repro_01
```

The original chopsticks right-hand example improved annotation-space tip error
from 8.05 mm / 35.97 degrees (Position) to 0.59 mm / 3.70 degrees (author
actual-distance setting). The bottle right-hand joint-step p95 worsened from
0.146 to 0.203 rad. Neither result establishes RGB correctness or a universally
smoother method. The historical scaled comparison changes geometry too, so it
is not an isolated algorithm ablation.

To repeat the original SAM resolution comparison:

```bash
SRC=results/retargeting_import_20261007/screw_unscrew_bottle_cap/source
env PYTHONPATH="$VIDEO_DEPS${PYTHONPATH:+:$PYTHONPATH}" "$SAM" \
  scripts/retarget_egodex.py segment-objects --source "$SRC" \
  --prompts docs/retargeting/configs/retarget_bottle_prompts.json \
  --sam-root "$SAMROOT" --output results/sam512_repro_01
env PYTHONPATH="$VIDEO_DEPS${PYTHONPATH:+:$PYTHONPATH}" "$SAM" \
  scripts/retarget_egodex.py segment-objects --source "$SRC" \
  --prompts docs/retargeting/configs/retarget_bottle_prompts.json \
  --sam-root "$SAMROOT" --native-input --output results/sam_native_repro_01
```

Both commands above return the historical 512x288 output size; `--native-input`
changes input resolution only. The final occlusion wrapper explicitly requests
960x540. The historical prompt's `bottle_and_cap` name overstated its coverage;
the detached cap was not separately prompted. The current prompt fixes that.

```bash
env PYTHONPATH="$VIDEO_DEPS${PYTHONPATH:+:$PYTHONPATH}" "$SAM" \
  scripts/compare_retarget_trackers.py --source "$SRC" \
  --seed results/sam_native_repro_01 --output results/cutie_repro_01 \
  --cutie-root "$CACHE/Cutie" --checkpoint "$CACHE/cutie-base-mega.pth"
```

The maintained legacy CLI also exposes `prepare-paired`, `segment-egodex`,
`align-sharpa`, `align-arms`, `batch-egodex` and `composite`. Its `prepare-paired`
default crop is NOT the full-image crop used here; specify
`--crop 0,0,1920,1080 --indices 20:86:2` when rebuilding these particular inputs.
`batch-egodex` is historical and is deliberately not part of the main recipe.

## Code and Artifact Map

| File | Responsibility |
| --- | --- |
| `src/swam/retargeting/geometry.py` | Native Sharpa mesh/FK and legacy annotation fitting |
| `src/swam/retargeting/connected.py` | Historical joint arm/hand fit with optional scale fitting |
| `src/swam/retargeting/batch.py` | Historical manifest-based collection and resume checks |
| `src/swam/retargeting/segmentation.py` | SAM prompts, tracking, exclusive labels and output sizing |
| `src/swam/retargeting/occlusion.py` | Observed-visibility compositing and legacy comparison |
| `src/swam/retargeting/anchored_arm.py` | Hard wrist constraint and native reverse arm chain |
| `scripts/retarget_egodex.py` | Legacy CLI entrypoint |
| `scripts/compare_egodex_retargeting.py` | Published method comparison and URDF adapters |
| `scripts/inspect_retarget_annotations.py` | Annotation-to-RGB audit |
| `scripts/compare_retarget_trackers.py` | Cutie comparison |
| `scripts/reestimate_egodex_rgb.py` | WiLoR/MediaPipe inference |
| `scripts/render_rgb_robot_hands.py` | Pixel-refined robot hand replacement |
| `scripts/occlude_rgb_robot_hands.py` | Full-size object masks and overlays |
| `scripts/refine_rgb_hand_alignment.py` | Bottle visible-silhouette fitting and correction filtering |
| `scripts/audit_rgb_hand_alignment.py` | Mask/motion summaries and contact sheets |
| `scripts/fit_arms_to_frozen_hands.py` | Arm-only fitting against accepted bottle hands |
| `scripts/capture_retarget_environment.py` | Read-only dependency/input fingerprint capture |

The historical handoff JSON files are retained under [history](history). They
record decisions at their original stage; later acceptance in this document
supersedes their earlier status text without rewriting the history.

To capture a later environment without replacing this snapshot:

```bash
python3 scripts/capture_retarget_environment.py \
  --output results/retarget_environment_repro_01.json
```

## Privacy and Remaining Work

This is an internal local commit, not publication approval. Raw EgoDex footage,
generated videos, per-person MANO vertices, learned checkpoints and result arrays
are excluded from version control. Sharpa's supplied license/notice accompany
the vendor assets. R1's earlier source checkout had no root LICENSE file; the
existing README records that limitation. WiLoR, MANO, EgoDex, Ultralytics and
other components have separate terms. Check each asset's terms before releasing
code, weights, source footage or derivatives; this report does not clear them.

`assets/sharpa/` contains a vendored subset of the upstream revision recorded
in the environment snapshot: the existing `wave_01` URDF/mesh files and vendor
README, LICENSE and NOTICE. Its local nested Git metadata is not part of the
SWAM commit; reproduction does not require initializing a Sharpa submodule.

Open issues are visible human remnants, reliable contact/occlusion under changed
robot geometry, temporal hand estimation, independent visible-keypoint ground
truth, metric calibration and a common-torso physical IK solution. Do not fix
these by quietly moving the accepted hand baseline, changing scales, or calling
mask-fit metrics ground-truth pose accuracy.
