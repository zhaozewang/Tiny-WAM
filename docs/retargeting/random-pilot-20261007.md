# Random RGB Retargeting Pilot

## Decision

The pipeline produces complete outputs for all four randomly drawn samples after
adding explicit missing-hand handling. **It is not ready for unattended dataset
acceptance.** Detection continuity, wrist-to-arm appearance and some close-up
mesh artifacts still fail visual review. The phone clip is the best preview
candidate, not a physically validated demonstration.

This is a later experiment, not a replacement of the accepted bottle baseline.
No existing accepted poses, masks, videos or training settings were overwritten.

## Sampling and Coverage

- Archive: `${SWAM_DATA_ROOT}/egodex/test.zip`, resolving to `${SWAM_DATA_ROOT}/egodex/test.zip`.
- Sampling seed: `20261008`; 4 members drawn uniformly without replacement from
  3,243 paired RGB/HDF5 members. No quality filtering, forced favorite or replacements.
- For longer sources, a uniformly random 33-frame window at source stride 2.
  Two sources are shorter than the requested window: preserve the initially
  recorded preparation errors, then process their entire shorter sequence at
  the same stride. Do not pad, repeat frames or replace them with easier clips.
- Total: 120 processed frames at 15 fps, 8 seconds of actual selected footage.
  This is a short-window pilot, not full-dataset or long-sequence validation.

| Archive member | Source indices | Frames | Left/right detections | Visual finding |
| --- | --- | ---: | --- | --- |
| `test/declutter_desk/7` | 78..142, stride 2 | 33 | 32 / 22 | Cloth occlusion, blur and identity ambiguity interrupt the overlay |
| `test/assemble_disassemble_furniture_bench_drawer/11` | 0..54, stride 2 | 28 | 27 / 28 | Visible pointing left hand missed at selected frame 22; implausible proximal arm configurations |
| `test/load_dispense_ice/5` | 71..135, stride 2 | 33 | 0 / 33 | Genuine single-hand source; later close-up has dark mesh/render artifacts, cause not isolated |
| `test/charge_uncharge_device/16` | 0..50, stride 2 | 26 | 26 / 26 | Best preview candidate; fine contact and whole-arm plausibility remain unvalidated |

The desk clip has two left labels at selected frame 17. The conservative policy
does not arbitrarily choose one or relabel it as the right hand. Missing detections
are not all truly invisible hands: source frames were inspected separately.

## Implementation Changes

- Parameterized source, pose, segmentation and sleeve paths in the existing
  entrypoints. Original default bottle paths and fitting objectives are retained.
- Added `--allow-missing-hands` as an explicit alternative to strict two-hand
  requirements. It hides missing/ambiguous hands and saves NaN rows with
  `left_valid`/`right_valid` arrays. It never invents or interpolates a hidden pose.
- Refinement and frozen-wrist arm fitting respect validity. Temporal correction
  is filtered independently within contiguous observed runs, never across gaps.
- Missing forearm masks skip only the arm; they do not move a hand to manufacture
  a plausible attachment. Invalid poses must not be consumed as training targets.
- The random batch runner preserves stage commands, logs, failed attempts,
  source frame indices, archive CRCs and the fixed sampling manifest.
- Object IDs remain 1/4; hands 2/3; sleeve masks use their own left/right IDs 1/2.
  These explicit schemas are still an experimental constraint, not arbitrary
  instance-label support.

Mask prompts were manually selected for EACH new video, including thin cables,
cloth and separated drawer parts. They are archived under
`configs/random_pilot_20261007/`. This batch therefore tests pose/refinement/arm
generalization with prompted SAM2 masks, **not fully automatic segmentation**.

## Checks and Review

All 4 outputs have complete selected-frame coverage, valid native joint limits
on observed poses, and unchanged hand parameters/pixels through the arm stage
(pixels checked before video encoding). Arm joint-limit excess is zero; maximum
step between consecutive valid frames is 0.3 rad. The largest wrist round-trip
error is about `5.33e-15` in virtual-camera units.

The existing bottle arm regression was rerun. Its joint angles and inferred base
transforms match the prior accepted arrays exactly. Focused tests: 22 compare/
geometry/visibility tests plus 3 RGB-estimation tests passed, with no skips.

Visual review inspected source and final first/middle/last frames for all four
clips, all 12 anomalous detection frames outside the genuine single-hand clip,
and additional full-resolution views. This is frame inspection, not independently
annotated accuracy or a claim that every possible video defect was excluded.

Mask IoU improves on the optimized masks, but these are not independent labels.
In particular, numerical wrist connectivity does not prevent an arm from showing
an exposed proximal end or leaving the original sleeve visible. Missing poses
can cause human/robot alternation in the overlay. The current result is still
counterfactual RGB compositing, not executable physical robot trajectories.

## Local Results

Root: `${TINY_WAM_ROOT}/results/rgb_random_pilot_20261007/`.

- `manifest.json`: exact frozen draw and window selection, including short-clip recovery.
- `clips/<key>/status.json`: attempted stages, failures, commands and durations.
- `clips/<key>/arms/robot_arms.mp4`: native 960x540 final overlay.
- `clips/<key>/smooth/robot_hands.mp4`: hand-only result with object occlusion.
- `review/all_samples.mp4`: 1920x388, 120 frames, 15 fps, 8 seconds; all four
  samples in sampled order. Panels: source RGB, refined hands, frozen hands + arms.
- `review/<key>.mp4`: individual three-panel comparisons.
- `review/contact.png`: first/middle/last final frames for every sample.
- `detection_gaps.png`: original RGB for the 12 ambiguous/missing detection frames.
- `review/verification.json`: processing checks; separate from visual verdicts.
- `review/visual_review.json`: manual findings and batch acceptance decision.
- `review/source_snapshot/`: experiment code; `review/working_diff.patch` records
  scoped tracked changes. Raw footage, predictions and generated media stay local.

## Reproduction

Use the isolated environments described in the main retargeting guide. They are
still local Remi dependencies; this is not a clean-machine installer. The commands
below replay THIS draw with its manually prepared prompts, not a new random seed:

```bash
cd "${TINY_WAM_ROOT:?Set TINY_WAM_ROOT to your checkout}"
P=${SWAM_RETARGET_ROOT}/compare-env/bin/python
RUN=results/rgb_random_repro_01
"$P" scripts/sample_rgb_retarget_batch.py --output "$RUN" --stage plan \
  --seed 20261008 --count 4 --frames 33
"$P" scripts/sample_rgb_retarget_batch.py --output "$RUN" --stage prepare-short
for case in "$RUN"/clips/*; do
  key=$(basename "$case")
  cp docs/retargeting/configs/random_pilot_20261007/"$key"/*_prompts.json "$case/"
done
for stage in estimate hands_visible segment sleeves refine smooth arms; do
  "$P" scripts/sample_rgb_retarget_batch.py --output "$RUN" --stage "$stage"
done
"$P" scripts/review_rgb_retarget_batch.py --output "$RUN"
```

Stage completion is not a visual quality gate. Failed attempts are deliberately
not automatically rerun or replaced; inspect their logs and use a new explicit
experiment/version to change the method. Do not reuse these prompt coordinates
for a new seed or a different source window.

Before scaling: resolve temporal detection/handedness through occlusion, establish
plausible proximal-arm constraints while keeping the wrist fixed, and isolate the
close-up mesh artifact. Keep a new random held-out batch for evaluating those
changes rather than treating these four now-inspected samples as unseen tests.
