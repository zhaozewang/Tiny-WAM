# Tiny-WAM working rules

This project lives on remi at `${TINY_WAM_ROOT}`. Preserve existing runs,
assets, checkpoints and shared media. Do not commit or push without an explicit
request. Store research notes in ignored `notes/` and generated outputs in
ignored `results/`. Model code belongs in `src/swam/`; experiment entrypoints in
`scripts/`. Training authorization and user corrections persist across turns.

## W&B: a small diagnostic dashboard, never a wall of metrics

- Reuse the existing project and ordinary run/project pages. Do not create new
  projects, duplicate runs, reports, or named saved-view URLs to organize a
  dashboard. Current SWAM run is in `zhaoze/swam`; preserve its history.
- Use exactly three dashboard sections: **Results**, **Media**, **Diagnostics**.
  Results and Media are open; Diagnostics is collapsed by default. Dashboard
  sections are not W&B run groups; run groups describe experiments/seeds.
- Keep the main dashboard to a few panels. Target at most four Results panels,
  two Media panels, and four Diagnostics panels. Do not create a panel for every
  split, camera, scalar, loss component, or coordinate.
- Combine semantically related curves with compatible units in a single plot.
  Put training/heldout and a meaningful baseline together. Use compact titles
  such as Loss, Prediction, Recovery, Gradients, and Update time.
- Combine three synchronized camera views into one media item/panel. Keep fit
  and heldout media in two panels; select prediction/recovery modes inside the
  panel. Label actual, tokenizer reconstruction, teacher-forced prediction,
  masked recovery, and open-loop rollout accurately. Do not call reconstruction
  or teacher forcing an open-loop prediction.
- Keep raw detailed scalar logs locally. Log only the curated diagnostic series
  to W&B for new runs. Existing detailed online history may remain hidden; do not
  delete scientific evidence just to reduce visible panels.
- Disable automatic panel generation on both ordinary project and run layouts.
  Metric prefixes alone do not organize a workspace. Update the actual saved
  layout and read it back to verify section names, panel counts and auto-panel
  settings before claiming cleanup is complete.
- When the user requests cleanup, fix the CURRENT run and project pages, not
  only the logging code for a future run. Do not restart training for a layout
  change. Back up layout specifications before editing.
- Put fixed configuration and parameter counts in config/summary. Keep the
  source optimizer step attached to metrics and videos. Detailed local records
  must remain available for diagnosis.

## Scientific reporting

- Separate resource probes using synthetic inputs from real-data fitting.
- Preserve observation/action alignment: action a_t is applied after observation
  t and causes observation t+1. Never fabricate actions from recorded qpos.
- Current flow-model design (2026-10-06): sample one camera per 10-second
  window and keep it fixed across all frames. Use three token roles: image,
  proprioception (qpos + qvel), action. The two equally sampled tasks predict
  all three modalities or only future proprioception/actions; the latter keeps
  the initial image and removes future image tokens. Retain the legacy five-token
  Gaussian model and its checkpoints for provenance. Goal conditions are 2–3
  noninitial joint-position waypoints, without velocity, with trial-relative
  timestamps. Use the same physical-time RoPE convention for frames and goals.
- Report train and heldout results separately and compare against persistence
  or cue-copy baselines. A falling NLL or four-trajectory fit does not establish
  general three-dimensional understanding.
- Do not hide changes to data, attention access, loss, or physical simulation in
  logging refactors. Keep actual videos, numerical evidence and run provenance.

## Long-running collection acceptance

- Before scaling a new collection protocol, finish at least one full-duration
  production batch, verify its shards and inspect actual start/middle/end video.
  A short smoke test or an active process does not establish that collection works.
- Treat deterministic data-validation failures as stop-and-diagnose errors; do not
  blindly rerun identical seeds. Preserve true error details in live status.
- Confirm simulator settings by reading authored/runtime values; USD instances
  may cause schema setters to skip collision parameters without raising errors.
- Keep validity thresholds independent of throughput goals. Record any policy,
  collision or dynamics changes in dataset metadata and use a new dataset version.
- Report complete verified trajectories separately from partial recorded blocks.
  Derive completion estimates from a full accepted batch, not startup throughput.


## Preserve user-approved experiment settings (2026-10-06)

- A setup the user has already tuned and accepted is an experimental baseline,
  not a suggestion. Requests to extend duration, generate more data, shard/cache
  data, move machines, parallelize, or improve throughput authorize those changes
  only. Preserve the baseline's data distribution and task semantics.
- Do not silently change action amplitudes, target ranges, target timing, hold
  probability, interpolation, reset behavior within a trajectory, joint limits,
  controllers, camera setup, contact/friction settings, reward, or supervision to
  make a job easier to run. A change described as conservative, robust, safer, or
  more stable is still a scientific change, not an implementation detail.
- Before scaling or starting downstream training, compare the actual generator
  and dataset provenance with the accepted baseline. Record the concrete config
  differences and verify matched-duration motion statistics and synchronized
  video. A successful run, finite values, or passing new thresholds does not
  establish equivalence to the accepted setup.
- If a genuine stability issue requires changing an accepted setting, report
  the observed failure, show the proposed differences and their consequences,
  and obtain the user's authorization for that scientific change unless already
  explicitly authorized in the conversation. Do not ask again for ordinary
  implementation work within the agreed settings. Do not silently weaken the
  experiment or its acceptance thresholds to get a pass.
- Extend the approved motion process continuously for longer episodes; do not
  replace it with a different random policy or add internal resets. Freeze and
  identify the baseline source/config before a large collection. Test the
  extension against that baseline before spending hours collecting or training.
- Specific SWAM baseline: `remi:${TINY_WAM_ROOT}/results/physical_random_5s/collect.py`. Its arm
  targets use center +/-0.15 rad clipped to limits, hand targets use the inner
  80 percent of joint ranges, targets change every 1 second with smoothstep
  interpolation, and no additional random hold probability is introduced.
  Preserve the full source, not just these abbreviated settings.
- Known deviation: r1_random_10min_v5 uses +/-0.08..0.15 rad arm ranges, 1..3
  second knots, 15 percent holds, and additional hand-range restrictions. The
  user rejected this unrequested drift. Do not describe v5 or training run
  751aa2f0 as an equivalent extension of the accepted 5-second experiment.
  Preserve existing evidence; do not silently overwrite it or mix corrected
  data into this run under the same dataset identity.
- When explaining unexpectedly weak motion, distinguish motion in source data
  from model behavior and preview age. An early optimizer-step preview cannot
  explain reduced movement in the original recordings. Verify before attributing
  the issue to insufficient training.

## Joint-limit fix (2026-10-06)

- User authorized fixing joint-limit instability after seed 108 failed both in
  a single environment and in a 16-environment batch. The batched collector now
  uses position/velocity solver iterations 64/8 (previously 16/4), recorded as
  `original_actions_solver64_v1`. This is an explicit physics-version change.
- Preserve original target sequences, 120 Hz physics, 24 Hz control, gains,
  collision settings, joint limits, and the 0.15 rad rejection threshold.
- The 16-seed regression (100–115) passed with zero position-limit excess at
  every physics step including warmup. This does not prove all future seeds safe.
  Keep per-physics-step auditing and reject invalid batches; do not clip states.
- Evidence: results/limit_debug_single108, results/limit_fixed16_64_8,
  results/limit_fixed16_audited. Preserve the original frozen collection contract
  and failed data as baseline evidence; do not relabel old runs as fixed.

## Approved four-GPU continuation on Jupyter (2026-10-06)

- User requested migrating the existing run from Earth to four Jupyter GPUs.
  Continue run `zhaoze/swam/211886b7` from the full step-6000 checkpoint with
  optimizer state preserved. Do not reinitialize or reset optimizer steps.
- User accepted halving the learning rate after the controlled spike diagnosis.
  Use base LR 5e-5 with the existing cosine schedule through step 50000; do not
  restart warmup. Preserve all other model, task, tokenizer and data settings.
- Jupyter devices 1,2,3,4; microbatch 3, accumulation 3, global batch 36.
  Entry points: `scripts/train_flow_distributed.py` and
  `scripts/supervise_jupyter_four.py`. Checkpoint metadata records this setup.
- Dataset remains the 68 fully encoded v5 trajectories (60 train, 4 validation,
  4 test). The new 512px / 240Hz ten-trajectory collection remains separate and
  is not silently added to this run.
- Keep per-step raw loss and asynchronous visualization every 200 optimizer
  updates. Callisto's evaluator uses `SWAM_TRAIN_HOST=jupyter`. Preserve the
  three approved video rows, 8-step denoising, and separate action plots.
- Four-GPU real-data preflight passed both tasks with full AdamW8bit state;
  measured peak allocation was approximately 20.7 GiB per GPU. This does not
  establish long-term convergence; check actual process and step progression.

## Project rename (2026-10-10)

- Canonical repository: `git@github.com:zhaozewang/Tiny-WAM.git`.
- Project directories are named `Tiny-WAM`; the former `swam` directory is a compatibility symlink.
- Preserve the Python import namespace `swam`, `SWAM_*` environment variables, existing W&B project/run IDs, and frozen dataset/provenance names. These are compatibility identities, not the repository display name.
