# Development

- Use `src/tinywam/tasks/manipulation/prediction/` for training, preparation and
  evaluation. Select experiments through `configs/`; reuse existing task code.
- Use the `tinywam` namespace and `TINYWAM_*` environment variables. Preserve
  historical dataset/run IDs and source snapshots.
- Preserve datasets, checkpoints, assets and run snapshots in their original
  locations. Put new outputs in `results/` and working notes in `notes/`.
- Experimental settings and dataset distinctions are in `docs/experiments.md`.
  Changes to sampling, dynamics, controllers, objectives or data distribution
  need explicit authorization and a separate experiment version.
- Check numerical outputs, sampling/RNG state, gradients and observation/action
  alignment when refactoring numerical code. Keep tests local and Git-ignored.
- Run full-duration collection checks before scaling. Inspect synchronized
  start/middle/end video and runtime simulator settings. Diagnose deterministic
  failures before retrying; keep failed data and validation thresholds.
- Report training and heldout results separately. Distinguish reconstruction,
  teacher forcing, open-loop prediction and closed-loop execution.
- Retain existing W&B runs and history. Use Results, Media and Diagnostics
  sections, with at most 4/2/4 panels respectively; collapse Diagnostics and
  disable automatic panel creation. Keep detailed logs locally and synchronize
  camera views into one media item. Back up layouts before editing.
- Keep configuration and source optimizer steps attached to metrics and media.
- Do not launch training, change README, commit or push without a request.
