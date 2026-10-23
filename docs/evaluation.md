# Closed-loop evaluation

Run in the Isaac Sim / Isaac Lab Python environment with the frozen weights and
reference bundle from `docs/reproduce.md`. Prepare the assembled URDF and USD
using `docs/collection.md`; set `paths.scene_assets` to that output directory.
The evaluation scene reproduces the historical v5 dataset: 32/8 solver
iterations, hand damping 0.3, 1 mm contact offsets and three 256² cameras.
It is distinct from the corrected 512² collection scene.

```sh
tiny-wam evaluate --config configs/evaluate.toml
```

Paths resolve relative to the config, independently of the working directory.
Choose a new output directory for each run. The checkpoint's optimizer step and
joint ordering must match the config and scene. No evaluation starts on import
or when requesting CLI help.

The default conditions are recorded-action replay, holding the initial command,
feedback at 4/8/16 flow evaluations, an additional noise seed at 8 evaluations,
and an 8-evaluation run without goals. The config selects flow counts, seeds,
reference sample names and the `reference`, `cached` or `cuda_graph` backend.

Each trial uses 121 observations and 120 actions at 24 Hz. The initial command
comes from the recording. At time t, actual RGB, joint positions, velocities and
the committed action a_t enter the model. It predicts a_(t+1), then the simulator
executes a_t for five 120 Hz physics steps. Only afterward does a_(t+1) become the
current command. Generated images and proprioception are discarded. Recorded
future frames appear only in the comparison video; the supplied timed goals are
sampled using the training window routine.

Replay must match recorded joint positions within 0.02 rad RMSE before policy
results are accepted. Trials reject nonfinite state, joint-limit excess above
0.15 rad or speed above 100 rad/s. A failed replay stops the evaluation; other
physics failures are recorded against their trial.

Outputs contain the copied config, checkpoint/reference/source hashes,
per-trial trajectories, synchronized videos and a JSON summary. Metrics separate
goal-waypoint errors, final-pose errors and deviation from the recorded
trajectory. The final recorded pose need not be one of the supplied goals.
Timing includes image/state encoding, prediction and action decoding; it excludes
physics, rendering and CUDA-graph setup. This is synchronous simulation, not a
real-time control deadline test or a dataset-wide task-success estimate.

The evaluator reuses the v5 collector scene; its function matches the original
evaluation snapshot. CPU
checks cover action timing, metric equivalence and inference dispatch. Publishing
this entrypoint does not imply a new full simulator benchmark has been run.
