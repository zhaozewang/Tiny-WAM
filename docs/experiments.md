# Experiment notes

## Reaching model

The model uses three token roles: image, proprioception (`qpos`, `qvel`) and
action. One camera is sampled per window and held fixed throughout it. Two
tasks are sampled equally: prediction of all modalities, and prediction of
proprioception/actions with only the initial image retained. Conditions are
2–3 noninitial joint-position goals with trial-relative timestamps; frames
and goals share the physical-time RoPE convention.

An action at time `t` is applied after observation `t` and affects observation
`t+1`. Recorded joint positions are not substitutes for commanded actions.
The early implementation used 10-second windows; the step-6,000 continuation
uses five-second windows (121 observations, 120 actions).

## Dataset versions

The original five-second collection source is
`results/physical_random_5s/collect.py`. Arm targets use the reference center
±0.15 rad, clipped to joint limits. Hand targets span the inner 80% of joint
ranges. Targets change every second with smoothstep interpolation and no
additional random holds. The full source defines the baseline.

`r1_random_10min_v5` is a different distribution: arm ranges of ±0.08–0.15 rad,
1–3 second target intervals, 15% holds and additional hand-range restrictions.
It is not an equivalent duration extension of the five-second collection.
The historical model uses this v5 dataset; corrected collections remain separate.
The older run `751aa2f0` also uses v5.

Longer episodes extend the same motion process without internal resets.
Changes to target timing/ranges, interpolation, controller gains, contact,
friction, joint limits or supervision define a new experimental condition.
Source snapshots, matched-duration motion statistics and synchronized videos
record these distinctions. The legacy five-token Gaussian model and its
checkpoints are retained as earlier experiments.

## Joint-limit correction

Seed 108 exceeded joint limits in both single- and 16-environment collection.
The `original_actions_solver64_v1` version raises position/velocity solver
iterations from 16/4 to 64/8. It retains target sequences, 120 Hz physics,
24 Hz control, gains, collision settings, joint limits and the 0.15 rad
rejection threshold.

Seeds 100–115 showed zero position-limit excess at every physics step,
including warmup. Collection continues to audit each physics step and reject
invalid batches without clipping states. Records are in
`results/limit_debug_single108`, `results/limit_fixed16_64_8` and
`results/limit_fixed16_audited`.

## Step-6,000 continuation

Run `zhaoze/swam/211886b7` resumes the complete step-6,000 checkpoint, including
optimizer state. Its settings are in `configs/train.toml`:

| Setting | Value |
| --- | --- |
| GPUs | 4 |
| Microbatch per GPU | 3 |
| Gradient accumulation | 3 |
| Global batch | 36 |
| Optimizer | AdamW8bit |
| Base learning rate | 5e-5 |
| Final step | 50,000 |
| Warmup | 200 global steps; not restarted on resume |
| Dataset | 68 v5 trajectories: 60 train, 4 validation, 4 test |

The learning rate was halved after a controlled loss-spike diagnosis. Other
model, tokenizer, task and data settings were retained. The separate ten-trajectory
512px / 240Hz collection is not included in this run.

The original distributed preflight covered both tasks with full optimizer
state and approximately 20.7 GiB peak allocation per GPU. Historical evaluation
ran every 200 updates with eight denoising steps, separate action plots and
three video rows: reference, teacher forcing and open-loop prediction. Training
loss and actual source optimizer steps accompany these records.
