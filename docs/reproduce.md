# Reproducing Tiny-WAM

The supported entrypoint is `tiny-wam`. It runs locally, reads explicit TOML
configuration, and does not require SSH aliases, a W&B account, or a separate
video-rendering worker. Run commands from the checkout after installation;
configuration paths resolve relative to the configuration file, not the shell's
working directory.

## Installation

Use Linux x86-64, Python 3.11, and an NVIDIA GPU supported by CUDA 12.8. The
inference experiment targets one RTX 4090 with 24 GB VRAM. Install FFmpeg for raw
video preparation; inference on the supplied reference does not require a simulator.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
```

`requirements.lock` pins transitive dependencies, including the CUDA build of
PyTorch. The `reproduce` and `train` extras in `pyproject.toml` describe direct
dependencies; the lock is the tested Linux installation recipe. The simulator
and RGB retargeting tools have separate dependencies and are not installed by
this recipe.

## Artifacts

Code alone is not a pretrained experiment. Obtain the existing frozen artifact
bundle and place its contents in `artifacts/`:

```text
artifacts/
  model.pt
  eval_reference.pt
  tokenizers/
    action.pt
    proprio.pt
  dcae/
    config.json
    diffusion_pytorch_model.safetensors
```

`configs/artifacts.json` records exact sizes and SHA-256 hashes. It identifies
step 50,000 and the historical v5 dataset. No public download URL is assigned
currently; this repository does not silently download a substitute checkpoint.
The bundle is approximately 7 GB. A local owner can create it from configured
source files without embedding their directory names:

```bash
tiny-wam export --config /path/to/private-inputs.toml --output /path/to/bundle
```

Use only trusted PyTorch artifacts: these historical checkpoints contain Python
objects and are loaded with `weights_only=False`.

## One-command inference reproduction

```bash
tiny-wam reproduce --config configs/reproduce.toml
```

This verifies the frozen inputs, measures the reference, KV-cache and CUDA-graph
implementations, and produces `results/reproduce/summary.csv`, `summary.json`,
`summary.png`, `summary.pdf`, and the individual timings in
`benchmark/results.json`. It evaluates 1, 2, 4 and 8 flow steps at 1, 24, 60 and
119 history frames on one recorded training sample and one validation sample.
Each setting has five warmups and 25 measured ticks. The seed, camera sampling,
goal sampling and Euler update are unchanged from the original benchmark.

Latency includes RGB transfer, input encoding, history update, sampling, action
decoding and transfer back to CPU. It excludes sensing, simulation, file I/O,
checkpoint loading, initial cache fill and graph capture; the latter two are
reported separately. Numerical differences compare decoded joint commands with
the reference implementation. This is a recorded-history latency and numerical
agreement experiment, **not** a closed-loop task-success benchmark. Timing
varies with the GPU, software stack and concurrent load.

Use a new `paths.output` for a repeat; existing runs are never overwritten.
To inspect the setup without running inference:

```bash
tiny-wam doctor --config configs/reproduce.toml
# Verify artifacts on a machine without a GPU:
tiny-wam doctor --config configs/reproduce.toml --cpu
```

## Data preparation and training

These are separate explicit commands; inference reproduction never starts training.
The historical model used v5 data. That collection differs from the later
approved collection contract. Preserve that identity when reproducing the old
model; do not regenerate a different distribution and call it v5.

For existing latents, supply `data/flow_latents_v5/index.json`,
`normalization.pt`, and the referenced `episode_*/block_*.pt` files. The index
owns split membership, episode identities and published block counts.
`configs/training-data.json` freezes all 68 episodes and 8,160 blocks for the
approved continuation; the training preflight checks this identity and every
published block before launching workers. For raw
v5 shards, supply `data/raw/r1_random_10min_v5/index.json` and `shards/`, then run:

```bash
tiny-wam prepare --config configs/prepare.toml
```

Preparation verifies shard checksums and joint order. It uses the frozen DC-AE,
state/action VAEs and normalization, preserving 24 Hz, the three camera views,
and the 121-observation/120-command alignment. Lossless MP4 shards must carry
verified RGB provenance. No resizing, tokenizer retraining, synthetic replacement
data or simulator modification happens implicitly. Preparation writes a new
latent directory and fails if it already exists.

`configs/train.toml` describes the approved continuation from step 6,000 to
50,000: four GPUs, microbatch 3, accumulation 3, global batch 36, AdamW8bit,
base LR 5e-5, the original cosine schedule and 200-step warmup indexed by global
step. Supply a **full** optimizer checkpoint at
`artifacts/training/step_006000.pt`, then:

```bash
tiny-wam doctor --config configs/train.toml --training
tiny-wam train --config configs/train.toml
```

The launcher handles `torchrun`. It saves per-rank metrics and an atomic
`latest.pt` every 200 steps. Optional W&B settings are in the config; logging is
disabled by default. There is no wait for another machine to render videos.
To resume a new interrupted run, set `paths.checkpoint` to its `latest.pt`,
`train.expected_step` to that checkpoint's step, and choose a new output folder.
Preserve `train.steps=50000` so the learning-rate schedule does not restart.

For a fresh scientific run, omit `paths.checkpoint` and set
`train.expected_step=0` in a copied config. This is not an exact reconstruction
of the original early training history: the released continuation config starts
at step 6,000. Full training was not rerun as part of the portability cleanup.

## Repository layout and existing tools

- `src/swam/tasks/manipulation/prediction/`: configuration, frozen encoding,
  training, inference benchmark, report generation and artifact verification.
- `src/swam/flow_*.py`, `state_tokenizer.py`: existing numerical model, sampling
  and cache implementations, retained under their original import names.
- `src/swam/retargeting/`: geometry and robot-hand retargeting.
- `configs/`: portable experiment settings and frozen artifact checksums.
- `scripts/`: preserved exploratory and orchestration sources. These are not
  the supported reproduction entrypoint and may require additional software.
- `results/`, `notes/`, `data/`, `artifacts/`, `external/`, `tests/`: local material,
  ignored by Git. Existing tracked tests remain tracked.

Original run snapshots and historical environment JSON retain their recorded
paths as provenance. They are not runtime defaults. They are excluded from
`git archive` source distributions; changing those records would invalidate
historical evidence. Existing experimental scripts retain their filenames;
path-only changes do not certify an old collection cache as identical. No
collection, physics or controller settings are changed by this cleanup.

For external retargeting installations, set `SWAM_RETARGET_ROOT`,
`SWAM_SAM_ROOT`, `SWAM_SAM_ENV_ROOT`, `SWAM_VIDEO_DEPS_ROOT`,
`SWAM_PROMPT_MODEL_ROOT` and `SWAM_DINO_MODEL_ROOT` as needed. Defaults live
under `external/`. `SWAM_DATA_ROOT`, `SWAM_MODEL_ROOT`, `SWAM_CACHE_ROOT` and
`SWAM_ASSETS_ROOT` override data, frozen models, scratch files and imported mesh
references. `SWAM_PYTHON` selects an Isaac Lab-compatible interpreter for
`run.sh`; otherwise it uses `python3` from the active environment. Documentation
examples using `TINY_WAM_ROOT` refer to the caller's checkout.

Robot assets, third-party checkpoints and source videos retain their original
licenses. Artifact export only copies local files; it does not grant permission
to redistribute them or publish a release.

## Portability verification

The 2026-10-22 check installed the locked environment from scratch and completed
all 96 inference settings (2,400 timed ticks) on one RTX 4090. Four-step CUDA
Graph inference averaged 37.47 ms, or 26.69 Hz, under the timing scope above.
[The verification record](reproduction-check.json) contains input hashes,
summary measurements and checks. Full distributed training was not rerun.
