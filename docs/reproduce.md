# Reproduction

## Install

Use Linux x86-64, Python 3.11 and a CUDA 12.8-compatible NVIDIA GPU. The inference
benchmark runs on one RTX 4090 with 24 GB VRAM. Raw-video preparation also needs
FFmpeg; scene preview and retargeting use separate environments.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
```

`requirements.lock` pins the tested dependencies, including PyTorch's CUDA build.
Paths in TOML configs resolve relative to their `root`, which is relative to
the config file.

## Weights and reference data

Place the step-50,000 model, frozen encoders and evaluation reference in:

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

The bundle is approximately 7 GB. Sizes and SHA-256 hashes are in
`configs/artifacts.json`; a public download link is not yet available.
Historical PyTorch checkpoints contain Python objects and use
`weights_only=False`, so load trusted files only.

To package existing files from a local config:

```bash
tiny-wam export --config inputs.toml --output bundle/
```

## Inference benchmark

```bash
tiny-wam reproduce --config configs/reproduce.toml
```

The command checks artifact hashes and compares reference, KV-cache and
CUDA-graph inference. It measures 1/2/4/8 flow steps at 1/24/60/119 history
frames on one training and one validation sample, with five warmups and
25 timed ticks per setting. Outputs in `results/reproduce/` include CSV/JSON
summaries, PNG/PDF plots and individual timings in `benchmark/results.json`.

Timing includes RGB transfer, encoding, history update, sampling, action
decoding and transfer to CPU. Sensing, simulation, disk I/O and model loading
are excluded. Initial cache fill and graph capture are reported separately.
Action differences are measured against reference inference. This benchmark
measures recorded-history latency and numerical agreement, not task success.

The [2026-10-22 measurements](reproduction-check.json) cover 96 settings and
2,400 timed ticks. Four-step CUDA-graph inference averaged 37.47 ms (26.69 Hz)
on an RTX 4090. Hardware, software versions and concurrent load affect timing.

Use a new `paths.output` for each run. To check inputs without running inference:

```bash
tiny-wam doctor --config configs/reproduce.toml
# File and dependency checks without CUDA:
tiny-wam doctor --config configs/reproduce.toml --cpu
```

## Prepare data

The model uses historical v5 data; see [dataset versions](experiments.md#dataset-versions).
Existing latents go in `data/flow_latents_v5/`, containing `index.json`,
`normalization.pt` and the indexed `episode_*/block_*.pt` files.
`configs/training-data.json` identifies the 68 episodes and 8,160 blocks.

Alternatively, put raw v5 `index.json` and `shards/` in
`data/raw/r1_random_10min_v5/` and run:

```bash
tiny-wam prepare --config configs/prepare.toml
```

Preparation checks shard hashes and joint order, then encodes with the frozen
DC-AE and state/action VAEs. It keeps three camera views at 24 Hz and
121 observations per 120 commands. MP4 shards require verified lossless RGB
provenance. The output latent directory must not already exist.

## Train

`configs/train.toml` resumes step 6,000 through 50,000 using four GPUs,
microbatch 3, accumulation 3, AdamW8bit and base LR 5e-5. Supply the full
checkpoint, including optimizer state, at `artifacts/training/step_006000.pt`.
Its checksum is in `configs/training-checkpoint.json`.

```bash
tiny-wam doctor --config configs/train.toml --training
tiny-wam train --config configs/train.toml
```

The launcher uses `torchrun`, writes per-rank metrics and saves `latest.pt`
every 200 steps. W&B is optional and disabled by default.

To resume an interrupted run, set `paths.checkpoint` to its `latest.pt`,
`train.expected_step` to the saved step, and use a new output folder. Keep
`train.steps=50000` to retain the original learning-rate schedule.

For a fresh run, omit `paths.checkpoint` and set `train.expected_step=0` in
a copied config. The supplied config reproduces the continuation, not the
original early training schedule. Validation covered data loading, microbatch
loss/gradients, checkpoint preflight and inference; full distributed training
and raw-dataset preparation were not rerun.

## Layout

- `src/swam/tasks/manipulation/prediction/`: preparation, training, benchmarking and reports.
- `src/swam/flow_*.py`, `state_tokenizer.py`: model, sampling, cache and tokenizers.
- `src/swam/retargeting/`: hand and arm geometry.
- `configs/`: experiment settings and artifact manifests.
- `scripts/`: earlier experiment and scene tools.

Data, weights, environments and generated outputs are Git-ignored. Historical
JSON snapshots retain their recorded paths and are excluded from `git archive`.

For external installations, `SWAM_DATA_ROOT`, `SWAM_MODEL_ROOT`,
`SWAM_CACHE_ROOT` and `SWAM_ASSETS_ROOT` override data, weights, cache and mesh
locations. Retargeting uses `SWAM_RETARGET_ROOT`, `SWAM_SAM_ROOT`,
`SWAM_SAM_ENV_ROOT`, `SWAM_VIDEO_DEPS_ROOT`, `SWAM_PROMPT_MODEL_ROOT` and
`SWAM_DINO_MODEL_ROOT`; defaults are under `external/`. `SWAM_PYTHON` selects
the Isaac Lab interpreter for `run.sh`. `TINY_WAM_ROOT` in documentation
examples means the checkout directory.

Third-party weights, meshes and videos retain their original licenses.
