# Tiny-WAM

A small latent world/action model for robot reaching. It models camera images,
joint state and actions at 24 Hz, conditioned on timed joint-position goals.
The repository also includes R1 + ISyHand scene assets and an RGB hand-retargeting
pipeline.

## Setup

Linux, Python 3.11 and CUDA 12.8. The inference benchmark uses one 24 GB RTX 4090.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
```

Place the model weights and reference data in `artifacts/`, then run:

```bash
tiny-wam reproduce --config configs/reproduce.toml
```

Weights and datasets are distributed separately; a download link is not yet
available. See [reproduction](docs/reproduce.md) for the required files, training
commands and benchmark protocol, and [experiment notes](docs/experiments.md)
for dataset versions and training settings.

## Scene preview

The scene contains Galaxea R1 arms, a fixed base and torso, two ISyHands at 85%
scale, wrist adapters and an ALOHA table. The left hand is mirrored. Rendered
views are head, left and right; the native wrist-camera housings are removed.

Preview requires a separate Isaac Lab / Isaac Sim 5.1 environment. Activate it
or set `TINYWAM_PYTHON` to its interpreter:

```bash
./run.sh --output results/stills --frames 1 --device cuda:0
./run.sh --output results/random_motion --random-preview --frames 150 --device cuda:0
```

Each run needs a new output directory. The random preview writes three 960×960
videos at 15 fps, a synchronized video, camera metadata and `trajectory.npz`.
It uses seed 42, arm targets within ±0.15 rad of the reference pose and joint
limits, and hand targets within URDF limits. Targets change once per second,
with cubic interpolation for display.

The preview is a kinematic animation of 48 joints, not a physics rollout.
Scaled and mirrored hand geometry retains transformed source inertia; dynamics
and collision-free motion have not been calibrated.

## Hand retargeting

The [retargeting guide](docs/retargeting/README.md) covers EgoDex, WiLoR hand
estimation, Sharpa replacement, object occlusion, silhouette refinement and
R1 arm fitting. These outputs are visual overlays, not executable robot
trajectories. Media and third-party model weights are kept outside Git.

## Assets

- `assets/galaxea/R1`: Galaxea R1 URDF, source commit `b81c32520b1d1a3b3299df0c038e042001b46ae3`. The upstream checkout had no root license file.
- `assets/isyhand`: supplied ISyHand v6 model, with its source license.
- `assets/menagerie/aloha`: MuJoCo Menagerie table and texture, source commit `4d038b3feae26ec82b46a4d586379114012a8ac7`, with upstream licenses.
- `assets/hand_replay`: joint trajectory used for assembly preview.
- `assets/arm_pose.json`: reference arm pose.

Mesh and third-party asset licenses apply separately from the repository code.
