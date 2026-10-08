# SWAM: R1 + ISyHand scene

Self-contained scene assets and preview code for the original Galaxea R1 arms and fixed base/torso, two ISyHands at 85% scale, a mirrored left hand, connected wrist adapters, and an ALOHA table. The two native wrist RealSense camera housings are removed. The three render cameras are head, close left, and close right.

## Run on remi

Uses the existing adjacent `../dexverse-stack/env` Isaac Lab / Isaac Sim 5.1 environment. Override `SWAM_PYTHON` to use another compatible interpreter. Code and assets do not depend on the remi-manip-test project.

```bash
cd /home/zw/projects/swam
./run.sh --output results/stills --frames 1 --device cuda:0
./run.sh --output results/random_motion --random-preview --frames 150 --device cuda:0
```

Use a new output directory for each run. Random preview produces `head.mp4`, `left.mp4`, `right.mp4` and `synchronized.mp4`, plus camera metadata and `trajectory.npz`. All streams are 15 fps; 150 frames is 10 seconds. The individual cameras render at 960x960. Seed 42; random arm targets within +/-0.15 rad of the reference pose and joint limits; hand targets within URDF limits; one target each second, cubic interpolation for display.

## Scope

These are kinematic animations with 48 moving joints (6 arm + 18 hand per side), not physics-controlled rollouts or trained policies. Collision-free motion and dynamic actuator settings are not validated. Scaled hand geometry retains source mass/inertia; left-hand frames, axes, meshes and inertia orientation are reflected. No claim of physically calibrated dynamics.

## Assets

- `assets/galaxea/R1`: official userguide-galaxea/URDF R1 model, source commit b81c32520b1d1a3b3299df0c038e042001b46ae3. Upstream root did not contain a LICENSE file in the source checkout.
- `assets/isyhand`: supplied ISyHand v6 model; source-project license retained. The repository license is not a separate assertion of licensing for the supplied meshes.
- `assets/menagerie/aloha`: MuJoCo Menagerie table and texture, source commit 4d038b3feae26ec82b46a4d586379114012a8ac7; licenses retained.
- `assets/hand_replay`: prior hand joint trajectory for assembly preview only.
- `assets/arm_pose.json`: accepted arm pose reference.

Previous generated media are preserved in the original project. New renders belong in this project's `results/` directory.

## RGB Hand Retargeting

The separate [RGB retargeting research and reproduction guide](docs/retargeting/README.md)
documents EgoDex diagnostics, published-method comparisons, WiLoR hand estimation,
Sharpa replacement, object occlusion, silhouette refinement, and R1 arms fitted
without moving the accepted hands. It includes exact local dependencies, commands,
validation results and limitations. These are visual overlays, not physically
validated robot demonstrations; private media and model weights remain outside Git.
