# Data collection

Collection requires Isaac Sim / Isaac Lab, the R1 and ISyHand assets described in
`docs/reproduce.md`, and a CUDA device with working camera rendering. Run from
the repository root with its `src` installed in the simulator Python environment.

The collectors represent different experiments; their settings are not interchangeable.

| Collector | Images | Position/velocity solver iterations | Hand damping | Motion |
|---|---|---|---|---|
| `tinywam.collect` | 256² | 16 / 4 | 0.1 | Original five-second random targets |
| `tinywam.collect_long` | 256² | 32 / 8 | 0.3 | v5 bounded targets, 1–3 s knots, holds |
| `tinywam.collect_continuous512` | 512² | 64 / 8 | 0.1 | Original one-second targets extended continuously |

All use 24 Hz observations/actions and 120 Hz physics. The v5 scene additionally
sets a 1 mm contact offset. The 512² collector uses visual partitions. See
`docs/experiments.md` for provenance and the distinction between these datasets.

## Commands

Prepare the assembled URDF and converted USD with the existing preview entrypoint:

```sh
./run.sh --output results/collection_assets --frames 1
```

Collect a single batch of continuous 512² episodes:

```sh
python -m tinywam.collect_continuous512 --root data/continuous512 \
  --assets results/collection_assets --ids 0 1 --seconds 600 --device cuda:0
```

The supervisor publishes completed batches into restartable TAR shards:

```sh
python scripts/generate_continuous512.py --root data/continuous512 \
  --assets results/collection_assets --episodes 10 --parallel 10 --seconds 600
```

For the historical v5 protocol, use `scripts/generate_dataset.py` with the same
I/O options. `scripts/review_long_batch.py --root data/v5` reviews the v5 256²,
ten-minute layout; it is not a validator for the 512² protocol.

`--storage-root /path/to/mount` optionally requires a particular mounted
filesystem. Supervisors retain a 500 GiB free-space floor plus their per-worker
reserve. They fail on changed configuration or source provenance. Use a new
output directory after a source revision; resume an old collection using its
original source snapshot. Do not replace an existing provenance directory.

`python scripts/convert_dataset_mp4.py --base data --dataset DATASET_NAME`
converts image blocks to lossless RGB video, verifies decoded pixels, and commits
through a recoverable transaction. Keep a backup before migrating a dataset.

These commands start collection only when invoked. Source publication does not
constitute a new full-duration simulator validation.
