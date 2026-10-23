"""Portable, explicit inputs for reaching training and inference experiments."""
from dataclasses import dataclass, field, fields
import os
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Paths:
    output: Path
    checkpoint: Path | None = None
    reference: Path | None = None
    tokenizers: Path | None = None
    image_model: Path | None = None
    latents: Path | None = None
    raw_dataset: Path | None = None
    manifest: Path | None = None
    training_manifest: Path | None = None
    scene_assets: Path | None = None


@dataclass(frozen=True)
class Benchmark:
    expected_step: int = 50000
    steps: tuple = (1, 2, 4, 8)
    history: tuple = (1, 24, 60, 119)
    repeats: int = 25
    warmup: int = 5

    def __post_init__(self):
        if not self.steps or min(self.steps) < 1:
            raise ValueError('benchmark.steps must contain positive flow counts')
        if not self.history or min(self.history) < 1 or max(self.history) > 120:
            raise ValueError('benchmark.history must be in [1, 120]')
        if self.repeats < 1 or self.warmup < 0:
            raise ValueError('Use positive repeats and nonnegative warmup')


@dataclass(frozen=True)
class Training:
    expected_step: int = 6000
    steps: int = 50000
    workers: int = 4
    batch: int = 3
    accumulation: int = 3
    base_lr: float = 5e-5
    checkpoint_every: int = 200
    frames: int = 121
    seed: int = 42
    optimizer: str = 'adamw8bit'
    wandb_mode: str = 'disabled'
    wandb_project: str = 'tinywam'
    wandb_entity: str | None = None

    def __post_init__(self):
        positive = (self.steps, self.workers, self.batch, self.accumulation,
                    self.base_lr, self.checkpoint_every)
        if min(positive) <= 0 or self.expected_step < 0 or self.frames < 4:
            raise ValueError('Invalid training dimensions or schedule')
        if self.expected_step >= self.steps:
            raise ValueError('Training checkpoint must precede the final step')
        if self.optimizer not in ('adamw8bit', 'adamw'):
            raise ValueError('optimizer must be adamw8bit or adamw')
        if self.wandb_mode not in ('disabled', 'offline', 'online'):
            raise ValueError('Invalid wandb_mode')


@dataclass(frozen=True)
class Evaluation:
    backend: str = 'reference'
    device: str = 'cuda:0'
    steps: tuple = (4, 8, 16)
    splits: tuple = ()
    goal_seed: int = 1729
    noise_seed: int = 1729
    extra_noise_seed: int = 2026
    ablation_steps: int = 8

    def __post_init__(self):
        if self.backend not in ('reference', 'cached', 'cuda_graph'):
            raise ValueError('Unknown evaluation backend')
        if not self.steps or min(self.steps) < 1 or self.ablation_steps < 1:
            raise ValueError('Evaluation flow counts must be positive')
        if len(set(self.steps)) != len(self.steps):
            raise ValueError('Evaluation flow counts must be unique')


@dataclass(frozen=True)
class Experiment:
    source: Path
    root: Path
    paths: Paths
    benchmark: Benchmark = field(default_factory=Benchmark)
    train: Training = field(default_factory=Training)
    model: dict = field(default_factory=dict)
    evaluate: Evaluation = field(default_factory=Evaluation)


def resolve_path(root, value):
    """Config paths do not depend on the process's working directory."""
    expanded = os.path.expandvars(value)
    if '$' in expanded:
        raise ValueError(f'Unset environment variable in path: {value}')
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_config(source):
    source = Path(source).resolve()
    with source.open('rb') as stream:
        values = tomllib.load(stream)
    unknown = set(values) - {'root', 'paths', 'benchmark', 'train', 'model', 'evaluate'}
    if unknown:
        raise ValueError(f'Unknown configuration sections: {sorted(unknown)}')

    # Resolve all I/O before importing GPU libraries or creating output folders.
    root = resolve_path(source.parent, values.get('root', '.'))
    paths = {}
    allowed = {item.name for item in fields(Paths)}
    for name, value in values.get('paths', {}).items():
        if name not in allowed:
            raise ValueError(f'Unknown path: {name}')
        paths[name] = resolve_path(root, value)
    benchmark = Benchmark(**values.get('benchmark', {}))
    train = Training(**values.get('train', {}))
    evaluate = Evaluation(**values.get('evaluate', {}))
    return Experiment(source, root, Paths(**paths), benchmark, train, values.get('model', {}), evaluate)
