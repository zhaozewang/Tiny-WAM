"""Portable entrypoints for model fitting, preparation and measured inference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import load_config


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='command', required=True)
    for name in ('doctor', 'reproduce', 'train', 'prepare', 'export', 'evaluate'):
        command = commands.add_parser(name)
        command.add_argument('--config', type=Path, required=True)
        if name == 'doctor':
            command.add_argument('--training', action='store_true')
            command.add_argument('--skip-hashes', action='store_true')
            command.add_argument('--cpu', action='store_true', help='Inspect files without requiring CUDA')
        if name == 'export':
            command.add_argument('--output', type=Path, required=True)
        if name == 'prepare':
            command.add_argument('--batch', type=int, default=16)
    report = commands.add_parser('report')
    report.add_argument('--input', type=Path, required=True)
    report.add_argument('--output', type=Path, required=True)
    return cli


def train(experiment):
    from .artifacts import check
    if 'RANK' not in os.environ:
        check(experiment, training=True)
        if experiment.paths.output.exists():
            raise FileExistsError(f'Choose a new output directory: {experiment.paths.output}')
        if experiment.train.workers > 1:
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nproc-per-node', str(experiment.train.workers), '--module',
                       'swam.tasks.manipulation.prediction', 'train',
                       '--config', str(experiment.source)]
            subprocess.run(command, check=True)
            return
    from .train import run
    run(experiment)


def reproduce(experiment):
    from .artifacts import check
    check(experiment)
    output = experiment.paths.output
    output.mkdir(parents=True, exist_ok=False)
    (output / 'experiment.toml').write_bytes(experiment.source.read_bytes())
    from .benchmark import run
    from .report import run as report
    run(experiment, output / 'benchmark')
    report(output / 'benchmark/results.json', output)
    print(f'Completed measured inference reproduction: {output}')


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.command == 'report':
        from .report import run
        run(arguments.input, arguments.output)
        return
    try:
        experiment = load_config(arguments.config)
        if arguments.command == 'doctor':
            from .artifacts import check
            result = check(experiment, training=arguments.training,
                           hashes=not arguments.skip_hashes, cuda=not arguments.cpu)
            print(json.dumps(result, indent=2))
        elif arguments.command == 'reproduce':
            reproduce(experiment)
        elif arguments.command == 'train':
            train(experiment)
        elif arguments.command == 'evaluate':
            from .feedback import run
            run(experiment)
        elif arguments.command == 'prepare':
            from .prepare import run
            run(experiment, batch_size=arguments.batch)
        elif arguments.command == 'export':
            from .artifacts import export
            export(experiment, arguments.output)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
