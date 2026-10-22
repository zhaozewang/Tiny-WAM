"""Verify and export the external inputs needed to repeat an experiment."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(paths, entry):
    name = entry['input']
    if name not in ('checkpoint', 'reference', 'tokenizers', 'image_model'):
        raise ValueError(f'Unknown artifact input: {name}')
    base = getattr(paths, name)
    if base is None:
        raise ValueError(f'Configure paths.{name}')
    if 'file' not in entry:
        return base
    relative = Path(entry['file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Artifact filename escapes its configured directory')
    # A caller may deliberately link large weights into the configured directory.
    return base / relative


def verify_manifest(paths):
    if paths.manifest is None:
        raise ValueError('Reproduction requires paths.manifest with input checksums')
    manifest = json.loads(paths.manifest.read_text())
    if not manifest.get('files'):
        raise ValueError('Artifact manifest has no files')
    inputs = {entry['input'] for entry in manifest['files']}
    if inputs != {'checkpoint', 'reference', 'tokenizers', 'image_model'}:
        raise ValueError('Manifest must cover model, reference, tokenizers and image model')
    for entry in manifest['files']:
        path = manifest_path(paths, entry)
        if not path.is_file():
            raise FileNotFoundError(f'Missing artifact: {path}')
        if path.stat().st_size != entry['bytes'] or checksum(path) != entry['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {path}')
    return len(manifest['files'])


def check_training(experiment):
    import torch
    from dataclasses import asdict
    from swam.flow_model import FlowConfig

    # Validate data identity and completeness before creating any distributed workers.
    paths = experiment.paths
    index = json.loads((paths.latents / 'index.json').read_text())
    norm = torch.load(paths.latents / 'normalization.pt', map_location='cpu', weights_only=False)
    if index['tokenizer_sha256'] != norm['tokenizer_sha256']:
        raise ValueError('Dataset index and normalization use different tokenizers')
    if not index['episodes']:
        raise ValueError('Latent dataset has no episodes')
    if paths.training_manifest is not None:
        expected = json.loads(paths.training_manifest.read_text())
        for key in ('tokenizer_sha256', 'fps', 'block_actions'):
            if index.get(key) != expected.get(key):
                raise ValueError(f'Training data differs from the frozen manifest: {key}')
        if index['episodes'] != expected['episodes']:
            raise ValueError('Training episode identity, splits or block counts differ from the frozen manifest')
    for row in index['episodes'].values():
        directory = paths.latents / f"episode_{row['episode']:05d}"
        for block in range(row['blocks']):
            if not (directory / f'block_{block:05d}.pt').is_file():
                raise FileNotFoundError(f'Missing published latent block {block} in {directory}')

    # A model-only checkpoint cannot reproduce a training continuation.
    if paths.checkpoint is None:
        return
    checkpoint = torch.load(paths.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if checkpoint['step'] != experiment.train.expected_step:
        raise ValueError('Training checkpoint step differs from expected_step')
    if checkpoint['config'] != asdict(FlowConfig(**experiment.model)):
        raise ValueError('Model configuration differs from the training checkpoint')
    if 'optimizer' not in checkpoint:
        raise ValueError('Training continuation needs full optimizer state')
    if checkpoint.get('tokenizer_sha256') != norm['tokenizer_sha256']:
        raise ValueError('Checkpoint and dataset use different tokenizers')


def check(experiment, training=False, hashes=True, cuda=True):
    paths = experiment.paths
    names = ('latents',) if training else ('checkpoint', 'reference', 'tokenizers', 'image_model')
    errors = []
    for name in names:
        value = getattr(paths, name)
        if value is None or not value.exists():
            errors.append(f'paths.{name}: missing {value or "configuration"}')
    if training and experiment.train.expected_step and not (paths.checkpoint and paths.checkpoint.is_file()):
        errors.append('Training continuation needs a full checkpoint at paths.checkpoint')
    modules = ['torch', 'numpy']
    modules += ['bitsandbytes'] if training and experiment.train.optimizer == 'adamw8bit' else []
    modules += [] if training else ['diffusers', 'matplotlib']
    for module in modules:
        if importlib.util.find_spec(module) is None:
            errors.append(f'Missing Python dependency: {module}')
    if cuda and importlib.util.find_spec('torch'):
        import torch
        if not torch.cuda.is_available():
            errors.append('CUDA is unavailable; install the CUDA build of PyTorch and check the driver')
        elif training and torch.cuda.device_count() < experiment.train.workers:
            errors.append(f'Training requires {experiment.train.workers} visible GPUs')
    if training and not errors:
        try:
            check_training(experiment)
        except (OSError, ValueError, KeyError) as error:
            errors.append(str(error))
    if not training and hashes and not errors:
        try:
            verify_manifest(paths)
        except (OSError, ValueError, KeyError) as error:
            errors.append(str(error))
    if errors:
        raise RuntimeError('\n'.join(errors) + '\nSee docs/reproduce.md for the artifact layout.')
    return dict(status='ready', mode='training' if training else 'reproduction', hashes=hashes and not training)


def export(experiment, output):
    """Make a portable directory; exporting does not upload or publish anything."""
    verify_manifest(experiment.paths)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(experiment.paths.manifest.read_text())
    names = {'checkpoint': 'model.pt', 'reference': 'eval_reference.pt',
             'tokenizers': 'tokenizers', 'image_model': 'dcae'}
    for entry in manifest['files']:
        source = manifest_path(experiment.paths, entry)
        destination = output / names[entry['input']]
        if 'file' in entry:
            destination /= entry['file']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if checksum(destination) != entry['sha256']:
            raise IOError(f'Export verification failed: {destination}')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
