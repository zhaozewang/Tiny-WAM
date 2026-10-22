"""Encode verified NPZ or lossless-video shards with frozen tokenizers."""
from contextlib import ExitStack
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

import numpy as np
import torch

from swam.shards import atomic_json, sha256
from .codec import InputCodec


def exact_rgb(pipe, frames):
    size = frames * 256 * 256 * 3
    data = bytearray()
    while len(data) < size:
        part = pipe.read(size - len(data))
        if not part:
            raise RuntimeError(f'Truncated RGB stream: {len(data)}/{size} bytes')
        data.extend(part)
    array = np.frombuffer(data, dtype=np.uint8)
    return array.reshape(frames, 256, 256, 3)


def stop_decoder(process):
    if process.poll() is None:
        process.kill()
    process.wait()
    process.stdout.close()


def open_decoders(archive, directory, stack):
    storage = json.load(archive.extractfile('storage.json'))
    if storage['resolution'] != [256, 256] or storage['fps'] != 24:
        raise ValueError('This experiment requires 256px, 24-Hz inputs; no implicit resizing')
    if storage['views'] != ['head', 'left', 'right']:
        raise ValueError('Unexpected camera ordering')
    decoders = []
    for view in storage['views']:
        path = Path(directory) / (view + '.mp4')
        with path.open('wb') as stream:
            shutil.copyfileobj(archive.extractfile(view + '.mp4'), stream)
        command = ['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path),
                   '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '2', 'pipe:1']
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        stack.callback(stop_decoder, process)
        decoders.append(process)
    return decoders


def episode_blocks(archive, count, decoders):
    previous = None
    for index in range(count):
        raw = archive.extractfile(f'block_{index:05d}.npz').read()
        with np.load(io.BytesIO(raw)) as source:
            block = {key: source[key] for key in source.files}
        expected = np.arange(index * 120, index * 120 + 121)
        if not np.array_equal(block['frame_indices'], expected):
            raise ValueError('Non-contiguous observation/action block')
        if len(block['qpos']) != 121 or len(block['actions']) != 120:
            raise ValueError('Expected 121 observations and 120 commands')
        if decoders:
            images = []
            for process in decoders:
                images.append(exact_rgb(process.stdout, 121 if index == 0 else 120))
            frames = np.stack(images, axis=1)
            block['rgb'] = frames if previous is None else np.concatenate((previous, frames), axis=0)
            previous = block['rgb'][-1:].copy()
        if block['rgb'].shape != (121, 3, 256, 256, 3):
            raise ValueError('Unexpected RGB shape')
        yield index, block


def encode_state(codec, name, raw):
    value = torch.from_numpy(raw).float().cuda()
    return codec.encode_state(name, value)[:, 0].half().cpu()


def encode_block(codec, block, batch_size, source_sha256):
    flat = block['rgb'].reshape(-1, 256, 256, 3)
    codes = []
    for start in range(0, len(flat), batch_size):
        image = torch.from_numpy(flat[start:start + batch_size].copy()).cuda()
        image = image.permute(0, 3, 1, 2).half() / 127.5 - 1
        latent = codec.image.encode(image).latent
        if latent.shape[1:] != (32, 8, 8):
            raise ValueError('DC-AE latent shape differs from the frozen experiment')
        codes.append(latent.flatten(1).cpu())
    images = torch.cat(codes).reshape(121, 3, 2048)
    proprio = np.concatenate((block['qpos'], block['qvel']), axis=-1)
    result = dict(images=images, proprio=encode_state(codec, 'proprio', proprio),
                  actions=encode_state(codec, 'action', block['actions']),
                  qpos=torch.from_numpy(block['qpos']), frames=torch.from_numpy(block['frame_indices']),
                  source_sha256=source_sha256)
    if not all(torch.isfinite(result[key]).all() for key in ('images', 'proprio', 'actions', 'qpos')):
        raise ValueError('Nonfinite encoded block')
    return result


def encode_episode(source, output, row, codec, batch_size):
    shard = source / 'shards' / row['file']
    if not shard.resolve().is_relative_to((source / 'shards').resolve()):
        raise ValueError('Shard filename escapes the dataset')
    if sha256(shard) != row['sha256']:
        raise ValueError(f'Raw shard checksum mismatch: {row["file"]}')
    canonical = row.get('original_sha256', row['sha256'])
    destination = output / f'episode_{row["episode"]:05d}'
    destination.mkdir()
    with ExitStack() as stack:
        archive = stack.enter_context(tarfile.open(shard))
        metadata = json.load(archive.extractfile('metadata.json'))
        for checkpoint in codec.checkpoints.values():
            if metadata['joint_names'] != checkpoint['joint_names']:
                raise ValueError('Joint ordering differs from the tokenizer')
        decoders = []
        if 'storage.json' in archive.getnames():
            if row.get('storage_format') != 'mp4_rgb_lossless_v1' or not row.get('rgb_verified_lossless'):
                raise ValueError('Video shards must have verified lossless RGB provenance')
            directory = stack.enter_context(tempfile.TemporaryDirectory(prefix='tiny-wam-rgb-'))
            decoders = open_decoders(archive, directory, stack)
        write_blocks(archive, row['blocks'], decoders, codec, batch_size, canonical, destination)
    return dict(episode=row['episode'], split=row['split'], blocks=row['blocks'],
                source_sha256=canonical, storage_sha256=row['sha256'])


def write_blocks(archive, count, decoders, codec, batch_size, source_sha256, destination):
    for index, block in episode_blocks(archive, count, decoders):
        encoded = encode_block(codec, block, batch_size, source_sha256)
        path = destination / f'block_{index:05d}.pt'
        temporary = path.with_suffix('.partial')
        torch.save(encoded, temporary)
        temporary.replace(path)
    for process in decoders:
        if process.stdout.read(1) or process.wait() != 0:
            raise RuntimeError('Unexpected extra frames or failed video decoder')


@torch.inference_mode()
def run(experiment, batch_size=16):
    paths = experiment.paths
    if not all((paths.raw_dataset, paths.latents, paths.reference, paths.tokenizers, paths.image_model)):
        raise ValueError('prepare needs raw_dataset, latents, reference, tokenizers and image_model paths')
    if batch_size < 1:
        raise ValueError('Encoding batch must be positive')
    raw_index = json.loads((paths.raw_dataset / 'index.json').read_text())
    reference = torch.load(paths.reference, map_location='cpu', weights_only=False)
    hashes = {}
    for name in ('proprio', 'action'):
        hashes[name] = sha256(paths.tokenizers / f'{name}.pt')
    if hashes != reference['norm']['tokenizer_sha256']:
        raise ValueError('Frozen reference normalization and tokenizers disagree')
    codec = InputCodec(reference, paths.tokenizers, paths.image_model)
    paths.latents.mkdir(parents=True, exist_ok=False)
    torch.save(reference['norm'], paths.latents / 'normalization.pt')
    index = dict(episodes={}, tokenizer_sha256=hashes, fps=24, block_actions=120)
    for row in raw_index['shards']:
        encoded = encode_episode(paths.raw_dataset, paths.latents, row, codec, batch_size)
        index['episodes'][str(row['episode'])] = encoded
        atomic_json(paths.latents / 'index.json', index)
        print(json.dumps(encoded), flush=True)
