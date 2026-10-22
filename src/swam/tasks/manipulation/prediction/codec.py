"""Frozen observation encoders and the joint-command decoder."""
from pathlib import Path

import torch
from swam.flow_model import FlowConfig, FlowWorldModel
from swam.state_tokenizer import ProprioVAE


def image_model_directory(path):
    path = Path(path)
    if (path / 'config.json').is_file():
        return path
    candidates = sorted(path.glob('*/config.json'))
    if len(candidates) != 1:
        raise ValueError(f'Expected one DC-AE config.json in {path}; found {len(candidates)}')
    return candidates[0].parent


def load_model(path, expected_step, device='cuda'):
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if checkpoint['step'] != expected_step:
        raise ValueError(f"Expected checkpoint {expected_step}, got {checkpoint['step']}")
    config = FlowConfig(**checkpoint['config'])
    model = FlowWorldModel(config).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    model.requires_grad_(False)
    return model


def device_values(values, device):
    result = {}
    for name, value in values.items():
        result[name] = value.to(device) if torch.is_tensor(value) else value
    return result


class InputCodec:
    def __init__(self, reference, tokenizers, image_model=None, device='cuda'):
        self.device = device
        self.norm = device_values(reference['norm'], device)
        self.models = {}
        self.checkpoints = {}

        # Both state encoders retain their checkpointed normalization.
        for name in ('proprio', 'action'):
            path = Path(tokenizers) / f'{name}.pt'
            checkpoint = torch.load(path, weights_only=False, map_location='cpu')
            model = ProprioVAE(input_dim=checkpoint['input_dim']).to(device).eval()
            model.load_state_dict(checkpoint['model'])
            model.requires_grad_(False)
            self.models[name] = model
            values = {key: value for key, value in checkpoint.items() if key != 'model'}
            self.checkpoints[name] = device_values(values, device)

        # State-only evaluation does not need to instantiate the image decoder.
        self.image = None
        if image_model is not None:
            from diffusers import AutoencoderDC
            directory = image_model_directory(image_model)
            model = AutoencoderDC.from_pretrained(directory, torch_dtype=torch.float16,
                                                   local_files_only=True)
            self.image = model.to(device).eval().requires_grad_(False)

    def encode_state(self, name, value):
        checkpoint = self.checkpoints[name]
        normalized = (value - checkpoint['mean']) / checkpoint['scale']
        latent = self.models[name].encode(normalized)[0]
        normalized = (latent - checkpoint['latent_mean']) / checkpoint['latent_scale']
        return normalized[:, None]

    def encode(self, rgb, state, action):
        image = rgb.to(self.device).permute(2, 0, 1)[None].half() / 127.5 - 1
        latent = self.image.encode(image).latent
        image = latent.flatten(1).float()
        image = ((image - self.norm['image_mean']) / self.norm['image_scale'])[:, None]
        proprio = self.encode_state('proprio', state)
        command = self.encode_state('action', action)
        return image, proprio, command

    def decode(self, latent):
        checkpoint = self.checkpoints['action']
        unscaled = latent * checkpoint['latent_scale'] + checkpoint['latent_mean']
        value = self.models['action'].decoder(unscaled)
        return (value * checkpoint['scale'] + checkpoint['mean']).cpu()
