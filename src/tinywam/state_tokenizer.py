"""Frame-wise continuous proprioception tokenizer with Fourier features."""
import torch
from torch import nn


class ProprioVAE(nn.Module):
    def __init__(self, input_dim=36, latent_dim=512, hidden_dim=512,
                 frequencies=(.25, .5, 1., 2.)):
        super().__init__()
        self.register_buffer('frequencies', torch.tensor(frequencies))
        expanded = input_dim * (1 + 2 * len(frequencies))
        self.encoder = nn.Sequential(nn.Linear(expanded, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, input_dim))

    def encode(self, x):
        phase = 2 * torch.pi * x.unsqueeze(-1) * self.frequencies
        features = torch.cat((x, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)
        h = self.encoder(features)
        return self.mean(h), self.logvar(h)

    def forward(self, x, sample=True):
        mean, logvar = self.encode(x)
        z = mean + torch.exp(.5 * logvar) * torch.randn_like(mean) if sample else mean
        return self.decoder(z), mean, logvar

    @staticmethod
    def kl(mean, logvar):
        return .5 * (mean.square() + logvar.exp() - 1 - logvar).mean()
