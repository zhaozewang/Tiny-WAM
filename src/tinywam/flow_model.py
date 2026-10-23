"""Three-token, frame-causal world/action flow model.

Times are physical seconds in one trial coordinate system. Inputs are normalized
latents, not pixels. An action at t causes observation t+1. This module does not
load tokenizers or claim that synthetic resource probes constitute training.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class FlowConfig:
    width: int = 2048
    layers: int = 22
    heads: int = 16
    intermediate: int = 4096
    dims: tuple = (2048, 512, 512)
    goal_dim: int = 48
    time_features: int = 256
    time_scale: float = 24.0
    noise_embedding_scale: float = 1000.0
    checkpointing: bool = True
    qk_norm: bool = True

    def __post_init__(self):
        assert self.width % self.heads == 0 and (self.width // self.heads) % 2 == 0
        assert len(self.dims) == 3 and self.time_features % 2 == 0


@dataclass
class FlowOutput:
    velocity: tuple
    loss_mask: torch.Tensor  # [B,T,3]; terminal action and unavailable tokens excluded
    clean_hidden: torch.Tensor | None = None


def layer_norm(x):
    with torch.autocast(x.device.type, enabled=False):
        return F.layer_norm(x.float(), (x.shape[-1],), eps=1e-6)


def rope(x, seconds, scale):
    # x: [B,H,L,D], seconds: [B,L]. No flattened-token position encoding.
    with torch.autocast(x.device.type, enabled=False):
        freq = 10000.0 ** (-torch.arange(x.shape[-1] // 2, device=x.device).float()
                          / (x.shape[-1] // 2))
        phase = seconds.float()[:, None, :, None] * scale * freq
        a, b = x.float().chunk(2, -1)
        out = torch.cat((a * phase.cos() - b * phase.sin(),
                         a * phase.sin() + b * phase.cos()), -1)
    return out.to(x.dtype)


def timestep_features(t, width, scale):
    freq = torch.exp(-math.log(10000) * torch.arange(width // 2, device=t.device).float()
                     / (width // 2))
    phase = t.float()[..., None] * scale * freq
    return torch.cat((phase.cos(), phase.sin()), -1)


def attention_mask(frame, known, valid):
    """Leakage-safe [clean; noisy] mask for packed tokens.

A known clean token cannot absorb an unknown same-frame clean answer. This is
necessary when noisy queries may inspect the known subset of their own frame.
Unknown clean tokens may carry teacher-forced answers to strictly later frames.
"""
    earlier = frame[None, :] < frame[:, None]
    same = frame[None, :] == frame[:, None]
    cc = earlier[None] | (same[None] & (~known[:, :, None] | known[:, None, :]))
    nc = earlier[None] | (same[None] & known[:, None, :])
    nn_mask = same[None].expand(known.shape[0], -1, -1)
    top = torch.cat((cc, torch.zeros_like(cc)), -1)
    bottom = torch.cat((nc, nn_mask), -1)
    mask = torch.cat((top, bottom), -2)
    live = torch.cat((valid, valid), -1)
    mask = mask & live[:, :, None] & live[:, None, :]
    # Invalid queries attend only themselves; they are never valid keys for others.
    eye = torch.eye(mask.shape[-1], device=mask.device, dtype=torch.bool)
    mask = mask | ((~live)[:, :, None] & eye[None])
    return mask[:, None]


class Attention(nn.Module):
    def __init__(self, cfg, cross=False):
        super().__init__()
        self.cfg, self.cross = cfg, cross
        self.q = nn.Linear(cfg.width, cfg.width, bias=False)
        self.k = nn.Linear(cfg.width, cfg.width, bias=False)
        self.v = nn.Linear(cfg.width, cfg.width, bias=False)
        self.out = nn.Linear(cfg.width, cfg.width, bias=False)

    def split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.cfg.heads, -1).transpose(1, 2)

    def forward(self, x, times, mask=None, context=None, context_times=None, context_valid=None):
        memory = x if context is None else context
        q, k, v = self.split(self.q(x)), self.split(self.k(memory)), self.split(self.v(memory))
        if self.cfg.qk_norm:
            def norm(y):
                with torch.autocast(y.device.type, enabled=False):
                    z = y.float() * torch.rsqrt(y.float().square().mean(-1, keepdim=True) + 1e-6)
                return z.to(y.dtype)
            q, k = norm(q), norm(k)
        q = rope(q, times, self.cfg.time_scale)
        k = rope(k, times if context_times is None else context_times, self.cfg.time_scale)
        if self.cross:
            # Slot 0 is the null competitor. Its key MUST stay zero even after
            # projection/normalization/RoPE; its value is trainable.
            k = torch.cat((torch.zeros_like(k[:, :, :1]), k[:, :, 1:]), dim=2)
            mask = context_valid[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(out.transpose(1, 2).reshape_as(x))


class FlowBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attention = Attention(cfg)
        self.condition_attention = Attention(cfg, cross=True)
        self.up = nn.Linear(cfg.width, cfg.intermediate, bias=False)
        self.gate = nn.Linear(cfg.width, cfg.intermediate, bias=False)
        self.down = nn.Linear(cfg.intermediate, cfg.width, bias=False)
        # Wan-style learned per-layer offsets, not claimed to be adaLN-Zero.
        self.modulation = nn.Parameter(torch.randn(6, cfg.width) / math.sqrt(cfg.width))

    def forward(self, x, modulation, times, mask, condition, goal_times, goal_valid):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
            modulation + self.modulation[None, None]).unbind(-2)
        h = layer_norm(x) * (1 + scale_a) + shift_a
        x = x.float() + gate_a * self.self_attention(h, times, mask).float()
        x = x + self.condition_attention(layer_norm(x), times, context=condition,
                    context_times=goal_times, context_valid=goal_valid).float()
        h = layer_norm(x) * (1 + scale_f) + shift_f
        return x + gate_f * self.down(F.silu(self.gate(h)) * self.up(h)).float()


class FlowWorldModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or FlowConfig()
        c = self.cfg
        self.inputs = nn.ModuleList(nn.Linear(d, c.width) for d in c.dims)
        self.roles = nn.Parameter(torch.randn(3, c.width) * .02)
        self.time_embed = nn.Sequential(nn.Linear(c.time_features, c.width), nn.SiLU(),
                                        nn.Linear(c.width, c.width))
        self.time_project = nn.Sequential(nn.SiLU(), nn.Linear(c.width, 6 * c.width))
        self.goal_encoder = nn.Sequential(nn.Linear(c.goal_dim, c.width), nn.SiLU(),
                                          nn.Linear(c.width, c.width))
        self.goal_role = nn.Parameter(torch.randn(1, 1, c.width) * .02)
        self.null_condition = nn.Parameter(torch.zeros(1, 1, c.width))
        self.blocks = nn.ModuleList(FlowBlock(c) for _ in range(c.layers))
        self.head_modulation = nn.Parameter(torch.randn(3, 2, c.width) / math.sqrt(c.width))
        self.heads = nn.ModuleList(nn.Linear(c.width, d) for d in c.dims)

    def forward(self, clean, noisy, tau, times, known, valid, *, task='full',
                goals=None, goal_times=None, goal_valid=None, return_clean=False):
        """clean/noisy: three [B,T,D_m] tensors; tau: [B,T] or [B,T,3].

        task='lowdim' physically removes future image tokens from BOTH streams;
        its initial image remains a condition. goals are normalized joint positions
        (no velocity), [B,G,goal_dim]; goal_times use the same seconds as times.
        Caller supplies tau=1/clean inputs for known slots; forward enforces this.
        """
        if task not in ('full', 'lowdim'):
            raise ValueError(task)
        b, t = times.shape
        assert known.shape == valid.shape == (b, t, 3)
        assert known.dtype == valid.dtype == torch.bool
        assert not bool((known & ~valid).any())
        assert bool((times[:, 1:] > times[:, :-1]).all())
        for m, d in enumerate(self.cfg.dims):
            assert clean[m].shape == noisy[m].shape == (b, t, d)
        if tau.ndim == 2:
            tau = tau[..., None].expand(-1, -1, 3)
        assert tau.shape == (b, t, 3)
        assert bool(((tau >= 0) & (tau <= 1)).all())
        select = torch.ones(t, 3, device=times.device, dtype=torch.bool)
        if task == 'lowdim':
            select[1:, 0] = False
        flat = select.flatten().nonzero().flatten()
        frame, role = flat // 3, flat % 3
        packed_known = known.flatten(1)[:, flat]
        packed_valid = valid.flatten(1)[:, flat]
        pieces, positions = [], []
        for m in range(3):
            at = (role == m).nonzero().flatten()
            f = frame[at]
            live = packed_valid[:, at, None]
            a = torch.where(live, clean[m][:, f], 0)
            n = torch.where(packed_known[:, at, None], a, noisy[m][:, f])
            n = torch.where(live, n, 0)
            h = self.inputs[m](torch.cat((a, n), dim=1)) + self.roles[m]
            pieces.append(h)
            positions.append(torch.cat((at, at + flat.numel())))
        order = torch.cat(positions).argsort()
        x = torch.cat(pieces, dim=1)[:, order]
        noisy_tau = tau.flatten(1)[:, flat]
        noisy_tau = torch.where(packed_known | ~packed_valid, 1., noisy_tau)
        full_tau = torch.cat((torch.ones_like(noisy_tau), noisy_tau), dim=1)
        with torch.autocast(x.device.type, enabled=False):
            e = self.time_embed(timestep_features(full_tau, self.cfg.time_features,
                                                  self.cfg.noise_embedding_scale))
            modulation = self.time_project(e).reshape(b, -1, 6, self.cfg.width)
        token_times = torch.cat((times[:, frame], times[:, frame]), dim=1)
        mask = attention_mask(frame, packed_known, packed_valid)
        null = (self.null_condition + self.goal_role).expand(b, -1, -1)
        if goals is None:
            # Keep condition parameters in the graph for distributed task mixtures.
            goals = x.new_zeros(b, 1, self.cfg.goal_dim)
            goal_times = times[:, :1]
            goal_valid = torch.zeros(b, 1, device=x.device, dtype=torch.bool)
        assert goals.shape[:2] == goal_times.shape
        assert goals.shape[-1] == self.cfg.goal_dim
        if goal_valid is None:
            goal_valid = torch.ones_like(goal_times, dtype=torch.bool)
        safe_goals = torch.where(goal_valid[..., None], goals, 0)
        condition = torch.cat((null, self.goal_encoder(safe_goals) + self.goal_role), dim=1)
        condition_times = torch.cat((times[:, :1], torch.where(goal_valid, goal_times, times[:, :1])), 1)
        condition_valid = torch.cat((torch.ones(b, 1, device=x.device, dtype=torch.bool), goal_valid), 1)
        for block in self.blocks:
            args = (x, modulation, token_times, mask, condition, condition_times, condition_valid)
            x = checkpoint(block, *args, use_reentrant=False) if self.training and self.cfg.checkpointing else block(*args)
        length = flat.numel()
        clean_hidden = x[:, :length] if return_clean else None
        x = x[:, length:]
        e = e[:, length:]
        outputs = []
        for m in range(3):
            at = (role == m).nonzero().flatten()
            shift, scale = (e[:, at, None, :] + self.head_modulation[m]).unbind(-2)
            v = self.heads[m](layer_norm(x[:, at]) * (1 + scale) + shift)
            outputs.append(v.new_zeros(b, t, self.cfg.dims[m]).index_copy(1, frame[at], v))
        loss_mask = valid & ~known & select[None]
        return FlowOutput(tuple(outputs), loss_mask, clean_hidden)


def make_noisy(clean, tau, known, valid, noise=None):
    if tau.ndim == 2:
        tau = tau[..., None].expand(-1, -1, 3)
    noise = tuple(torch.randn_like(x) for x in clean) if noise is None else noise
    noisy, targets = [], []
    for m, x in enumerate(clean):
        x = torch.where(valid[..., m, None], x, 0)
        eps = noise[m]
        s = tau[..., m, None]
        z = s * x + (1-s) * eps
        z = torch.where(known[..., m, None], x, z)
        noisy.append(torch.where(valid[..., m, None], z, 0))
        targets.append(torch.where((valid & ~known)[..., m, None], x-eps, 0))
    return tuple(noisy), tuple(targets)


def flow_loss(output, targets, weights=(1., 1., 1.)):
    # Per-sample / per-modality means: image dimension cannot silently dominate.
    losses, active = [], []
    for m, pred in enumerate(output.velocity):
        mask = output.loss_mask[..., m]
        delta = torch.where(mask[..., None], pred.float()-targets[m].float(), 0.)
        mse = delta.square().mean(-1)
        losses.append((mse.sum(-1) / mask.sum(-1).clamp_min(1)) * weights[m])
        active.append(mask.any(-1).float() * weights[m])
    return (torch.stack(losses, -1).sum(-1) /
            torch.stack(active, -1).sum(-1).clamp_min(1e-8)).mean()
