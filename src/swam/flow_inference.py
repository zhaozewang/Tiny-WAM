"""Inference-only KV cache for the unchanged frame-causal flow checkpoint.

Only observed/committed frames enter the cache. Denoised predictions do not
become history until the caller explicitly appends them. Training stays in
flow_model; flow_sampling.next_frame remains the uncached reference.
"""
from contextlib import nullcontext
import copy

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from swam.flow_model import layer_norm, timestep_features


def rotary_terms(seconds, width, scale):
    frequency = 10000.0 ** (-torch.arange(width // 2, device=seconds.device).float() / (width // 2))
    phase = seconds.float()[:, None, :, None] * scale * frequency
    return phase.cos(), phase.sin()


def rotate(x, terms):
    cosine, sine = terms
    left, right = x.float().chunk(2, -1)
    return torch.cat((left * cosine - right * sine, left * sine + right * cosine), -1).to(x.dtype)


def normalize_qk(x, enabled):
    if not enabled:
        return x
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


def frozen_inference_model(model, dtype=torch.bfloat16):
    """Copy autocast linear weights once; retain FP32 residuals and time MLPs.

    The reference sampler already rounds these linear weights under autocast.
    Keeping that representation avoids repeating the casts inside CUDA graphs.
    The original model and its checkpoint are untouched.
    """
    result = copy.deepcopy(model).eval().requires_grad_(False)
    for name, module in result.named_modules():
        if isinstance(module, torch.nn.Linear) and not name.startswith(('time_embed', 'time_project')):
            module.to(dtype=dtype)
    return result


class FlowInference:
    """One fixed-condition stream, reset by constructing a new instance.

    The model must remain in eval mode with unchanged weights. max_frames is
    a capacity, not a sliding window: exceeding it raises instead of truncating.
    CUDA graph capture is optional and does not change the Euler schedule.
    """

    @torch.inference_mode()
    def __init__(self, model, goals=None, goal_times=None, *, task='full', max_frames=121,
                 dtype=torch.bfloat16):
        if model.training:
            raise ValueError('Inference requires model.eval()')
        if task not in ('full', 'lowdim'):
            raise ValueError(task)
        if max_frames < 1:
            raise ValueError('max_frames must be positive')
        if (goals is None) != (goal_times is None):
            raise ValueError('Supply both goals and goal_times')
        self.model = model
        self.task = task
        self.max_frames = max_frames
        self.dtype = dtype
        self.goals = goals
        self.goal_times = goal_times
        self.frames = 0
        self.length = 0
        self.cache = []
        self.condition_cache = []
        self.schedules = {}
        self.graphs = {}
        self.append_graphs = {}
        self.initialized = False

    def precision(self):
        device = next(self.model.parameters()).device
        if self.dtype is None:
            return nullcontext()
        return torch.autocast(device.type, dtype=self.dtype)

    def backend(self):
        if next(self.model.parameters()).is_cuda:
            return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
        return nullcontext()

    def embeddings(self, tau):
        cfg = self.model.cfg
        with torch.autocast(tau.device.type, enabled=False):
            value = self.model.time_embed(timestep_features(tau, cfg.time_features, cfg.noise_embedding_scale))
            modulation = self.model.time_project(value).unflatten(-1, (6, cfg.width))
        return value, modulation

    def initialize(self, history, seconds):
        model, cfg = self.model, self.model.cfg
        batch, device = history[0].shape[0], history[0].device
        self.origin = seconds[:, :1].clone()
        null = (model.null_condition + model.goal_role).expand(batch, -1, -1)
        if self.goals is None:
            condition = null
            condition_times = self.origin
        else:
            if self.goals.shape[:2] != self.goal_times.shape:
                raise ValueError('Goal and time shapes differ')
            condition = torch.cat((null, model.goal_encoder(self.goals) + model.goal_role), 1)
            condition_times = torch.cat((self.origin, self.goal_times), 1)
        terms = rotary_terms(condition_times, cfg.width // cfg.heads, cfg.time_scale)
        capacity = self.max_frames * 3
        self.live = torch.zeros(batch, capacity, dtype=torch.bool, device=device)
        for block in model.blocks:
            attn = block.condition_attention
            key = normalize_qk(attn.split(attn.k(condition)), cfg.qk_norm)
            key = rotate(key, terms)
            key[:, :, :1] = 0
            value = attn.split(attn.v(condition))
            self.condition_cache.append((key, value))
            shape = (batch, cfg.heads, capacity, cfg.width // cfg.heads)
            self.cache.append((key.new_zeros(shape), value.new_zeros(shape)))
        _, self.clean_modulation = self.embeddings(torch.ones(batch, 1, device=device))
        self.initialized = True

    def encode(self, latents, roles):
        pieces = [self.model.inputs[role](latents[role]) + self.model.roles[role] for role in roles]
        return torch.cat(pieces, dim=1)

    def blocks(self, x, modulation, terms, mask, *, append=False, pending=None):
        cfg = self.model.cfg
        for index, block in enumerate(self.model.blocks):
            shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
                modulation + block.modulation[None, None]).unbind(-2)
            h = layer_norm(x) * (1 + scale_a) + shift_a
            attn = block.self_attention
            query = rotate(normalize_qk(attn.split(attn.q(h)), cfg.qk_norm), terms)
            key = rotate(normalize_qk(attn.split(attn.k(h)), cfg.qk_norm), terms)
            value = attn.split(attn.v(h))
            old_key, old_value = self.cache[index]
            output = F.scaled_dot_product_attention(query, torch.cat((old_key, key), 2),
                                                    torch.cat((old_value, value), 2), attn_mask=mask)
            output = attn.out(output.transpose(1, 2).reshape_as(x))
            x = x.float() + gate_a * output.float()
            if append:
                stop = self.length + key.shape[2]
                old_key[:, :, self.length:stop].copy_(key)
                old_value[:, :, self.length:stop].copy_(value)
            if pending is not None:
                pending[index][0].copy_(key)
                pending[index][1].copy_(value)
            attn = block.condition_attention
            query = rotate(normalize_qk(attn.split(attn.q(layer_norm(x))), cfg.qk_norm), terms)
            key, value = self.condition_cache[index]
            output = F.scaled_dot_product_attention(query, key, value)
            x = x + attn.out(output.transpose(1, 2).reshape_as(x)).float()
            h = layer_norm(x) * (1 + scale_f) + shift_f
            x = x + gate_f * block.down(F.silu(block.gate(h)) * block.up(h)).float()
        return x

    def frame(self, latents, seconds, roles, *, append=False, pending=None):
        with self.precision(), self.backend():
            x = self.encode(latents, roles)
            mask = torch.cat((self.live, torch.ones_like(self.live[:, :len(roles)])), -1)[:, None, None]
            terms = rotary_terms(seconds, self.model.cfg.width // self.model.cfg.heads, self.model.cfg.time_scale)
            self.blocks(x, self.clean_modulation, terms, mask, append=append, pending=pending)

    @torch.inference_mode()
    def append(self, latents, seconds):
        """Append exactly one measured frame and its committed action."""
        if self.frames >= self.max_frames:
            raise ValueError('History capacity exceeded; reset the stream explicitly')
        if any(x.shape[1] != 1 for x in latents) or seconds.shape != latents[0].shape[:2]:
            raise ValueError('append expects one frame per batch element')
        if self.initialized and self.frames and not bool((seconds > self.last_time).all()):
            raise ValueError('History timestamps must increase')
        with self.precision(), self.backend():
            if not self.initialized:
                self.initialize(latents, seconds)
        roles = (0, 1, 2) if self.task == 'full' or self.frames == 0 else (1, 2)
        if roles in self.append_graphs:
            graph, buffers, time_buffer, pending = self.append_graphs[roles]
            for target, value in zip(buffers, latents):
                target.copy_(value)
            time_buffer.copy_(seconds)
            graph.replay()
            # The graph reads the live prefix. Commit new K/V after replay,
            # keeping the append location dynamic as the stream grows.
            for (key, value), (new_key, new_value) in zip(self.cache, pending):
                key[:, :, self.length:self.length + len(roles)].copy_(new_key)
                value[:, :, self.length:self.length + len(roles)].copy_(new_value)
        else:
            self.frame(latents, seconds, roles, append=True)
        self.live[:, self.length:self.length + len(roles)] = True
        self.length += len(roles)
        self.frames += 1
        self.last_time = seconds.clone()

    @torch.inference_mode()
    def capture_append(self):
        """Capture measured-frame processing without committing dummy history."""
        if not self.initialized or not self.live.is_cuda:
            raise ValueError('CUDA capture requires an initialized CUDA stream')
        roles = (0, 1, 2) if self.task == 'full' else (1, 2)
        buffers = tuple(torch.zeros(self.live.shape[0], 1, dim, device=self.live.device)
                        for dim in self.model.cfg.dims)
        seconds = self.last_time.clone()
        pending = [(torch.empty_like(k[:, :, :len(roles)]), torch.empty_like(v[:, :, :len(roles)]))
                   for k, v in self.cache]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.frame(buffers, seconds, roles, pending=pending)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.frame(buffers, seconds, roles, pending=pending)
        self.append_graphs[roles] = graph, buffers, seconds, pending

    def schedule(self, steps):
        if steps not in self.schedules:
            device = self.live.device
            self.schedules[steps] = [self.embeddings(torch.full((self.live.shape[0], 1), k / steps, device=device))
                                     for k in range(steps)]
        return self.schedules[steps]

    def integrate(self, noise, seconds, steps, terminal):
        roles = tuple(m for m in range(3) if not (m == 0 and self.task == 'lowdim')
                      and not (m == 2 and terminal))
        z = list(noise)
        terms = rotary_terms(seconds, self.model.cfg.width // self.model.cfg.heads, self.model.cfg.time_scale)
        mask = torch.cat((self.live, torch.ones_like(self.live[:, :len(roles)])), -1)[:, None, None]
        with self.precision(), self.backend():
            for embedding, modulation in self.schedule(steps):
                x = self.blocks(self.encode(z, roles), modulation, terms, mask)
                for index, role in enumerate(roles):
                    shift, scale = (embedding[:, :, None] + self.model.head_modulation[role]).unbind(-2)
                    hidden = layer_norm(x[:, index:index + 1]) * (1 + scale) + shift
                    velocity = self.model.heads[role](hidden)
                    z[role] = z[role] + velocity.float() / steps
        if self.task == 'lowdim':
            z[0] = torch.zeros_like(z[0])
        if terminal:
            z[2] = torch.zeros_like(z[2])
        return tuple(z)

    @torch.inference_mode()
    def capture(self, steps=8, terminal=False):
        """Capture a fixed Euler schedule; history buffers remain live between calls."""
        if not self.initialized or not self.live.is_cuda:
            raise ValueError('CUDA capture requires an initialized CUDA stream')
        if steps < 1:
            raise ValueError('steps must be positive')
        key = steps, terminal
        noise = tuple(torch.zeros(self.live.shape[0], 1, dim, device=self.live.device) for dim in self.model.cfg.dims)
        seconds = self.last_time + 1 / 24
        self.schedule(steps)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.integrate(noise, seconds, steps, terminal)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.integrate(noise, seconds, steps, terminal)
        self.graphs[key] = graph, noise, seconds, output

    @torch.inference_mode()
    def predict(self, seconds, *, steps=8, generator=None, terminal=False, use_graph=False):
        if not self.initialized or steps < 1:
            raise ValueError('Append history first and use positive steps')
        if seconds.shape != self.last_time.shape or not bool((seconds > self.last_time).all()):
            raise ValueError('Prediction timestamps must follow the latest history frame')
        noise = tuple(torch.randn(self.live.shape[0], 1, dim, device=self.live.device, generator=generator)
                      for dim in self.model.cfg.dims)
        if use_graph:
            graph, buffers, time_buffer, output = self.graphs[steps, terminal]
            for target, value in zip(buffers, noise):
                target.copy_(value)
            time_buffer.copy_(seconds)
            graph.replay()
            result = tuple(value.clone() for value in output)
        else:
            result = self.integrate(noise, seconds, steps, terminal)
        if not bool(torch.stack([value.isfinite().all() for value in result]).all()):
            raise RuntimeError('Non-finite rollout')
        return result


@torch.inference_mode()
def cached_rollout(model, initial, times, goals, goal_times, *, task='full', steps=8,
                   generator=None, progress=None, use_graph=False, dtype=torch.bfloat16):
    """Open-loop rollout from exactly one initial frame, matching flow_sampling.

    Prepare frozen_inference_model once before calling for CUDA graph execution.
    This function commits predictions as open-loop history. For real feedback,
    use FlowInference directly and append actual observations/committed actions.
    """
    if len(initial) != 3 or any(x.shape[1] != 1 for x in initial):
        raise ValueError('Supply exactly one initial frame of all three modalities')
    if times.ndim != 2 or times.shape[1] < 1 or times.shape[0] != initial[0].shape[0]:
        raise ValueError('Times must contain the initial frame and prediction timestamps')
    history = tuple(value.clone() for value in initial)
    if times.shape[1] == 1:
        return history
    engine = FlowInference(model, goals, goal_times, task=task,
                           max_frames=times.shape[1], dtype=dtype)
    engine.append(initial, times[:, :1])
    if use_graph:
        engine.capture_append()
        if times.shape[1] > 2:
            engine.capture(steps)
        engine.capture(steps, terminal=True)
    for frame in range(1, times.shape[1]):
        terminal = frame == times.shape[1] - 1
        value = engine.predict(times[:, frame:frame + 1], steps=steps, generator=generator,
                               terminal=terminal, use_graph=use_graph)
        history = tuple(torch.cat((past, now), 1) for past, now in zip(history, value))
        if not terminal:
            engine.append(value, times[:, frame:frame + 1])
        if progress:
            progress(frame)
    return history
