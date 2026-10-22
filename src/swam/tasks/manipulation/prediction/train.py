"""Distributed flow fitting with explicit data, checkpoints and local logging."""
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.attention import sdpa_kernel, SDPBackend

from swam.flow_data import GrowingWindows
from swam.flow_model import FlowConfig, FlowWorldModel, flow_loss
from swam.flow_tasks import prepare_window
from swam.shards import atomic_json


def learning_rate(step, base_lr, steps):
    # Preserve the original 200-step warmup and global cosine schedule on resume.
    return base_lr * min(1., step / 200) * (
        .1 + .9 * .5 * (1 + math.cos(math.pi * max(0, step - 200) / max(1, steps - 200))))


def sample_seed(step, rank, microbatch, world, accumulation, seed=42):
    virtual_rank = rank + microbatch * world
    return seed + step * world * accumulation + virtual_rank


def task_for_step(step):
    return 'full' if random.Random(1234 + step).random() < .5 else 'lowdim'


def initialize_run(experiment, rank, world):
    settings = experiment.train
    if rank == 0:
        experiment.paths.output.mkdir(parents=True, exist_ok=False)
        (experiment.paths.output / 'experiment.toml').write_bytes(experiment.source.read_bytes())
    if world > 1:
        dist.barrier()
    if rank != 0 or settings.wandb_mode == 'disabled':
        return None

    # Remote logging is optional and never assumes an account or an old run ID.
    import wandb
    config = asdict(settings)
    config['global_batch'] = settings.batch * world * settings.accumulation
    return wandb.init(project=settings.wandb_project, entity=settings.wandb_entity,
                      mode=settings.wandb_mode, config=config,
                      dir=str(experiment.paths.output), job_type='train')


class Trainer:
    def __init__(self, experiment, rank, world, local):
        self.experiment = experiment
        self.settings = experiment.train
        self.rank = rank
        self.world = world
        self.output = experiment.paths.output
        self.first = 0

        # Model initialization and resume precede the first step-specific random seed.
        torch.manual_seed(self.settings.seed)
        self.config = FlowConfig(**experiment.model)
        self.raw = FlowWorldModel(self.config).cuda()
        self.optimizer = self.make_optimizer()
        self.restore()
        self.model = self.raw
        if world > 1:
            self.model = DistributedDataParallel(self.raw, device_ids=[local],
                gradient_as_bucket_view=True, bucket_cap_mb=25)
        self.data = GrowingWindows(experiment.paths.latents,
            seed=self.settings.seed + rank, frames=self.settings.frames)

    def make_optimizer(self):
        if self.settings.optimizer == 'adamw8bit':
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit
        else:
            optimizer = torch.optim.AdamW
        return optimizer(self.raw.parameters(), lr=self.settings.base_lr, weight_decay=.01)

    def restore(self):
        path = self.experiment.paths.checkpoint
        if path is None:
            if self.settings.expected_step:
                raise ValueError('A nonzero expected_step requires a full training checkpoint')
            return
        checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if checkpoint['config'] != asdict(self.config):
            raise ValueError('Model configuration differs from the checkpoint')
        if checkpoint['step'] != self.settings.expected_step:
            raise ValueError(f"Expected step {self.settings.expected_step}, got {checkpoint['step']}")
        if 'optimizer' not in checkpoint:
            raise ValueError('Training resume requires optimizer state, not model-only weights')
        self.raw.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.first = checkpoint['step']
        self.tokenizer_hashes = checkpoint.get('tokenizer_sha256')

    def microbatch(self, step, micro, task):
        # Keep per-rank sampling and PyTorch random-number consumption unchanged.
        seed = sample_seed(step, self.rank, micro, self.world,
                           self.settings.accumulation, self.settings.seed)
        self.data.rng.seed(seed)
        torch.manual_seed(seed)
        batch, metadata = self.data.batch(self.settings.batch)
        for name, value in batch.items():
            batch[name] = value.cuda(non_blocking=True)
        inputs, target, _ = prepare_window(**batch, task=task)
        sync = nullcontext()
        if self.world > 1 and micro < self.settings.accumulation - 1:
            sync = self.model.no_sync()
        with sync:
            loss = self.forward_loss(inputs, target)
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                (loss / self.settings.accumulation).backward()
        return loss.detach() / self.settings.accumulation, metadata

    def forward_loss(self, inputs, target):
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]), torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = self.model(**inputs)
            loss = flow_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        return loss

    def update(self, step):
        started = time.monotonic()
        self.data.refresh()
        task = task_for_step(step)
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device='cuda')
        metadata = []
        for micro in range(self.settings.accumulation):
            loss, samples = self.microbatch(step, micro, task)
            loss_sum += loss
            metadata.extend(samples)

        # Apply the original clipping, schedule and optimizer order.
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        if not torch.isfinite(grad):
            raise RuntimeError('Nonfinite gradient')
        rate = learning_rate(step, self.settings.base_lr, self.settings.steps)
        for group in self.optimizer.param_groups:
            group['lr'] = rate
        self.optimizer.step()
        torch.cuda.synchronize()
        average = loss_sum.clone()
        if self.world > 1:
            dist.all_reduce(average)
        average /= self.world
        row = dict(step=step, task=task, loss=float(loss_sum), global_loss=float(average),
                   gradient_norm=float(grad), lr=rate, seconds=time.monotonic() - started,
                   samples=metadata, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        self.optimizer.zero_grad(set_to_none=True)
        return row

    def save_checkpoint(self, step):
        if self.world > 1:
            dist.barrier()
        if self.rank == 0:
            training = asdict(self.settings)
            training['global_batch'] = self.settings.batch * self.world * self.settings.accumulation
            checkpoint = dict(model=self.raw.state_dict(), optimizer=self.optimizer.state_dict(),
                step=step, config=asdict(self.config), training=training,
                tokenizer_sha256=self.data.norm['tokenizer_sha256'])
            torch.save(checkpoint, self.output / 'latest.partial')
            (self.output / 'latest.partial').replace(self.output / 'latest.pt')
        if self.world > 1:
            dist.barrier()

    def record(self, row, log, run):
        log.write(json.dumps(row) + '\n')
        if self.rank != 0:
            return
        atomic_json(self.output / 'status.json', dict(stage='training', updated=time.time(), **row))
        print(json.dumps(row), flush=True)
        if run is not None:
            run.log({'Results/Loss': row['global_loss'],
                     'Diagnostics/Gradient norm': row['gradient_norm'],
                     'Diagnostics/Update seconds': row['seconds']}, step=row['step'])

    def fit(self, run):
        self.model.train()
        expected = getattr(self, 'tokenizer_hashes', None)
        if expected is not None and expected != self.data.norm['tokenizer_sha256']:
            raise ValueError('Tokenizer hashes differ from the resumed checkpoint')
        with (self.output / f'metrics_rank{self.rank}.jsonl').open('a', buffering=1) as log:
            self.fit_steps(log, run)
        if self.rank == 0:
            atomic_json(self.output / 'status.json', dict(stage='complete',
                        step=self.settings.steps, updated=time.time()))

    def fit_steps(self, log, run):
        for step in range(self.first + 1, self.settings.steps + 1):
            row = self.update(step)
            self.record(row, log, run)
            if step % self.settings.checkpoint_every == 0 or step == self.settings.steps:
                self.save_checkpoint(step)


def run(experiment):
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if world != experiment.train.workers:
        raise ValueError(f'Configured {experiment.train.workers} workers, launched {world}')
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA; use doctor to inspect the installation')

    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    logging_run = None
    try:
        logging_run = initialize_run(experiment, rank, world)
        trainer = Trainer(experiment, rank, world, local)
        trainer.fit(logging_run)
    finally:
        if logging_run is not None:
            logging_run.finish()
        if dist.is_initialized():
            dist.destroy_process_group()
