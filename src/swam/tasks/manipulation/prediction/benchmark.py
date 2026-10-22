"""Real-input latency including encoding, history updates and action decoding."""
import gc
import json
import time

import numpy as np
import torch

from swam.flow_inference import FlowInference, frozen_inference_model
from swam.flow_sampling import next_frame
from swam.flow_tasks import prepare_window
from swam.shards import atomic_json, sha256
from .codec import InputCodec, load_model


def stats(values):
    values = np.asarray(values)
    return dict(mean_ms=float(values.mean()), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), hz=1000 / float(values.mean()),
                below_24hz_budget_fraction=float((values <= 1000 / 24).mean()))


def measure(operation, reset, repeats, warmup):
    for _ in range(warmup):
        reset()
        operation()
    durations = []
    for _ in range(repeats):
        reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = operation()
        torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    return dict(stats(durations), samples_ms=durations), output


class BenchmarkCase:
    def __init__(self, model, fast, codec, sample, count):
        self.model = model
        self.codec = codec
        self.count = count
        self.sample = sample

        # Sample the camera and goals once for this recorded window.
        batch = {}
        for name, value in sample['latents'].items():
            batch[name] = value[None].cuda()
        generator = torch.Generator(device='cuda').manual_seed(1729)
        inputs, _, metadata = prepare_window(**batch, task='full', generator=generator)
        keep = inputs['goal_valid'][0]
        self.goals = inputs['goals'][:, keep]
        self.goal_times = inputs['goal_times'][:, keep]
        self.times = inputs['times']
        self.prefix = tuple(value[:, :count - 1] for value in inputs['clean'])
        self.current_time = self.times[:, count - 1:count]
        self.next_time = self.times[:, count:count + 1]

        # Time one recorded control tick, retaining the preceding cached history.
        camera = int(metadata['camera'][0])
        raw = sample['raw']
        self.rgb = raw['rgb'][count - 1, camera].clone()
        state = torch.cat((raw['qpos'][count - 1], raw['qvel'][count - 1]))
        self.state = state[None].cuda().float()
        self.action = raw['actions'][count - 1][None].cuda().float()
        self.engine = FlowInference(fast, self.goals, self.goal_times, max_frames=121)
        self.prefill_ms = self.prefill()

    def prefill(self):
        start = time.perf_counter()
        for frame in range(self.count - 1):
            latent = tuple(value[:, frame:frame + 1] for value in self.prefix)
            self.engine.append(latent, self.times[:, frame:frame + 1])
        encoded = self.codec.encode(self.rgb, self.state, self.action)
        self.engine.append(encoded, self.current_time)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000

    def reset(self):
        # Mask stale cache entries; append overwrites them without rebuilding the prefix.
        length = (self.count - 1) * 3
        self.engine.length = length
        self.engine.frames = self.count - 1
        self.engine.live[:, length:] = False
        previous = max(0, self.count - 2)
        self.engine.last_time = self.times[:, previous:previous + 1].clone()

    def record_stage(self, stages, name, start):
        if stages is None:
            return start
        torch.cuda.synchronize()
        end = time.perf_counter()
        stages[name] = (end - start) * 1000
        return end

    def operation(self, mode, steps, generator, stages=None):
        generator.manual_seed(731)
        if stages is not None:
            torch.cuda.synchronize()
        start = time.perf_counter() if stages is not None else 0.
        current = self.codec.encode(self.rgb, self.state, self.action)
        start = self.record_stage(stages, 'encode_ms', start)

        # Reference execution rebuilds history; cached modes commit only this frame.
        if mode == 'reference':
            history = tuple(torch.cat((old, new), 1) for old, new in zip(self.prefix, current))
        else:
            self.engine.append(current, self.current_time)
        start = self.record_stage(stages, 'history_ms', start)
        if mode == 'reference':
            result = next_frame(self.model, history, self.times[:, :self.count + 1],
                                self.goals, self.goal_times, steps=steps, generator=generator)
        else:
            result = self.engine.predict(self.next_time, steps=steps, generator=generator,
                                         use_graph=mode == 'cuda_graph')
        start = self.record_stage(stages, 'sampling_ms', start)
        decoded = self.codec.decode(result[2])
        self.record_stage(stages, 'decode_and_transfer_ms', start)
        return decoded

    def measure_mode(self, mode, steps, settings):
        self.engine.append_graphs.clear()
        if mode == 'cuda_graph':
            self.engine.capture_append()
        generator = torch.Generator(device='cuda')
        operation = lambda: self.operation(mode, steps, generator)
        timings, predicted = measure(operation, self.reset, settings.repeats, settings.warmup)

        # Synchronized stage measurements are separate from end-to-end samples.
        stage_rows = []
        for _ in range(5):
            self.reset()
            stages = {}
            self.operation(mode, steps, generator, stages)
            stage_rows.append(stages)
        means = {}
        for name in stage_rows[0]:
            means[name] = float(np.mean([row[name] for row in stage_rows]))
        return timings, predicted, means

    def measure_steps(self, steps, settings):
        start = time.perf_counter()
        self.engine.capture(steps)
        torch.cuda.synchronize()
        capture_ms = (time.perf_counter() - start) * 1000
        baseline = None
        for mode in ('reference', 'cached', 'cuda_graph'):
            timings, predicted, stages = self.measure_mode(mode, steps, settings)
            if baseline is None:
                baseline = predicted
            difference = predicted - baseline
            yield dict(sample=self.sample['label'], history_frames=self.count,
                       steps=steps, mode=mode, **timings, stages=stages,
                       prefill_ms=self.prefill_ms, capture_ms=capture_ms,
                       action_difference_rmse_rad=float(difference.square().mean().sqrt()),
                       action_difference_max_rad=float(difference.abs().max()),
                       peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        del self.engine.graphs[steps, False]


def selected_samples(reference):
    selected = []
    splits = set()
    for name, sample in reference['samples'].items():
        split = sample['metadata']['split']
        if split in splits:
            continue
        selected.append(dict(sample, label=name))
        splits.add(split)
    return selected[:2]


def run_case(case, settings, report, destination):
    for steps in settings.steps:
        for row in case.measure_steps(steps, settings):
            report['rows'].append(row)
            atomic_json(destination, report)
            print(json.dumps(row), flush=True)


@torch.inference_mode()
def run(experiment, output):
    output.mkdir(parents=True, exist_ok=False)
    paths = experiment.paths
    settings = experiment.benchmark
    torch.set_num_threads(4)
    model = load_model(paths.checkpoint, settings.expected_step)
    fast = frozen_inference_model(model)
    reference = torch.load(paths.reference, map_location='cpu', weights_only=False)
    codec = InputCodec(reference, paths.tokenizers, paths.image_model)
    report = dict(checkpoint_sha256=sha256(paths.checkpoint),
                  reference_sha256=sha256(paths.reference), checkpoint_step=settings.expected_step,
                  device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  repeats=settings.repeats, warmup=settings.warmup, task='full', batch_size=1,
                  includes='RGB transfer, encoding, history update, sampling, action decode and CPU transfer',
                  excludes='Sensing, simulation, disk I/O, checkpoint loading, cache prefill and graph capture',
                  sampling='Euler tau=k/N; seed 731; unchanged checkpoint', rows=[], complete=False)

    # Every case is independent, while each measured tick reuses its history cache.
    for sample in selected_samples(reference):
        for count in settings.history:
            case = BenchmarkCase(model, fast, codec, sample, count)
            run_case(case, settings, report, output / 'results.json')
            del case
            gc.collect()
            torch.cuda.empty_cache()
    report['complete'] = True
    atomic_json(output / 'results.json', report)
    return report
