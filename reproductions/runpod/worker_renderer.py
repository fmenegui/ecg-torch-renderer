"""Bounded GPU profile of the training renderer on one rented card.

Measures the current renderer, not a historical version: warm latency and
images/s per preset and batch, eager versus compiled, cold compilation, peak
allocated memory, CUDA kernel launches per forward, CPU/GPU numerical parity
under a deterministic configuration, the TF32 effect, and how a failed
on-device validation behaves. No patient data is used: signals are artificial.
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from ecg_gpu import ECGImageLayer, ECGSupervisionLayer, RenderConfig, SupervisionConfig
from ecg_gpu.audit import PRESETS
from ecg_gpu.demo import demo_signals
from ecg_gpu.gpu_renderer import ECGRenderer

ALL_FLAGS = dict(page_model='physical', stroke='isotropic', lead_marks=True, scan_modes=True,
                 device_filters=True, report_text=True, phone_isp=True, coordinate_targets=True,
                 print_technology=True, archive_damage=True, page_corners=True, calibration_target=True,
                 office_marks=True, perspective_relief=True, text_target=True)
CUDA_TEST_MODULES = ['tests.test_renderer', 'tests.test_supervision', 'tests.test_paper',
                     'tests.test_grid_points', 'tests.test_training_adapters', 'tests.test_realism_v090']


class Deadline:
    def __init__(self, seconds_left):
        self.end = time.time() + seconds_left

    def left(self):
        return self.end - time.time()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def signals(batch, device):
    values, fs, _ = demo_signals(fs=500, duration=10)
    return torch.tensor(values, dtype=torch.float32, device=device)[None].expand(batch, -1, -1).contiguous(), fs


def time_layer(layer, sig, warmup, steps):
    torch.cuda.synchronize()
    start = time.perf_counter()
    layer(sig)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - start) * 1e3
    for _ in range(warmup):
        layer(sig)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    latencies = []
    for _ in range(steps):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        layer(sig)
        end.record()
        end.synchronize()
        latencies.append(begin.elapsed_time(end))
    latencies.sort()
    median = latencies[len(latencies) // 2]
    return {'first_call_ms': first_ms, 'median_ms': median, 'p90_ms': latencies[int(.9 * (len(latencies) - 1))],
            'images_per_s': sig.shape[0] / median * 1e3,
            'peak_additional_allocated_mib': (torch.cuda.max_memory_allocated() - base) / 2**20}


def throughput(deadline, fs_device):
    rows = []
    cases = []
    for preset in ['fast', 'balanced', 'full']:
        for batch in [1, 4, 16]:
            for compiled in [False, True]:
                cases.append((preset, dict(PRESETS[preset]), batch, compiled, 'image'))
    cases += [('full+all_flags', {**PRESETS['full'], **ALL_FLAGS}, batch, False, 'image') for batch in [1, 4]]
    cases += [('balanced', dict(PRESETS['balanced']), batch, False, 'supervision') for batch in [4, 16]]
    for name, config, batch, compiled, kind in cases:
        if deadline.left() < 420:
            rows.append({'preset': name, 'batch': batch, 'compiled': compiled, 'kind': kind, 'skipped': 'deadline'})
            continue
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        row = {'preset': name, 'batch': batch, 'compiled': compiled, 'kind': kind, 'config': config}
        try:
            sig, fs = signals(batch, fs_device)
            options = dict(compile_raster=compiled, compile_scene=compiled)
            if kind == 'supervision':
                layer = ECGSupervisionLayer(fs=fs, config=RenderConfig(**config), supervision=SupervisionConfig(
                    outputs=('lead_masks', 'instance_boxes')), **options).cuda()
                call = layer
            else:
                layer = ECGImageLayer(fs=fs, config=RenderConfig(**config), **options).cuda()
                call = layer
            steps = 20 if name in ('fast', 'balanced') else 10
            row.update(time_layer(call, sig, warmup=3, steps=steps))
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {str(error)[:400]}'
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ['preset', 'batch', 'compiled', 'kind', 'median_ms', 'images_per_s',
                                                  'peak_additional_allocated_mib', 'error']}), flush=True)
        del layer
    return rows


def kernel_launches(device):
    from torch.profiler import profile, ProfilerActivity
    result = []
    for name, config, batch in [('balanced', PRESETS['balanced'], 4), ('balanced', PRESETS['balanced'], 16),
                                ('full', PRESETS['full'], 4), ('full+all_flags', {**PRESETS['full'], **ALL_FLAGS}, 4)]:
        sig, fs = signals(batch, device)
        layer = ECGImageLayer(fs=fs, config=RenderConfig(**config)).cuda()
        layer(sig)
        torch.cuda.synchronize()
        wall = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            layer(sig)
            torch.cuda.synchronize()
        wall = (time.perf_counter() - wall) * 1e3
        events = prof.events()
        kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
        busy_ms = sum(e.time_range.elapsed_us() for e in kernels) / 1e3
        result.append({'preset': name, 'batch': batch, 'cuda_kernels': len(kernels),
                       'aten_ops': sum(1 for e in events if e.name.startswith('aten::')),
                       'gpu_busy_ms': busy_ms, 'profiled_wall_ms': wall,
                       'gpu_busy_fraction': busy_ms / wall if wall else None})
        print(json.dumps(result[-1]), flush=True)
        del layer
    return result


def parity(device):
    """Same sampled parameters on CPU and GPU, deterministic observation path."""
    out = {}
    config = RenderConfig(height=384, width=512, profile='clean', camera_mode='direct', jpeg_enabled=False)
    sig, fs = signals(2, 'cpu')
    cpu = ECGRenderer(config)
    gpu = ECGRenderer(config).to(device)
    params = cpu.sample_parameters(2, generator=torch.Generator().manual_seed(5), photo=False)
    a = cpu(sig, fs, stage='all', photo=False, parameters=params, generator=torch.Generator().manual_seed(9))
    b = gpu(sig.to(device), fs, stage='all', photo=False, parameters={k: v.to(device) for k, v in params.items()},
            generator=torch.Generator(device=device).manual_seed(9))
    out['clean_fixed_parameters'] = {
        'image_max_abs': float((a.image - b.image.cpu()).abs().max()),
        'image_mean_abs': float((a.image - b.image.cpu()).abs().mean()),
        'targets_max_abs': {k: float((v - b.targets[k].cpu()).abs().max()) for k, v in a.targets.items()
                            if torch.is_tensor(v) and v.is_floating_point()}}
    # Same seed, same mixed defaults: the two generators produce different captures.
    layer_cpu = ECGImageLayer(fs=fs, config=RenderConfig(height=384, width=512), seed=7, randomize=False).eval()
    layer_gpu = ECGImageLayer(fs=fs, config=RenderConfig(height=384, width=512), seed=7, randomize=False).eval().to(device)
    x, y = layer_cpu(sig), layer_gpu(sig.to(device)).cpu()
    out['mixed_same_seed'] = {'image_max_abs': float((x - y).abs().max()), 'image_mean_abs': float((x - y).abs().mean())}
    # TF32 effect on the GPU itself, same seed.
    full = ECGImageLayer(fs=fs, config=RenderConfig(height=768, width=1024), seed=11, randomize=False).eval().to(device)
    gsig = signals(4, device)[0]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    strict = full(gsig)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    loose = full(gsig)
    difference = (strict - loose).abs()
    out['tf32_on_vs_off'] = {'gpu_supports_tf32': torch.cuda.get_device_capability()[0] >= 8,
                             'image_max_abs': float(difference.max()), 'image_mean_abs': float(difference.mean()),
                             'pixels_differing_over_one_8bit_code': float((difference > 1 / 255).float().mean())}
    torch.backends.cuda.matmul.allow_tf32 = False
    return out


ASSERT_PROBE = r'''
import torch, sys
from ecg_gpu import ECGImageLayer, RenderConfig
layer = ECGImageLayer(fs=500., config=RenderConfig(height=128, width=160)).cuda()
signal = torch.zeros(1, 12, 5000, device="cuda")
try:
    layer(signal, parameters={"blur": torch.tensor([5.], device="cuda")})
    torch.cuda.synchronize()
    print("first_call_returned_without_error")
except Exception as error:
    print("first_error:", type(error).__name__, str(error)[:300].replace("\n", " "))
try:
    torch.ones(1, device="cuda").sum().item()
    print("context_usable_after_failure")
except Exception as error:
    print("context_poisoned:", type(error).__name__, str(error)[:200].replace("\n", " "))
'''


def assert_probe():
    result = subprocess.run([sys.executable, '-c', ASSERT_PROBE], capture_output=True, text=True, timeout=300)
    return {'returncode': result.returncode, 'stdout': result.stdout[-2000:], 'stderr_tail': result.stderr[-1500:]}


def cuda_tests():
    result = subprocess.run([sys.executable, '-m', 'unittest', '-v', *CUDA_TEST_MODULES], capture_output=True,
                            text=True, timeout=1500)
    lines = [line for line in result.stderr.splitlines() if 'cuda' in line.lower() and ('...' in line)]
    return {'returncode': result.returncode, 'cuda_test_lines': lines[-40:], 'tail': result.stderr[-3000:]}


def gallery(out, device):
    from ecg_gpu.convenience import to_pil
    sig, fs = signals(6, device)
    layer = ECGImageLayer(fs=fs, config=RenderConfig(height=768, width=1024), seed=3).cuda()
    for index, image in enumerate(to_pil(layer(sig))):
        image.save(out / f'gpu-default-{index}.jpg', quality=92)
    flagged = ECGImageLayer(fs=fs, config=RenderConfig(height=768, width=1024, **ALL_FLAGS), seed=3).cuda()
    for index, image in enumerate(to_pil(flagged(sig))):
        image.save(out / f'gpu-all-flags-{index}.jpg', quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    args = parser.parse_args()
    workspace = Path(args.workspace)
    execution = json.loads((workspace / 'execution.json').read_text())
    deadline = Deadline(execution['deadline'] - time.time() - 150)
    out = workspace / 'outputs' / 'renderer-profile'
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable on the rented pod')
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {'schema': 'ecg-renderer-gpu-profile-v1', 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
              'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
              'signals': 'artificial demo_signals, 500 Hz, 10 s', 'started_at': time.time()}
    sections = [('parity', lambda: parity(device)), ('kernel_launches', lambda: kernel_launches(device)),
                ('assert_probe', assert_probe), ('throughput', lambda: throughput(deadline, device)),
                ('gallery', lambda: gallery(out, device)), ('cuda_tests', cuda_tests)]
    for name, section in sections:
        if deadline.left() < 240:
            report[name] = {'skipped': 'deadline'}
            continue
        begin = time.time()
        try:
            report[name] = section()
        except Exception:
            report[name] = {'error': traceback.format_exc()[-3000:]}
        report.setdefault('section_seconds', {})[name] = time.time() - begin
        write(out / 'report.json', report)
        print('section done:', name, round(time.time() - begin, 1), 's', flush=True)
    report['finished_at'] = time.time()
    write(out / 'report.json', report)


if __name__ == '__main__':
    main()
