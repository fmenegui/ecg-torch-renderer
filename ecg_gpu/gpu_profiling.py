"""Opt-in profiling of the public renderer or an on-the-fly training workload.

Compilation and warmup precede timing. Instrumented iterations are a separate
experiment. Inputs must already be on the renderer's device. No signal values
or patient metadata are written to the report.
"""
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import zipfile

import numpy as np
import torch


@contextmanager
def region_annotations(layer):
    """Attach temporary profiler labels, with no cost outside this context."""
    renderer = layer.renderer
    regions = [(renderer, '_curves'), (renderer, '_grid'), (renderer, '_text'),
               (renderer, '_stamp'), (renderer.document, 'deform'),
               (renderer.document, 'background'), (renderer.document, 'cast_shadow'),
               (renderer.document, 'object_shadow'), (renderer.camera, 'optical_blur'),
               (renderer.camera, 'jpeg')]
    saved = []
    try:
        for owner, name in regions:
            original = getattr(owner, name)
            saved.append((owner, name, original))
            def wrapped(*args, _original=original, _name=name, **kwargs):
                with torch.profiler.record_function('ecg/' + _name):
                    return _original(*args, **kwargs)
            setattr(owner, name, wrapped)
        yield
    finally:
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)


def _number(event, name, legacy=None):
    value = getattr(event, name, None)
    if value is None and legacy is not None:
        value = getattr(event, legacy, 0)
    return float(value or 0)


def _operator(event):
    return {'operator': event.key, 'calls': event.count,
            'cpu_total_ms': event.cpu_time_total / 1000,
            'cpu_self_ms': event.self_cpu_time_total / 1000,
            'device_total_ms': _number(event, 'device_time_total', 'cuda_time_total') / 1000,
            'device_self_ms': _number(event, 'self_device_time_total', 'self_cuda_time_total') / 1000,
            'cpu_self_memory_bytes': _number(event, 'self_cpu_memory_usage'),
            'device_self_memory_bytes': _number(event, 'self_device_memory_usage', 'self_cuda_memory_usage'),
            'input_shapes': event.input_shapes}


def _write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profile_layer(layer, signals, out, *, steps=20, warmup=5, profile_steps=3,
                  with_stack=False, memory_timeline=False, workload=None,
                  workload_name='ECGImageLayer.forward'):
    """Export timings, operators, kernels, allocations and a Perfetto trace.

    ``workload`` optionally runs one complete training step, including backward
    and optimizer. When supplied, it is executed during warmup, timing AND
    profiling, so model/optimizer/randomization state will advance. This utility
    is intended for a disposable benchmark instance. Region totals may overlap.
    It does not measure bandwidth, utilization, SM occupancy or clinical quality.
    """
    if min(steps, profile_steps) < 1 or warmup < 0:
        raise ValueError('Positive steps/profile_steps and nonnegative warmup required')
    if signals.device != layer.renderer.device:
        raise ValueError('Signals and renderer must already be on the same device')
    device = signals.device
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('Detailed profiling supports CPU and CUDA')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Never include stale artifacts in a new results archive.
    if any(out.iterdir()):
        raise ValueError('Profiling output directory must be empty')
    call = workload or (lambda: layer(signals))
    sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    sync()
    start = time.perf_counter()
    result = call()
    sync()
    cold_ms = (time.perf_counter() - start) * 1000
    del result
    for _ in range(warmup):
        result = call()
        del result
    sync()
    baseline = torch.cuda.memory_allocated(device) if device.type == 'cuda' else None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    times, device_times = [], []
    for _ in range(steps):
        sync()
        a = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        b = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        start = time.perf_counter()
        if a is not None:
            a.record()
        result = call()
        if b is not None:
            b.record()
        sync()
        times.append((time.perf_counter() - start) * 1000)
        if a is not None:
            device_times.append(a.elapsed_time(b))
        del result
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    stacks = with_stack or memory_timeline
    with region_annotations(layer), torch.profiler.profile(
            activities=activities, record_shapes=True, profile_memory=True,
            with_stack=stacks) as profile:
        for index in range(profile_steps):
            with torch.profiler.record_function('ecg/step_' + str(index)):
                result = call()
                del result
            profile.step()
        sync()
    operators = sorted((_operator(e) for e in profile.key_averages(group_by_input_shape=True)),
                       key=lambda row: row['cpu_self_ms'], reverse=True)
    _write_csv(out / 'operators.csv', operators)
    kernels = defaultdict(lambda: {'calls': 0, 'total_ms': 0.})
    for event in profile.events():
        if 'CUDA' in str(event.device_type):
            record = kernels[event.name]
            record['calls'] += 1
            record['total_ms'] += _number(event, 'device_time_total', 'cuda_time_total') / 1000
    kernel_rows = sorted(({'kernel': key, **value} for key, value in kernels.items()),
                         key=lambda row: row['total_ms'], reverse=True)
    _write_csv(out / 'kernels.csv', kernel_rows)
    trace = out / 'trace.json'
    profile.export_chrome_trace(str(trace))
    with trace.open('rb') as source, gzip.open(out / 'trace.json.gz', 'wb') as destination:
        shutil.copyfileobj(source, destination)
    trace.unlink()
    timeline = {'requested': memory_timeline, 'exported': False}
    if memory_timeline:
        try:
            profile.export_memory_timeline(str(out / 'memory-timeline.json.gz'), device=str(device))
            timeline['exported'] = True
        except (ValueError, RuntimeError, AssertionError) as error:
            timeline['reason'] = str(error)
    tables = [profile.key_averages().table(sort_by='self_cpu_time_total', row_limit=40)]
    if device.type == 'cuda':
        tables.append(profile.key_averages().table(sort_by='self_device_time_total', row_limit=40))
    (out / 'tables.txt').write_text('\n\n'.join(tables) + '\n')
    source_root = Path(__file__).resolve().parent
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(source_root.glob('gpu_*.py'))}
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
                   'numpy': np.__version__, 'platform': platform.platform(),
                   'cpu_threads': torch.get_num_threads(), 'cuda_build': torch.version.cuda,
                   'cudnn': torch.backends.cudnn.version(), 'device': str(device),
                   'source_sha256': source_hashes}
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        environment.update(gpu_name=properties.name, gpu_total_bytes=properties.total_memory,
                           compute_capability=list(torch.cuda.get_device_capability(device)))
    report = {'schema': 'ecg-renderer-detailed-profile-v1', 'environment': environment,
              'configuration': asdict(layer.renderer.config),
              'compile_raster': layer.renderer.compile_raster,
              'input_shape': list(signals.shape), 'input_dtype': str(signals.dtype),
              'workload': workload_name, 'cuda_measured': device.type == 'cuda',
              'timing': {'first_call_ms': cold_ms, 'first_call_includes_compilation_or_cache_loading': layer.renderer.compile_raster,
                         'warmup': warmup, 'steps': steps, 'latencies_ms': times,
                         'median_ms': float(np.median(times)), 'p90_ms': float(np.quantile(times, .9)),
                         'images_s': signals.shape[0] * steps * 1000 / sum(times),
                         'cuda_event_ms_median': float(np.median(device_times)) if device_times else None,
                         'cuda_peak_allocated_bytes': peak, 'cuda_peak_reserved_bytes': reserved,
                         'cuda_baseline_allocated_bytes': baseline,
                         'cuda_peak_additional_allocated_bytes': peak - baseline if peak is not None else None,
                         'profiling_excluded': True},
              'profile': {'steps': profile_steps, 'shapes': True, 'memory': True, 'stacks': stacks,
                          'cuda_kernel_events': sum(row['calls'] for row in kernel_rows),
                          'cuda_kernel_activity_captured': bool(kernel_rows),
                          'regions': [row for row in operators if row['operator'].startswith('ecg/')],
                          'top_cpu_operators': operators[:25],
                          'top_device_operators': sorted(operators, key=lambda r: r['device_self_ms'], reverse=True)[:25] if device.type == 'cuda' else [],
                          'top_device_allocations': sorted(operators, key=lambda r: r['device_self_memory_bytes'], reverse=True)[:25] if device.type == 'cuda' else [],
                          'top_kernels': kernel_rows[:25], 'memory_timeline': timeline},
              'limitations': ['Instrumented profile times are not benchmark latencies',
                              'Parent/child regions and concurrent GPU events may overlap',
                              'Allocator peaks exclude external GPU allocations and are not DRAM traffic',
                              'No hardware bandwidth, SM occupancy or utilization counters collected',
                              'Workload state advances during warmup, timing and profiling'],
              'signals_or_patient_metadata_exported': False}
    if device.type == 'cuda' and not kernel_rows:
        report['limitations'].append('CUDA kernel activities unavailable. Check CUPTI/runtime profiler permissions')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    archive = out.parent / (out.name + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as package:
        for path in sorted(out.iterdir()):
            package.write(path, path.name)
    report['archive'] = str(archive.resolve())
    return report
