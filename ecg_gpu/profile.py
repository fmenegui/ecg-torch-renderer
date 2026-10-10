"""Detailed renderer profiling in one command, with a portable results ZIP."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .demo import demo_signals
from .audit import PRESETS
from .gpu_cli import load_signals
from .gpu_module import ECGImageLayer
from .gpu_profiling import profile_layer
from .gpu_renderer import RenderConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--preset', choices=list(PRESETS), default='balanced')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--profile-steps', type=int, default=3)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--compile-raster', action='store_true')
    parser.add_argument('--stacks', action='store_true')
    parser.add_argument('--memory-timeline', action='store_true', help='Enable stacks and export allocation timeline. Adds profiling overhead')
    parser.add_argument('--train-step', action='store_true', help='Include a tiny artificial CNN, AMP, backward and AdamW')
    parser.add_argument('--input', type=Path, help='Optional NPY/NPZ/WFDB signals. Input values and metadata are not exported')
    parser.add_argument('--fs', type=float)
    parser.add_argument('--units', choices=['mV', 'uV', 'V'])
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.profile_steps, args.threads) < 1 or args.warmup < 0:
        parser.error('Positive batch/steps/profile-steps/threads and nonnegative warmup required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable. Select a GPU runtime or explicitly use --device cpu')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if args.input:
        values, rates, lengths = load_signals(args.input, args.fs, args.units)
        if len(values) == 1:
            values = np.repeat(values, args.batch_size, axis=0)
            if lengths is not None:
                lengths = np.repeat(lengths, args.batch_size)
        elif len(values) != args.batch_size:
            parser.error('Input batch must have one signal or match --batch-size')
    else:
        values, _, _ = demo_signals(fs=500, duration=10)
        values = np.repeat(values[None], args.batch_size, axis=0)
        rates, lengths = 500, None
    signals = torch.tensor(values, dtype=torch.float32, device=device)
    rates = torch.as_tensor(rates, dtype=torch.float32, device=device)
    lengths = torch.as_tensor(lengths, device=device) if lengths is not None else None
    layer = ECGImageLayer(fs=500, config=RenderConfig(profile='mixed', severity=.6, **PRESETS[args.preset]),
                          seed=1973, compile_raster=args.compile_raster).to(device)
    image_step = lambda: layer(signals, fs=rates, lengths=lengths)
    workload, name = image_step, 'ECGImageLayer.forward, input already on device'
    if args.train_step:
        torch.manual_seed(1973)
        cnn = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, stride=4), torch.nn.ReLU(),
              torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 2)).to(device)
        classes = torch.arange(len(signals), device=device).remainder(2)
        optimizer = torch.optim.AdamW(cnn.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        def training_step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                output = cnn(image_step())
                loss = torch.nn.functional.cross_entropy(output, classes)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            return loss.detach()
        workload, name = training_step, 'renderer + tiny artificial CNN + AMP + backward + AdamW'
    report = profile_layer(layer, signals, args.out, steps=args.steps, warmup=args.warmup,
              profile_steps=args.profile_steps, with_stack=args.stacks,
              memory_timeline=args.memory_timeline, workload=workload, workload_name=name)
    print(json.dumps({'archive': report['archive'], 'cuda_measured': report['cuda_measured'],
                     'timing': report['timing'], 'cuda_kernel_events': report['profile']['cuda_kernel_events']}), flush=True)


if __name__ == '__main__':
    main()
