"""Measure the actual device and export illustrative images outside the loop."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .demo import demo_signals
from .gpu_renderer import ECGRenderer, RenderConfig
from .schema_v05 import LEADS, LAYOUTS


def json_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {name: json_tensors(item) for name, item in value.items()}
    return value


def png(tensor):
    return Image.fromarray((tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--signals', type=Path, help='Optional authorized public/local NPZ, mV, not loaded inside the timed loop')
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--stage', choices=['image', 'geometry', 'trace', 'metadata', 'all'], default='geometry')
    parser.add_argument('--profile', choices=['clean', 'scan', 'photo', 'mixed'], default='mixed')
    parser.add_argument('--severity', type=float, default=.6)
    parser.add_argument('--antialias', type=int, default=4)
    parser.add_argument('--surface',choices=['mixed','off','flat','curl','fold','accordion','crumpled','wrinkled','rolled','corner'],default='mixed')
    parser.add_argument('--surface-resolution',type=int,default=96)
    parser.add_argument('--inverse-iterations',type=int,default=5)
    parser.add_argument('--shadow-samples',type=int,default=8)
    parser.add_argument('--examples', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save-targets', action='store_true', help='Also serialize the large dense paired tensors for auditing')
    parser.add_argument('--canonical', action='store_true', help='Measure rectified printing without camera geometry')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('No CUDA GPU available. Use --device cpu for software verification only.')
    if min(args.batch_size, args.steps, args.threads) < 1 or args.warmup < 0:
        parser.error('Positive batch size, steps and threads required')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if args.signals:
        with np.load(args.signals, allow_pickle=False) as source:
            fs, names = float(source['fs']), [str(name).upper() for name in source['lead_names']]
            signals = np.asarray(source['signals'], np.float32)[[names.index(name.upper()) for name in LEADS]]
            if 'units' in source and str(source['units']).lower() != 'mv':
                raise ValueError('Input signals must be mV')
        provenance = {'type':'user-supplied signal, not a clinical validation cohort',
                      'npz_sha256':hashlib.sha256(args.signals.read_bytes()).hexdigest()}
    else:
        signals, fs, _ = demo_signals(fs=500, duration=10)
        provenance = {'type':'entirely artificial demo ECG', 'real_patients':0}
    signals = torch.as_tensor(signals, dtype=torch.float32, device=device)[None].expand(args.batch_size, -1, -1).contiguous()
    renderer = ECGRenderer(RenderConfig(height=args.height, width=args.width, profile=args.profile,
                                       severity=args.severity, antialias_samples=args.antialias,
                                       surface_mode=args.surface,surface_resolution=args.surface_resolution,
                                       inverse_iterations=args.inverse_iterations,shadow_samples=args.shadow_samples)).to(device)
    synchronize = torch.cuda.synchronize if device.type == 'cuda' else lambda: None
    for index in range(args.warmup):
        batch = renderer(signals, fs, stage=args.stage, photo=not args.canonical, generator=torch.Generator(device=device).manual_seed(1973 + index))
        synchronize()
        del batch
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    durations = []
    for index in range(args.steps):
        generator = torch.Generator(device=device).manual_seed(21973 + index)
        synchronize()
        start = time.perf_counter()
        batch = renderer(signals, fs, stage=args.stage, photo=not args.canonical, generator=generator)
        synchronize()
        durations.append(time.perf_counter() - start)
        del batch
    report = {'schema':'ecg-replot-renderer-benchmark-v1', 'device':str(device), 'torch_version':str(torch.__version__),
              'cuda_measured':device.type == 'cuda', 'gpu_name':torch.cuda.get_device_name() if device.type == 'cuda' else None,
              'host':platform.machine(), 'cpu_threads':args.threads, 'stage':args.stage, 'canonical_mode':args.canonical,
              'batch_size':args.batch_size, 'warmup_batches':args.warmup, 'measured_batches':args.steps,
              'batch_latency_ms_median':float(np.median(durations) * 1000),
              'batch_latency_ms_p90':float(np.quantile(durations, .9) * 1000),
              'images_per_second':args.batch_size * len(durations) / sum(durations),
              'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
              'configuration':renderer.specification(), 'signal_provenance':provenance,
              'timed_scope':('print and camera simulation, image-only' if args.stage=='image' else 'print, selected masks, camera simulation')+'. Excludes disk I/O, PNG encoding, signal transfer and font initialization',
              'physionet_comparison_executed':False, 'realism_validated_against_real_photos':False,
              'clinical_validation':False, 'examples':[]}
    if args.examples:
        thumb_width = 640
        thumb_height = round(args.height * thumb_width / args.width)
        mosaic = Image.new('RGB', (thumb_width * 2, (thumb_height + 36) * (len(LAYOUTS)-1)), 'white')
        draw = ImageDraw.Draw(mosaic)
        for layout in range(len(LAYOUTS)-1):
            for col, photographic in enumerate([False, True]):
                preview_renderer = ECGRenderer(RenderConfig(height=args.height, width=args.width,
                                                            profile='photo' if photographic else 'scan', severity=.7,
                                                            antialias_samples=args.antialias)).to(device)
                generator = torch.Generator(device=device).manual_seed(44000 + layout + col * 700)
                parameters = {'layout':layout, 'speed_index':1, 'gain_index':1,
                              'header_present':1., 'quarter':0, 'angle':.055 if photographic else 0.,
                              'label_present':1., 'occlusion_active':float(layout == 4 and photographic)}
                example = preview_renderer(signals[:1], fs, stage='all', generator=generator,
                                           parameters=parameters, photo=photographic, return_field=True)
                stem = LAYOUTS[layout + 1].replace('+', '-') + ('-photo' if photographic else '-scan')
                picture = png(example.image[0])
                picture.save(args.out / (stem + '.png'))
                (args.out / (stem + '.json')).write_text(json.dumps(json_tensors(example.metadata), indent=2))
                if layout == 0 and photographic:
                    overlay = picture.copy()
                    owners = example.targets['traces'][0].amax(0).cpu().numpy()
                    array = np.asarray(overlay).copy()
                    selected = owners > .25
                    array[selected] = (array[selected] * .4 + np.array([0,225,95]) * .6).astype(np.uint8)
                    Image.fromarray(array).save(args.out / 'trace-overlay.png')
                    # Dense audit targets remain optional local verification data.
                    if args.save_targets:
                        torch.save({'targets':example.targets, 'inverse_field':example.inverse_field}, args.out / 'paired-targets.pt')
                x, y = col * thumb_width, layout * (thumb_height + 36)
                draw.text((x + 12, y + 8), f'{LAYOUTS[layout + 1]} | {"simulated photo" if photographic else "simulated scan"}', fill='black')
                mosaic.paste(picture.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS), (x, y + 36))
                report['examples'].append(stem + '.png')
                del example, preview_renderer
        mosaic.save(args.out / 'preview.png')
    (args.out / 'benchmark.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key:report[key] for key in ['device','cuda_measured','images_per_second','batch_latency_ms_median','batch_latency_ms_p90','clinical_validation']}))


if __name__ == '__main__':
    main()
