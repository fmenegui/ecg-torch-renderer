"""Build a diverse offline Mitsuba geometry bank for the torch GPU renderer.

Each entry is one true-3D paper capture: folds, multifolds, wrinkles,
crumpling, curl, missing corners, tears and holes, under a jittered camera.
Only geometry is stored (inverse UV field, visibility, normals and relief),
never ECG content, so the GPU renderer composes any signal, layout, material
and camera noise on top at batch time with one grid_sample.

    python examples/build_geometry_bank.py --out build/geometry-bank.npz --count 96

The bank resolution is independent of the training resolution: the renderer
samples it with normalized coordinates. CPU only (Mitsuba llvm_ad_rgb).
"""
import argparse
import dataclasses
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_gpu import DocumentAssets, ECGRenderer, RenderConfig  # noqa: E402
from ecg_gpu.demo import demo_signals  # noqa: E402
from ecg_gpu.gpu_appearance import srgb_to_linear  # noqa: E402
from ecg_gpu.pathtrace import PathTraceConfig, capture_geometry_assets, pathtrace_capture  # noqa: E402

SURFACES = [('fold', .20), ('multifold', .20), ('wrinkled', .20), ('crumpled', .20), ('curl', .15), ('flat', .05)]


def sample_config(rng, width, height, seed):
    surface = rng.choices([s for s, _ in SURFACES], [w for _, w in SURFACES])[0]
    # Phone held above the sheet, mostly in front and slightly to one side.
    distance = rng.uniform(.26, .42)
    tilt = math.radians(rng.uniform(4, 32))
    azimuth = rng.uniform(-math.pi, math.pi) * .35 - math.pi / 2
    origin = (distance * math.sin(tilt) * math.cos(azimuth), distance * math.sin(tilt) * math.sin(azimuth),
              distance * math.cos(tilt))
    target = (rng.uniform(-.02, .02), rng.uniform(-.02, .02), .006)
    key_angle = rng.uniform(-math.pi, math.pi)
    key_radius = rng.uniform(.10, .30)
    config = dict(width=width, height=height, samples_per_pixel=4, label_samples_per_axis=1, surface=surface,
                  bend_radians=rng.uniform(.25, 1.05), camera_origin=origin, camera_target=target,
                  field_of_view=rng.uniform(42, 62), key_origin=(key_radius * math.cos(key_angle),
                  key_radius * math.sin(key_angle), rng.uniform(.30, .50)),
                  geometry_seed=seed, surface_strength=rng.uniform(.5, 1.3), fold_count=rng.randint(2, 6),
                  wrinkle_amplitude_m=rng.uniform(.0002, .0012), mesh_columns=97, mesh_rows=49)
    if rng.random() < .15:
        config['missing_corner'] = rng.uniform(.05, .2)
    if rng.random() < .15:
        config['edge_tear'] = rng.uniform(.05, .2)
    if rng.random() < .10:
        config['hole_radius'] = rng.uniform(.03, .08)
    return config


def ambient_occlusion(white_page, config, mask, spp):
    """Light-direction-free shading: a white sheet under a uniform sky.

    Flat unoccluded paper receives radiance equal to the sky radiance, so the
    ratio is the fraction of sky each point sees plus interreflection: dark
    inside folds and creases, 1 on open paper. The GPU renderer multiplies it
    with its own sampled lights, so no baked key-light direction conflicts.
    """
    sky = .5
    capture = pathtrace_capture(white_page, config=dataclasses.replace(
        config, key_radiance=0., environment_radiance=sky, samples_per_pixel=spp))
    radiance = srgb_to_linear(capture['image']).mean(1, keepdim=True) / sky
    # Occlusion is low-frequency; a normalized sigma=2 px blur inside the sheet removes MC noise.
    taps = torch.arange(-4, 5, dtype=torch.float32)
    kernel = torch.exp(-taps.square() / (2 * 2. ** 2))
    kernel = kernel[:, None] * kernel[None]
    kernel = (kernel / kernel.sum())[None, None]
    weight = torch.nn.functional.conv2d(mask, kernel, padding=4)
    blurred = torch.nn.functional.conv2d(radiance * mask, kernel, padding=4)
    shading = torch.where(weight > 1e-3, blurred / weight.clamp_min(1e-3), torch.ones_like(radiance))
    return shading.clamp(.15, 1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, default=96)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--seed', type=int, default=1973)
    parser.add_argument('--occlusion-spp', type=int, default=48, help='0 disables the baked ambient occlusion')
    args = parser.parse_args()
    rng = random.Random(args.seed)
    values, fs, _ = demo_signals(fs=500, duration=10)
    # Content is irrelevant to geometry; the page only fixes the physical sheet size.
    renderer = ECGRenderer(RenderConfig(height=args.height, width=args.width, profile='clean'))
    # 25 mm/s over 10 s plus margins: a 271 mm wide sheet, close to A4/letter landscape.
    canonical = renderer(torch.tensor(values)[None], fs, photo=False, as_texture=True, return_instances=True,
                         parameters={'speed_index': 1}, generator=torch.Generator().manual_seed(args.seed))
    white_page = dataclasses.replace(canonical, image=torch.ones_like(canonical.image))
    banks, configs, failures = [], [], []
    begin = time.time()
    for index in range(args.count):
        config = sample_config(rng, args.width, args.height, args.seed * 1000 + index)
        try:
            trace_config = PathTraceConfig(**config)
            capture = pathtrace_capture(canonical, config=trace_config, return_geometry=True)
            bank = capture_geometry_assets(capture)
            if args.occlusion_spp:
                bank.shading = ambient_occlusion(white_page, trace_config, bank.mask, args.occlusion_spp)
            visible = float(bank.mask.mean())
            if visible < .25:
                failures.append({'index': index, 'reason': f'only {visible:.0%} of the frame is paper', 'config': config})
                continue
            banks.append(bank)
            configs.append({**config, 'visible_fraction': visible})
        except Exception as error:
            failures.append({'index': index, 'reason': f'{type(error).__name__}: {error}'[:300], 'config': config})
        if index % 8 == 7:
            print(json.dumps({'done': index + 1, 'kept': len(banks), 'seconds': round(time.time() - begin)}), flush=True)
    if not banks:
        raise RuntimeError('No usable geometry captured')
    merged = DocumentAssets(
        inverse_field=torch.cat([b.inverse_field for b in banks]), mask=torch.cat([b.mask for b in banks]),
        normals=torch.cat([b.normals for b in banks]), height=torch.cat([b.height for b in banks]),
        shading=torch.cat([b.shading for b in banks]) if args.occlusion_spp else None,
        provenance={'schema': 'ecg-geometry-bank-v1', 'builder': 'examples/build_geometry_bank.py',
                    'seed': args.seed, 'entries': len(banks), 'resolution': [args.height, args.width],
                    'sampling': 'center-ray Mitsuba llvm_ad_rgb', 'contains_document_content': False,
                    'shading': f'path-traced uniform-sky ambient occlusion, {args.occlusion_spp} spp' if args.occlusion_spp else None,
                    'configs': configs, 'rejected': len(failures)})
    out = Path(args.out)
    merged.to_npz(out)
    out.with_suffix('.json').write_text(json.dumps({'entries': len(banks), 'rejected': failures,
                                                    'seconds': time.time() - begin}, indent=2))
    print(json.dumps({'out': str(out), 'entries': len(banks), 'rejected': len(failures),
                      'bytes': out.stat().st_size, 'seconds': round(time.time() - begin)}))


if __name__ == '__main__':
    main()
