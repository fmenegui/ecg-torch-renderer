"""Compare configurable point masks while retaining the same subpixel centers.

python examples/grid_point_sizes.py --out build/grid-point-sizes --offline
PNG files are previews. Float32 NPZ arrays preserve the training targets.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu import (ECGImageLayer, GridPointConfig, capture_scenario,
                     rasterize_grid_points, decode_grid_points)
from ecg_gpu.demo import demo_signals


def _pil(tensor):
    array = tensor.detach().cpu().numpy()
    if array.ndim == 3:
        array = array.transpose(1, 2, 0)
    return Image.fromarray((array.clip(0, 1)*255).round().astype('uint8'))


def export(out, *, offline=False, spp=64):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    torch.set_num_threads(2)
    font = ImageFont.truetype('DejaVuSans.ttf', 20)
    small = ImageFont.truetype('DejaVuSans.ttf', 16)
    settings = [('Gaussiana sigma 0,6 px', GridPointConfig(sigma_pixels=.6)),
                ('Gaussiana sigma 1,2 px', GridPointConfig(sigma_pixels=1.2)),
                ('Disco raio 2,5 px', GridPointConfig(mask_shape='disk', radius_pixels=2.5))]
    # Exact numerical illustration of one center, shared across all sizes.
    center = torch.tensor([[[5.35, 5.7]]])
    isolated = Image.new('RGB', (1152, 470), '#edf1f3')
    isolated_draw = ImageDraw.Draw(isolated)
    isolated_errors = []
    for col, (label, config) in enumerate(settings):
        target = rasterize_grid_points(center, torch.ones(1, 1, dtype=torch.bool),
                                       torch.ones(1, 1, 12, 12), config=config)
        tile = _pil(target['grid_points_mask'][0, 0]).convert('RGB').resize((384, 384), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(tile)
        for step in range(0, 385, 32):
            draw.line((step, 0, step, 383), fill='#33424a')
            draw.line((0, step, 383, step), fill='#33424a')
        x, y = (5.35+.5)*32, (5.7+.5)*32
        draw.line((x-9, y, x+9, y), fill='#ff5353', width=2)
        draw.line((x, y-9, x, y+9), fill='#ff5353', width=2)
        isolated_draw.text((col*384+12, 10), label, font=font, fill='#183441')
        isolated.paste(tile, (col*384, 44))
        isolated_draw.text((col*384+12, 438), 'Mesmo centro: (5,35, 5,70) px', font=small, fill='#183441')
        if config.mask_shape == 'gaussian':
            decoded = decode_grid_points(target['grid_points_mask'], threshold=.1, max_points=1)
            error = (decoded['xy'][0, 0]-center[0, 0]).norm()
            isolated_errors.append({'sigma':config.sigma_pixels, 'decoded_error_px':float(error)})
    isolated.save(out/'centro-subpixel-mascaras.png')
    values, fs, _ = demo_signals(fs=500, duration=10)
    signals = torch.tensor(values)[None]
    config, parameters = capture_scenario('folded_photo', width=1024, height=768, severity=.8)
    parameters.update(layout='3x4+r1', grid_phase_x=.3, grid_phase_y=.7, grid_strength=.4, object_shadow='none')
    layer = ECGImageLayer(fs=fs, config=config, parameters=parameters, randomize=False, seed=421)
    rendered = layer.render_batch(signals, stage='all', return_grid_points=True,
                                  grid_point_config=GridPointConfig(return_offsets=True))
    captures = [('online', rendered.image, rendered.targets)]
    if offline:
        import drjit as dr
        from ecg_gpu.pathtrace import pathtrace_capture, PathTraceConfig
        dr.set_thread_count(2)
        config, parameters = capture_scenario('clean_print', width=1536, height=1152)
        parameters.update(layout='3x4+r1', grid_phase_x=.3, grid_phase_y=.7, grid_strength=.4)
        layer = ECGImageLayer(fs=fs, config=config, parameters=parameters, randomize=False, seed=422)
        canonical = layer.render_batch(signals, photo=False, stage='all', as_texture=True, return_instances=True)
        capture = pathtrace_capture(canonical, config=PathTraceConfig(width=1024, height=768,
            surface='fold', bend_radians=.65, occluder=True, samples_per_pixel=spp,
            key_radiance=5.2, environment_radiance=.18), seed=423, return_grid_points=True,
            grid_point_config=GridPointConfig(return_offsets=True))
        captures.append(('offline', capture['image'], capture))
    contact = Image.new('RGB', (1440, 65+len(captures)*335), '#edf1f3')
    draw = ImageDraw.Draw(contact)
    for col, label in enumerate(['ECG e centros', *[s[0] for s in settings]]):
        draw.text((col*360+12, 15), label, font=font, fill='#183441')
    reports = []
    for row, (name, image, targets) in enumerate(captures):
        xy, present, valid = targets['grid_points_xy'], targets['grid_points_valid'], targets['valid']
        rgb = _pil(image[0])
        rgb.save(out/(name+'-ecg.png'))
        np.savez_compressed(out/(name+'-coordenadas-offsets.npz'), **{
            k:v.cpu().numpy() for k, v in targets.items() if k.startswith('grid_points_')})
        overlay = rgb.copy()
        marking = ImageDraw.Draw(overlay)
        for x, y in xy[0, present[0]].tolist():
            marking.ellipse((x-1.6, y-1.6, x+1.6, y+1.6), fill='#00cddd')
        y0 = 65+row*335
        draw.text((12, y0), name+' | centros identicos em todos os tamanhos de mascara', font=small, fill='#183441')
        crop = (300, 270, 620, 510)
        contact.paste(overlay.crop(crop).resize((360, 270), Image.Resampling.LANCZOS), (0, y0+40))
        masks = {}
        for col, (_, config) in enumerate(settings, 1):
            outputs = rasterize_grid_points(xy, present, valid, config=config)
            mask = outputs['grid_points_mask']
            preview = _pil(mask[0, 0])
            preview.save(out/(name+'-mascara-'+str(col)+'.png'))
            contact.paste(preview.convert('RGB').crop(crop).resize((360, 270), Image.Resampling.NEAREST),
                          (col*360, y0+40))
            masks['mask_'+str(col)] = mask.numpy()
        np.savez_compressed(out/(name+'-mascaras-float32.npz'), **masks)
        selected = targets['grid_points_valid']
        reports.append({'backend':name, 'points':int(selected.sum()),
            'max_material_residual_px':float(targets['grid_points_error_px'][selected].max()),
            'max_image_residual_px':float(targets['grid_points_error_image_px'][selected].max())})
    contact.save(out/'comparacao-tamanhos-mascaras.jpg', quality=95, subsampling=0)
    report = {'schema':'ecg-grid-mask-sizes-v1', 'settings':[asdict(s[1]) for s in settings],
        'ideal_gaussian_decoder':isolated_errors, 'captures':reports,
        'positions_changed_by_mask_size':False, 'signals':'entirely artificial',
        'source_sha256':hashlib.sha256(signals.numpy().astype('<f4').tobytes()).hexdigest(),
        'cuda_verified':False, 'real_image_detector_accuracy':False,
        'precision_note':'Numerical residuals verify the represented geometry, not real-world physical calibration.'}
    (out/'subpixel-report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--spp', type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(export(args.out, offline=args.offline, spp=args.spp), indent=2))
