"""Export 5 mm lattice points, point-only masks and a reviewed comparison.

python examples/grid_points.py --out build/grid-points --offline --spp 128
Signals are entirely artificial. Coordinates are geometric training labels,
not outputs of a trained Dotter detector. No asset is downloaded.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu import ECGImageLayer, GridPointConfig, capture_scenario
from ecg_gpu.demo import demo_signals


def export_grid_points(out, *, offline=False, spp=128, device='cpu'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    torch.set_num_threads(2)
    values, fs, _ = demo_signals(fs=500, duration=10)
    signals = torch.tensor(values, device=device)[None]
    source_hash = hashlib.sha256(signals.cpu().numpy().astype('<f4').tobytes()).hexdigest()
    point_config = GridPointConfig(sigma_pixels=1.1)
    cases = []
    for index, (name, recipe) in enumerate((('online-plano', 'clean_print'), ('online-dobra-perspectiva', 'folded_photo'))):
        config, parameters = capture_scenario(recipe, width=1024, height=768, severity=.8)
        parameters.update(layout='3x4+r1', grid_phase_x=.3, grid_phase_y=.7, grid_strength=.4,
                          object_shadow='none', grid_style=0)
        layer = ECGImageLayer(fs=fs, config=config, parameters=parameters, randomize=False, seed=420+index).to(device)
        rendered = layer.render_batch(signals, photo=recipe != 'clean_print', stage='all',
                                      return_grid_points=True, grid_point_config=point_config)
        if not torch.equal(rendered.reference, signals):
            raise AssertionError('Reference signal changed')
        cases.append((name, rendered.image, rendered.targets, rendered.metadata['grid_points']))
    if offline:
        from ecg_gpu.pathtrace import PathTraceConfig, pathtrace_capture
        import drjit as dr
        dr.set_thread_count(2)
        config, parameters = capture_scenario('clean_print', width=1536, height=1152)
        parameters.update(layout='3x4+r1', grid_phase_x=.3, grid_phase_y=.7, grid_strength=.4)
        layer = ECGImageLayer(fs=fs, config=config, parameters=parameters, randomize=False, seed=422)
        canonical = layer.render_batch(signals.cpu(), photo=False, stage='all', as_texture=True, return_instances=True)
        for index, (name, settings) in enumerate((
            ('offline-dobra-caneta', dict(surface='fold', bend_radians=.65, occluder=True)),
            ('offline-rugas-recorte', dict(surface='wrinkled', bend_radians=.8,
                 edge_tear=.08, missing_corner=.14, hole_radius=.05, geometry_seed=83)))):
            capture_config = PathTraceConfig(width=1024, height=768, samples_per_pixel=spp,
                key_radiance=5.2, environment_radiance=.18, **settings)
            capture = pathtrace_capture(canonical, config=capture_config, seed=423+index,
                return_grid_points=True, grid_point_config=point_config)
            cases.append((name, capture['image'], capture, capture['metadata']))
    font_path = 'DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 17)
    tile_w, tile_h, header, caption = 512, 384, 100, 65
    sheet = Image.new('RGB', (tile_w*3, header+len(cases)*(tile_h+caption)+40), '#e9edf0')
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(('Imagem de ECG', 'Intersecoes de 5 mm', 'Mascara somente dos pontos')):
        draw.text((column*tile_w+18, 20), label, font=font, fill='#172937')
    draw.text((18, 58), 'Rotulos geometricos sinteticos. Branco na mascara = ponto. Linhas de 1 mm excluidas.',
              font=small, fill='#405260')
    reports = []
    for index, (name, image_tensor, targets, metadata) in enumerate(cases):
        rgb = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        image = Image.fromarray((rgb.clip(0, 1)*255).round().astype('uint8'))
        tensors = {key: value.detach().cpu().numpy() for key, value in targets.items()
                   if key.startswith('grid_points_')}
        present = tensors['grid_points_valid'][0]
        xy = tensors['grid_points_xy'][0, present]
        ids = tensors['grid_points_ids'][0, present]
        canonical_xy = tensors['grid_points_canonical_xy'][0, present]
        mm = tensors['grid_points_mm'][0, present]
        error = tensors['grid_points_error_px'][0, present]
        mask = tensors['grid_points_mask'][0, 0]
        if not np.isfinite(xy).all() or not (error <= point_config.max_error_pixels).all():
            raise AssertionError('Invalid point export')
        if not np.all(mask <= targets['valid'][0, 0].detach().cpu().numpy()+1e-6):
            raise AssertionError('Point mask escaped visibility')
        binary = Image.fromarray(((mask > .25)*255).astype('uint8'))
        overlay = image.copy()
        overlay_draw = ImageDraw.Draw(overlay)
        for x, y in xy:
            overlay_draw.ellipse((float(x)-2.2, float(y)-2.2, float(x)+2.2, float(y)+2.2), fill='#00d5e8')
        image.save(out/(name+'.png'))
        overlay.save(out/(name+'-overlay.png'))
        binary.save(out/(name+'-mask.png'))
        np.savez_compressed(out/(name+'-targets.npz'), **tensors)
        with (out/(name+'-pontos.csv')).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('row', 'column', 'x_px', 'y_px', 'canonical_x_px', 'canonical_y_px', 'x_mm', 'y_mm', 'round_trip_error_canonical_px'))
            writer.writerows(np.column_stack((ids, xy, canonical_xy, mm, error)))
        y = header+index*(tile_h+caption)
        draw.text((18, y+10), name, font=font, fill='#172937')
        label = f'{len(xy)} pontos aceitos | erro de retorno maximo: {error.max():.4f} px canonicos'
        draw.text((18, y+38), label, font=small, fill='#405260')
        for column, tile in enumerate((image, overlay, binary.convert('RGB'))):
            sheet.paste(tile.resize((tile_w, tile_h), Image.Resampling.LANCZOS if column < 2 else Image.Resampling.NEAREST),
                        (column*tile_w, y+caption))
        if index == 1:
            # The unmodified binary target is enlarged by nearest-neighbor.
            crop = (300, 280, 620, 520)
            detail = Image.new('RGB', (1440, 400), '#e9edf0')
            detail_draw = ImageDraw.Draw(detail)
            for column, (label, tile) in enumerate(zip(('ECG', 'Pontos 5 mm', 'Mascara'), (image, overlay, binary.convert('RGB')))):
                detail_draw.text((column*480+14, 12), label, font=font, fill='#172937')
                detail.paste(tile.crop(crop).resize((480, 360), Image.Resampling.NEAREST if column == 2 else Image.Resampling.LANCZOS),
                             (column*480, 40))
            detail.save(out/'detalhe-pontos-5mm.png')
        reports.append({'case':name, 'active_points':int(tensors['grid_points_active'].sum()),
            'accepted_visible_points':int(present.sum()), 'max_round_trip_error_canonical_px':float(error.max()),
            'mean_round_trip_error_canonical_px':float(error.mean()), 'metadata':metadata})
    draw.text((18, sheet.height-28), 'Visibilidade geometrica. Nao demonstra legibilidade nem desempenho de uma rede treinada.',
              font=small, fill='#405260')
    sheet.save(out/'comparacao-pontos-5mm.jpg', quality=95, subsampling=0)
    report = {'schema':'ecg-grid-points-example-v1', 'spacing_mm':5., 'source_sha256':source_hash,
              'source':'entirely artificial demo ECG', 'device':device, 'cuda_verified':device == 'cuda',
              'coordinate_convention':'pixel centers, x=0,y=0 at the top-left pixel center',
              'reference_signal_preserved':True, 'cases':reports}
    (out/'grid-points.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--spp', type=int, default=128)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    result = export_grid_points(args.out, offline=args.offline, spp=args.spp, device=args.device)
    print(json.dumps({'cases':len(result['cases']), 'out':str(args.out)}))
