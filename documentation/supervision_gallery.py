"""Append reproducible paired supervision examples to the documentation gallery."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from ecg_gpu import ECGSupervisionLayer, SupervisionConfig, RenderConfig
from ecg_gpu.demo import demo_signals
from ecg_gpu.schema_v05 import LEADS
from gallery import BASE, save_tensor


def generate(out, device='cpu', compile_raster=True):
    torch.set_num_threads(2)
    out = Path(out)
    root = out / 'assets/gallery'
    config = RenderConfig(height=768, width=1024, profile='photo', severity=.6,
                          antialias_samples=3, shadow_samples=6)
    supervision = SupervisionConfig(outputs=('lead_masks', 'trace_mask', 'bw', 'wb',
                                  'lead_boxes', 'instance_masks', 'instance_boxes'))
    params = {**BASE, 'surface_mode': 'fold', 'layout': '3x4+r3',
              'object_shadow': 'phone'}
    # All outputs come from this single capture, with identical deformation.
    device = torch.device(device)
    layer = ECGSupervisionLayer(fs=500, config=config, parameters=params, seed=804,
        randomize=False, supervision=supervision, compile_raster=compile_raster).to(device)
    values, _, _ = demo_signals(fs=500, duration=10)
    batch = layer(torch.tensor(values, device=device)[None])
    palette = torch.tensor([[.99,.40,.33],[.95,.67,.15],[.59,.75,.22],[.21,.70,.45],
        [.20,.76,.76],[.19,.56,.89],[.38,.43,.88],[.63,.38,.86],[.82,.38,.72],
        [.95,.45,.60],[.70,.63,.43],[.42,.69,.85]], device=device)
    colored = (batch['lead_masks'][0, :, None] * palette[:, :, None, None]).sum(0).clamp(0, 1)
    views = {
        'supervision-rgb': ('Input RGB', batch['image'][0]),
        'supervision-bw': ('White background, black leads', batch['bw'][0].expand(3, -1, -1)),
        'supervision-wb': ('Black background, white leads', batch['wb'][0].expand(3, -1, -1)),
        'supervision-channels': ('12 independent lead channels', colored),
        'supervision-valid': ('Visible supervision support', batch['valid'][0].expand(3, -1, -1)),
    }
    for index, lead in enumerate(LEADS):
        views['supervision-lead-' + lead] = (lead + ' channel', batch['lead_masks'][0, index:index+1].expand(3, -1, -1))
    report = json.loads((out / 'gallery.json').read_text())
    examples = [item for item in report['examples'] if not item['id'].startswith('supervision-')]
    for key, (title, value) in views.items():
        image = save_tensor(value, root / (key + '.jpg'))
        if key in {'supervision-bw', 'supervision-wb'}:
            gray = (value[0].detach().cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(gray).save(root / (key + '.png'))
        thumb = image.copy()
        thumb.thumbnail((480, 360), Image.Resampling.LANCZOS)
        thumb.save(root / (key + '-thumb.jpg'), quality=86)
        examples.append(dict(id=key, title=title, category='Supervision',
            description='One paired observation. Coverage excludes grid, labels and calibration pulses.',
            image='assets/gallery/' + key + '.jpg', thumbnail='assets/gallery/' + key + '-thumb.jpg',
            parameters=params, configuration=asdict(config), supervision=asdict(supervision), seed=804, photo=True))
    for kind in ['lead', 'instance']:
        image = Image.fromarray((batch['image'][0].detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        draw = ImageDraw.Draw(image)
        boxes = batch[kind + '_boxes'][0].cpu().tolist()
        ids = batch['lead_ids' if kind == 'lead' else 'instance_lead_ids'][0].cpu().tolist()
        for slot, (box, lead_id) in enumerate(zip(boxes, ids)):
            if box[0] < 0:
                continue
            color = tuple((palette[lead_id].cpu().numpy() * 255).astype(int))
            x0, y0, x1, y1 = box
            draw.rectangle((x0, y0, x1-1, y1-1), outline=color, width=2)
            label = LEADS[lead_id] + (' #' + str(slot) if kind == 'instance' else '')
            draw.text((x0+2, max(0, y0-13)), label, fill=color, stroke_width=1, stroke_fill='black')
        key = 'supervision-' + kind + '-boxes'
        image.save(root / (key + '.jpg'), quality=92, subsampling=0)
        image.thumbnail((480, 360), Image.Resampling.LANCZOS)
        image.save(root / (key + '-thumb.jpg'), quality=86)
        examples.append(dict(id=key, title=kind.title() + ' bounding boxes', category='Supervision',
            description='Pixel-edge xyxy bounds. Lead boxes aggregate repetitions. Instance boxes keep each printed occurrence separate.',
            image='assets/gallery/' + key + '.jpg', thumbnail='assets/gallery/' + key + '-thumb.jpg',
            parameters=params, configuration=asdict(config), supervision=asdict(supervision), seed=804, photo=True))
    for item in examples:
        item['image_sha256'] = hashlib.sha256((out / item['image']).read_bytes()).hexdigest()
    report.update(examples=examples, figures=len(examples), supervision_outputs={key:list(value.shape) for key,value in batch.items()})
    (out / 'gallery.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'figures': len(examples), 'paired_outputs': len(batch)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('docs'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--compile-raster', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    generate(args.out, args.device, args.compile_raster)
