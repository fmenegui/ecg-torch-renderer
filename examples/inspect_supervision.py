"""Export each of 12 lead masks and both lead/instance bounding-box views."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu import ECGSupervisionLayer, SupervisionConfig, capture_scenario, load_rgb_bank
from ecg_gpu.demo import demo_signals
from ecg_gpu.gpu_cli import plain
from ecg_gpu.schema_v05 import LEADS


def inspect_supervision(out, background=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    c, p = capture_scenario('folded_photo', width=1024, height=768, severity=.8)
    p.update(layout='3x4+r3', lead_present=1., angle=.06, zoom=.86,
             object_shadow='none', artifact_present=0., tear_active=0.)
    bank = load_rgb_bank(background) if background else None
    synth = ECGSupervisionLayer(fs=500, config=c, parameters=p, backgrounds=bank,
        supervision=SupervisionConfig(outputs=('lead_masks', 'lead_boxes', 'instance_masks', 'instance_boxes')),
        seed=781, randomize=False).eval()
    values, _, _ = demo_signals(fs=500, duration=10)
    batch = synth(torch.tensor(values)[None])
    rgb = (batch['image'][0].permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
    Image.fromarray(rgb).save(out/'rgb.png')
    np.savez_compressed(out/'targets.npz', **{name: value.numpy() for name, value in batch.items() if name != 'image'})
    tile_w, tile_h = 384, 288
    sheet = Image.new('RGB', (4*tile_w, 3*(tile_h+34)), '#f0f0ed')
    draw = ImageDraw.Draw(sheet)
    for index, lead in enumerate(LEADS):
        mask = batch['lead_masks'][0, index].numpy()
        mask_image = Image.fromarray((mask*255).round().astype(np.uint8))
        mask_image.save(out/('mask-'+lead+'.png'))
        panel = np.zeros_like(rgb)
        panel[..., 1] = (mask*255).round().astype(np.uint8)
        preview = Image.fromarray(panel)
        painter = ImageDraw.Draw(preview)
        if batch['lead_box_valid'][0, index]:
            x0, y0, x1, y1 = batch['lead_boxes'][0, index].tolist()
            painter.rectangle((x0, y0, x1-1, y1-1), outline='#ffd070', width=2)
        x, y = index%4*tile_w, index//4*(tile_h+34)
        draw.text((x+12, y+10), lead+' | semantic waveform mask + envelope', fill='#222')
        sheet.paste(preview.resize((tile_w, tile_h), Image.Resampling.LANCZOS), (x, y+34))
    sheet.save(out/'twelve-lead-masks.jpg', quality=90, subsampling=0)
    report = {'class_order': LEADS, 'mask_shape': list(batch['lead_masks'].shape),
        'lead_boxes': plain(batch['lead_boxes']), 'lead_box_valid': plain(batch['lead_box_valid']),
        'instance_boxes': plain(batch['instance_boxes']),
        'instance_box_valid': plain(batch['instance_box_valid']),
        'instance_lead_ids': plain(batch['instance_lead_ids']),
        'instance_active': plain(batch['instance_active']),
        'box_convention': 'xyxy pixel edges, exclusive right/bottom',
        'lead_box_definition': 'union envelope of all visible printed occurrences of one lead',
        'instance_box_definition': 'one visible printed occurrence, preferred for YOLO'}
    (out/'supervision.json').write_text(json.dumps(report, indent=2)+'\n')
    print({'masks': 12, 'active_instances': int(batch['instance_active'].sum()),
           'visible_instance_boxes': int(batch['instance_box_valid'].sum())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--background', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    inspect_supervision(args.out, args.background)
