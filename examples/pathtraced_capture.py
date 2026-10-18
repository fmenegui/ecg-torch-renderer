"""Create actual 3D path-traced ECGs with aligned masks and detection boxes.

Optional dependency: pip install '.[pathtrace]'
Example: python examples/pathtraced_capture.py --out build/pathtraced --spp 128
Supply --input recording.npz --fs 500 for your own signals. Default artificial
waveforms verify engineering only. No asset or pretrained weight is downloaded.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu import ECGImageLayer, capture_scenario
from ecg_gpu.demo import demo_signals
from ecg_gpu.gpu_cli import load_signals, plain
from ecg_gpu.pathtrace import pathtrace_capture, PathTraceConfig
from ecg_gpu.dataset_export import samples_from_batch, export_yolo_dataset


def export_pathtraced(out, *, width=1024, height=768, spp=128, input_path=None, fs=None,
                      background=None, adverse=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Capture output must be a new or empty directory')
    if input_path is None:
        values, fs, _ = demo_signals(fs=500, duration=10)
        signals = torch.tensor(values)[None]
        source = 'entirely artificial demo ECG'
    else:
        values, rates, lengths = load_signals(input_path, fs=fs)
        if len(values) != 1:
            raise ValueError('Example takes one recording at a time')
        if lengths is not None:
            values = values[..., :lengths[0]]
        fs = float(np.asarray(rates).reshape(-1)[0])
        signals = torch.from_numpy(values)
        source = str(Path(input_path).name)
    source_hash = hashlib.sha256(signals.numpy().astype('<f4').tobytes()).hexdigest()
    canonical_width = max(1024, width)
    canonical_height = round(canonical_width*height/width)
    c, p = capture_scenario('clean_print', width=canonical_width, height=canonical_height)
    p.update(layout='3x4+r1', grid_strength=.38, minor_strength=.30, line_width=1.25)
    layer = ECGImageLayer(fs=fs, config=c, parameters=p, randomize=False).eval()
    canonical = layer.render_batch(signals, stage='all', photo=False, return_instances=True)
    cases = [dict(name='flat',surface='flat',bend_radians=0.),
             dict(name='curl',surface='curl',bend_radians=.55),
             dict(name='fold',surface='fold',bend_radians=.24,occluder=True)]
    if adverse:
        cases = [
            dict(name='stored_folds',surface='multifold',bend_radians=.8,fold_count=4,
                 geometry_seed=41, material=dict(paper_age=.19,foxing_strength=.3,
                 thermal_fade=.14,fiber_strength=.4)),
            dict(name='crumpled_torn',surface='crumpled',bend_radians=1.1,fold_count=7,
                 geometry_seed=41,missing_corner=.20,edge_tear=.14,hole_radius=.045,
                 key_origin=(-.24,-.09,.12),key_size=(.16,.09),
                 key_radiance=12.,environment_radiance=.18,
                 material=dict(water_strength=.55,stain_strength=.20,stain_x=.54,
                 stain_y=.46,thermal_fade=.23,paper_age=.07)),
            dict(name='wrinkled_coffee',surface='wrinkled',bend_radians=.7,
                 geometry_seed=109,key_origin=(-.20,-.12,.18),key_size=(.20,.12),
                 key_radiance=10.,environment_radiance=.20,
                 material=dict(coffee_strength=.7,coffee_radius=.18,
                 stain_strength=.15,stain_x=.64,stain_y=.54,water_strength=.3)),
            dict(name='oblique_fragments',surface='wrinkled',bend_radians=.55,
                 geometry_seed=83,camera_origin=(.12,-.14,.38),camera_up=(.18,1.,0.),
                 missing_corner=.23,edge_tear=.12,
                 material=dict(paper_age=.13,edge_wear_strength=.8,
                 hole_active=[1.,0.,0.],hole_center=[[-.24,.12],[0.,0.],[0.,0.]],
                 hole_radius=[[.06,.05],[.03,.03],[.03,.03]],tear_raggedness=.8))]
    tile_w, tile_h = 640, round(640*height/width)
    sheet = Image.new('RGB', (tile_w*2, (tile_h+40)*len(cases)), '#efefec')
    draw = ImageDraw.Draw(sheet)
    reports = []
    for index, case in enumerate(cases):
        case = dict(case)
        name = case.pop('name')
        material = case.pop('material', {})
        if material:
            layer = ECGImageLayer(fs=fs, config=c, parameters={**p,**material},
                                  randomize=False,seed=781+index).eval()
            canonical = layer.render_batch(signals, stage='all', photo=False, return_instances=True)
        config = PathTraceConfig(width=width, height=height, samples_per_pixel=spp,
                                 **case)
        batch = pathtrace_capture(canonical, config=config, background_texture=background,
                                  seed=241000+index)
        rgb = (batch['image'][0].permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
        Image.fromarray(rgb).save(out/(name+'.png'))
        targets = {name: value.numpy() for name, value in batch.items()
                   if isinstance(value, torch.Tensor) and name != 'image'}
        np.savez_compressed(out/(name+'-targets.npz'), **targets)
        colored = np.asarray([[.95,.2,.2],[.2,.65,1],[.2,.8,.25],[.95,.65,.1],
            [.7,.3,.95],[.2,.8,.8],[1,.4,.7],[.5,.6,.15],[.4,.5,.95],
            [.95,.45,.1],[.5,.8,.4],[.6,.35,.8]], np.float32)
        masks = batch['lead_masks'][0].numpy()
        mask_rgb = np.maximum.reduce(masks[:, :, :, None]*colored[:, None, None], axis=0)
        overlay = rgb/255.*(1-masks.max(0)[:, :, None]*.7)+mask_rgb*.7
        overlay_image = Image.fromarray((overlay.clip(0, 1)*255).round().astype(np.uint8))
        # Boxes belong to individual printed instances, retaining repeated II.
        boxes = ImageDraw.Draw(overlay_image)
        for box, valid, lead in zip(batch['instance_boxes'][0], batch['instance_box_valid'][0],
                                    batch['instance_lead_ids'][0]):
            if valid:
                x0, y0, x1, y1 = box.tolist()
                color = tuple((colored[int(lead)]*255).astype(int))
                boxes.rectangle((x0, y0, x1-1, y1-1), outline=color, width=2)
        overlay_image.save(out/(name+'-targets.png'))
        draw.text((12, index*(tile_h+40)+12), name+' | actual 3D light transport', fill='#222')
        draw.text((tile_w+12, index*(tile_h+40)+12), '12 lead masks + printed-instance boxes', fill='#222')
        sheet.paste(Image.fromarray(rgb).resize((tile_w, tile_h), Image.Resampling.LANCZOS),
                    (0, index*(tile_h+40)+40))
        sheet.paste(overlay_image.resize((tile_w, tile_h), Image.Resampling.LANCZOS),
                    (tile_w, index*(tile_h+40)+40))
        sample = samples_from_batch(batch, source_ids=[source_hash], group_ids=['one-source'],
            sample_ids=[name], provenance=[{'source_sha256': source_hash, 'capture': batch['metadata']}])
        # A single source must stay in one partition. Do not fabricate a val set.
        export_yolo_dataset(sample, out/('yolo-'+name), val_fraction=0.)
        reports.append({'name': name, 'config': asdict(config), 'material_parameters':material,
                        'capture': batch['metadata'],
                        'visible_leads': int(batch['lead_box_valid'].sum()),
                        'visible_instances': int(batch['instance_box_valid'].sum()),
                        'rgb_sha256': hashlib.sha256((out/(name+'.png')).read_bytes()).hexdigest()})
    sheet.save(out/'pathtraced-gallery.jpg', quality=90, subsampling=0)
    report = {'schema': 'ecg-pathtrace-example-v1', 'source': source, 'source_sha256': source_hash,
              'fs': fs, 'clinical_validation': False, 'real_image_equivalence': False,
              'source_signal_preserved': torch.equal(canonical.reference, signals),
              'background': str(background) if background else None, 'cases': reports}
    (out/'pathtraced.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--spp', type=int, default=128)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--fs', type=float)
    parser.add_argument('--adverse', action='store_true', help='Irregular folds, liquid damage and missing material')
    parser.add_argument('--background', type=Path, default=Path(__file__).resolve().parents[1]/'assets/backgrounds/wood_table_001_diff_1k.jpg')
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = export_pathtraced(args.out, width=args.width, height=args.height, spp=args.spp,
                               input_path=args.input, fs=args.fs, background=args.background,adverse=args.adverse)
    print(json.dumps({'cases': len(report['cases']), 'signal_preserved': report['source_signal_preserved']}))
