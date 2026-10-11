"""Generate every documentation figure from the actual public renderer.

Run from the repository root after installing the package. Images use entirely
artificial waveforms. Recipes include their configuration, parameters and seed.
Encoding happens after rendering and is never part of the training hot path.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from ecg_gpu import ECGRenderer, RenderConfig, LayoutSpec, DocumentAssets, __version__
from ecg_gpu.demo import demo_signals
from ecg_gpu.gpu_document import SURFACES, BACKGROUNDS, SHADOWS


BASE = {
    'layout': '3x4+r1', 'speed_index': 1, 'gain_index': 1, 'sequential': True,
    'quarter': 0, 'angle': .04, 'zoom': .82, 'translate_x': 0., 'translate_y': 0.,
    'perspective_x': .10, 'perspective_y': -.035, 'radial': .006,
    'curl_x': 0., 'curl_y': 0., 'crease_warp': 0., 'crease_strength': 0.,
    'surface_mode': 'flat', 'surface_strength': .95, 'relief_view': [.52, -.30],
    'paper_palette': 0, 'paper_tint': 0., 'paper_age': 0., 'paper_noise': .005,
    'palette': 0, 'grid_strength': .30, 'minor_strength': .27, 'grid_width': .28,
    'grid_present': 1., 'grid_style': 0, 'line_width': 1.25,
    'pulse_present': 1., 'label_present': 1., 'lead_present': 1.,
    'label_below': False, 'header_present': 1., 'font_size': 12.,
    'ink_color': [.065, .060, .058], 'ink_density': .96, 'thermal_fade': .04,
    'coffee_strength': 0., 'bleed_strength': 0., 'stain_strength': 0., 'band_strength': 0.,
    'exposure': .98, 'light_x': .035, 'light_y': -.025, 'shadow_strength': 0.,
    'vignette': .03, 'ambient_light': .61, 'flash_strength': .018,
    'specular_strength': .025, 'ambient_occlusion': .22,
    'white_balance': [1., .985, .975], 'blur': .45, 'motion': .45,
    'motion_angle': .35, 'read_noise': .0015, 'shot_noise': .0015,
    'gamma': 1., 'jpeg_quality': 88., 'jpeg_active': 1.,
    'background_type': 'wood', 'background_angle': .12, 'object_shadow': 'none',
    'object_shadow_strength': .56, 'object_shadow_position': [.6, -.12],
    'object_shadow_size': [.7, .85], 'object_shadow_angle': .25,
    'object_shadow_softness': .025, 'cast_shadow_strength': .42,
    'cast_shadow_softness': .018, 'tear_active': 0., 'occlusion_active': 0.,
    'pen_active': 0., 'artifact_present': 0.,
}


def recipes(layout_names):
    items = []
    def add(key, category, title, description, parameters, photo=True):
        items.append(dict(id=key, category=category, title=title,
                          description=description, parameters=parameters, photo=photo))
    for name in layout_names:
        add('layout-' + name, 'Layouts', name,
            'Canonical paper view. The layout name counts rows × columns. Rhythm strips use the full recording window.',
            {'layout': name}, photo=False)
    for name in SURFACES:
        add('paper-' + name, 'Paper', name.title(),
            'A controlled 2.5D paper surface with coherent normals, illumination and relief projection.',
            {'surface_mode': name, 'layout': '3x4+r3'})
    for name in BACKGROUNDS:
        add('table-' + name, 'Backgrounds', name.title(),
            'Procedural table material behind the same sheet of paper.', {'background_type': name})
    for name in SHADOWS:
        add('shadow-' + name, 'Lighting', name.title() + ' shadow',
            'An analytic foreground silhouette. This is a procedural shadow, not a photographed hand or phone.',
            {'object_shadow': name})
    colors = ['Ivory', 'Pink', 'Cream', 'Yellow', 'Blue', 'Green', 'Gray', 'Aged']
    for index, name in enumerate(colors):
        add('color-' + name.lower(), 'Paper', name + ' paper',
            'Paper palette selected independently of grid ink and table material.', {'paper_palette': index})
    for index, name in enumerate(['Lines', 'Major lines', 'Dots', 'Major dots']):
        add('grid-' + str(index), 'Print', name,
            'Grid style is sampled or fixed separately from the waveform.', {'grid_style': index})
    for index, name in enumerate(['Red', 'Blue', 'Green', 'Gray', 'Orange']):
        add('grid-color-' + str(index), 'Print', name + ' grid',
            'One of five registered grid-ink palettes.', {'palette': index})
    cases = [
        ('perspective', 'Geometry', 'Perspective', 'Projective capture with a tilted sheet.', {'perspective_x': .27, 'perspective_y': -.14}),
        ('rotation', 'Geometry', 'Rotation', 'Rotation preserves the physical time and voltage mapping.', {'angle': .25}),
        ('quarter-turn', 'Geometry', 'Quarter turn', 'Orientation is configurable independently of the layout.', {'quarter': 1}),
        ('crop', 'Geometry', 'Partial crop', 'Only visible paper can supervise image-space recovery.', {'zoom': 1.18, 'translate_x': .15}),
        ('radial', 'Geometry', 'Lens distortion', 'Radial displacement of the camera sampling field.', {'radial': .07}),
        ('defocus', 'Camera', 'Defocus', 'Optical blur softens the printed line and the grid together.', {'blur': 1.8}),
        ('motion', 'Camera', 'Motion blur', 'Directional blur from a moving capture.', {'motion': 6., 'motion_angle': .6}),
        ('noise', 'Camera', 'Sensor noise', 'Read and shot noise are applied in the camera model.', {'read_noise': .032, 'shot_noise': .025}),
        ('jpeg', 'Camera', 'JPEG artifacts', 'Block DCT quantization approximation inside Torch.', {'jpeg_quality': 25., 'jpeg_active': 1.}),
        ('exposure', 'Camera', 'Low exposure', 'Exposure and normal-based illumination combine before camera encoding.', {'exposure': .65}),
        ('white-balance', 'Camera', 'White balance', 'Channel gains simulate a warm capture.', {'white_balance': [1.14, .96, .83]}),
        ('vignette', 'Camera', 'Vignette', 'Spatial attenuation at the edge of the camera frame.', {'vignette': .28}),
        ('tear', 'Artifacts', 'Torn edge', 'A paper-edge mask removes the affected visible pixels.', {'tear_active': 1., 'tear_side': 0, 'tear_position': .2, 'tear_size': .2}),
        ('annotation', 'Artifacts', 'Printed annotation', 'Font-atlas text serves as a configurable annotation proxy.', {'artifact_present': [1., 0., 0., 0.], 'artifact_text': 0, 'artifact_x': .65, 'artifact_y': .45, 'artifact_angle': -.22}),
        ('pen', 'Artifacts', 'Pen stroke', 'A procedural ink stroke overlays the printed page.', {'pen_active': 1.}),
        ('coffee', 'Artifacts', 'Coffee stain', 'A procedural stain, with no photographed texture asset.', {'coffee_strength': .7}),
        ('bleed', 'Artifacts', 'Reverse-side bleed', 'A print-through approximation beneath the observed page.', {'bleed_strength': .22}),
        ('thermal-fade', 'Print', 'Thermal fading', 'Local line contrast changes without changing the source signal.', {'thermal_fade': .75}),
        ('occlusion', 'Artifacts', 'Opaque occlusion', 'An explicit occluder with a visibility mask.', {'occlusion_active': 1., 'occlusion_x': .45, 'occlusion_y': .4, 'occlusion_w': .25, 'occlusion_h': .08}),
        ('missing-leads', 'Print', 'Missing lead slots', 'Per-slot visibility can remove traces or labels without fabricating signal recovery.', {'lead_present': [1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.]}),
        ('no-grid', 'Print', 'Grid-free page', 'The signal line and calibration marks remain independently controlled.', {'grid_present': 0.}),
        ('no-labels', 'Print', 'Labels hidden', 'Lead labels can be sampled separately from trace visibility.', {'label_present': 0.}),
    ]
    for key, category, title, description, parameters in cases:
        add(key, category, title, description, parameters)
    for index, speed in enumerate([12.5, 25, 50]):
        add('speed-' + str(index), 'Calibration', str(speed) + ' mm/s',
            'The page width and recording duration define one shared pixel/mm scale.', {'speed_index': index})
    for index, gain in enumerate([5, 10, 20]):
        add('gain-' + str(index), 'Calibration', str(gain) + ' mm/mV',
            'Amplitude and the calibration pulse use the same voltage scale.', {'gain_index': index})
    add('simultaneous', 'Calibration', 'Simultaneous columns',
        'Every column begins at time zero. Sequential columns instead partition the recording window.', {'sequential': False}, photo=False)
    return items


def save_tensor(value, path):
    values = value.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    image = Image.fromarray((values * 255).round().astype(np.uint8))
    image.save(path, quality=92, subsampling=0)
    return image


def generate(out, device='cpu', compile_raster=True):
    out = Path(out)
    images = out / 'assets' / 'gallery'
    images.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device(device)
    values, fs, _ = demo_signals(fs=500, duration=10)
    signal = torch.tensor(values, device=device)[None]
    configs = {
        'photo': RenderConfig(height=768, width=1024, profile='photo', severity=.6, antialias_samples=3, shadow_samples=6),
        'paper': RenderConfig(height=768, width=1024, profile='clean', severity=0, antialias_samples=3, shadow_samples=0),
        'tall': RenderConfig(height=1152, width=1024, profile='clean', severity=0, antialias_samples=3, shadow_samples=0),
        'wide': RenderConfig(height=512, width=2048, profile='clean', severity=0, antialias_samples=3, shadow_samples=0),
    }
    renderers = {key: ECGRenderer(config, compile_raster=compile_raster).to(device) for key, config in configs.items()}
    items = recipes([spec.name for spec in renderers['paper'].layout_specs])
    for index, item in enumerate(items):
        kind = 'photo' if item['photo'] else 'paper'
        if not item['photo'] and item['parameters'].get('layout', '').startswith('12x1'):
            kind = 'tall'
        if item['parameters'].get('layout') == '1x12':
            kind = 'wide'
        renderer = renderers[kind]
        parameters = {**BASE, **item['parameters']}
        if not item['photo']:
            parameters.update(surface_mode='flat', quarter=0, angle=0., zoom=1.,
                              perspective_x=0., perspective_y=0., radial=0., jpeg_active=0.,
                              exposure=1., white_balance=[1., 1., 1.], read_noise=0., shot_noise=0.,
                              blur=.18, motion=0., object_shadow='none')
        seed = 62000 + index * 97
        batch = renderer(signal, fs, stage='image', photo=item['photo'], parameters=parameters,
                         generator=torch.Generator(device=device).manual_seed(seed))
        if not torch.isfinite(batch.image).all():
            raise AssertionError('Nonfinite gallery image: ' + item['id'])
        slug = re.sub(r'[^a-zA-Z0-9_-]', '-', item['id'])
        image = save_tensor(batch.image[0], images / (slug + '.jpg'))
        thumbnail = image.copy()
        thumbnail.thumbnail((480, 360), Image.Resampling.LANCZOS)
        thumbnail.save(images / (slug + '-thumb.jpg'), quality=86)
        item.update(image='assets/gallery/' + slug + '.jpg',
                    thumbnail='assets/gallery/' + slug + '-thumb.jpg',
                    seed=seed, configuration=asdict(configs[kind]), parameters=parameters,
                    image_sha256=hashlib.sha256((images / (slug + '.jpg')).read_bytes()).hexdigest())
        if index % 10 == 0:
            print(json.dumps({'generated': index + 1, 'total': len(items), 'id': item['id']}), flush=True)
    # Paired maps are computed in one observation, not illustrated by an image model.
    r = renderers['photo']
    paired = r(signal, fs, stage='all', return_scene=True, return_instances=True,
               return_canonical=True, return_field=True,
               parameters={**BASE, 'surface_mode': 'crumpled', 'layout': '3x4+r3'},
               generator=torch.Generator(device=device).manual_seed(1973))
    trace = paired.targets['traces'][0].amax(0)
    valid = paired.targets['valid'][0, 0]
    mask_views = {
        'target-rgb': paired.image[0],
        'target-traces': trace[None].expand(3, -1, -1),
        'target-valid': valid[None].expand(3, -1, -1),
        'target-overlay': torch.lerp(paired.image[0], torch.tensor([.05, .95, .45], device=device)[:, None, None].expand_as(paired.image[0]), (trace * valid)[None] * .7),
        'target-normals': (paired.scene['normals'][0] + 1) / 2,
        'target-height': (paired.scene['height'][0] / paired.scene['height'][0].max().clamp_min(1e-6)).expand(3, -1, -1),
        'target-canonical': paired.canonical_image[0],
    }
    for key, value in mask_views.items():
        image = save_tensor(value, images / (key + '.jpg'))
        image.thumbnail((480, 360), Image.Resampling.LANCZOS)
        image.save(images / (key + '-thumb.jpg'), quality=86)
        items.append(dict(id=key, title=key.replace('target-', '').title(), category='Targets',
                          description='A paired tensor from the same crumpled-paper observation.',
                          image='assets/gallery/' + key + '.jpg', thumbnail='assets/gallery/' + key + '-thumb.jpg',
                          parameters={**BASE, 'surface_mode': 'crumpled', 'layout': '3x4+r3'},
                          configuration=asdict(configs['photo']), seed=1973, photo=True))
    report = {'schema': 'ecg-renderer-documentation-gallery-v1', 'version': __version__, 'paper_model': 'material-paper-v1', 'device': str(device),
              'torch_version': str(torch.__version__), 'compile_raster': compile_raster,
              'source': 'entirely artificial waveforms', 'real_patient_records': 0,
              'clinical_validation': False, 'figures': len(items),
              'target_keys': list(paired.targets), 'metadata_keys': list(paired.metadata),
              'scene_keys': list(paired.scene), 'examples': items}
    (out / 'gallery.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'figures': len(items)}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('docs'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--compile-raster', action=argparse.BooleanOptionalAction, default=True)
    options = parser.parse_args()
    generate(options.out, options.device, options.compile_raster)
