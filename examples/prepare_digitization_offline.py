"""Extend the declared online probe with real path-traced flat/folded papers."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu import ECGImageLayer, capture_scenario
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.gpu_cli import plain
from ecg_gpu.pathtrace import PathTraceConfig, pathtrace_capture


def prepare(inputs, out, *, spp=64):
    inputs, out = Path(inputs), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    manifest = json.loads((inputs/'manifest.json').read_text())
    if sha256_file(inputs/'source.npz') != manifest['source_sha256']:
        raise ValueError('Source signal checksum mismatch')
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs/'source.npz', out/'source.npz')
    for case in manifest['cases']:
        if sha256_file(inputs/case['image']) != case['image_sha256']:
            raise ValueError('Source PNG checksum mismatch')
        shutil.copy2(inputs/case['image'], out/case['image'])
    with np.load(inputs/'source.npz') as data:
        values, fs = data['values_mv'], int(data['fs'])
    for name, layout, surface, bend in (('offline-flat', '3x4+r1', 'flat', 0.),
                                        ('offline-fold', '6x2', 'fold', .4)):
        config, parameters = capture_scenario('clean_print', width=2048, height=1280, severity=.35)
        parameters.update(layout=layout, font=0, font_size=20., grid_phase_x=0., grid_phase_y=0.)
        layer = ECGImageLayer(fs=fs, config=config, parameters=parameters, seed=20261021, randomize=False).eval()
        canonical = layer.render_batch(torch.from_numpy(values)[None], stage='all',
            photo=False, as_texture=True, return_instances=True)
        capture_config = PathTraceConfig(width=2048, height=1280, samples_per_pixel=spp,
            surface=surface, bend_radians=bend, camera_origin=(.015, -.08, .37),
            field_of_view=48., environment_radiance=.2, key_radiance=4.)
        result = pathtrace_capture(canonical, config=capture_config, seed=20261021)
        rgb = (result['image'][0].permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
        filename = name+'.png'
        Image.fromarray(rgb).save(out/filename)
        manifest['cases'].append({'name': name, 'layout': layout, 'scenario': 'offline-'+surface,
            'width': 2048, 'height': 1280, 'seed': 20261021, 'config': asdict(config),
            'pathtrace_config': asdict(capture_config), 'capture_metadata': result['metadata'],
            'image': filename, 'image_sha256': sha256_file(out/filename),
            'metadata': plain(canonical.metadata)})
        print(json.dumps({'prepared': name, 'spp': spp}), flush=True)
        del layer, canonical, result
    manifest['schema'] = 'ecg-paper-adaptation-inputs-v1'
    manifest['parent_manifest_sha256'] = sha256_file(inputs/'manifest.json')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--spp', default=64, type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    prepare(args.inputs, args.out, spp=args.spp)
