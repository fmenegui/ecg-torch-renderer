"""Image-only exploratory checks on existing physical captures.

No real image is used to select a threshold or train a model. No modified
source photographs are redistributed. Unannotated detections are not accuracy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.field_rectifier import GridFieldNet
from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.runner import _write_json
from examples.preview_hybrid_rectification import detect


def main(source, checkpoint, out, device='cpu'):
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    model = GridFieldNet(contrast=state['variant'] != 'raw-point').to(device).eval()
    model.load_state_dict(state['state_dict'])
    manifest = json.loads((source/'manifest.json').read_text())
    report = {'status': 'old-checkpoint-exploratory', 'checkpoint_sha256': sha256_file(checkpoint),
              'annotation': 'none', 'accuracy': None, 'cases': [], 'training_executed': False,
              'limits': ['Physical photographs/scans of prints from real ECG signals, not a clinical capture cohort',
                         'Four ECG records, twelve physical captures, no manually annotated grid gold standard',
                         'Unknown training overlap of public reference models',
                         'Source record 22 and all previously evaluated patients excluded',
                         'No adapted source images redistributed']}
    for row in manifest['cases']:
        if row['image_type'] not in ('color-scan', 'gray-scan', 'phone-photo') or row['image_id'].startswith('D0_000022'):
            continue
        path = source/row['image']
        if sha256_file(path) != row['image_sha256']:
            raise ValueError('Physical image checksum differs')
        rgb = np.asarray(Image.open(path).convert('RGB'))
        height, width = rgb.shape[:2]
        ratio = min(768/width, 512/height)
        w, h = round(width*ratio), round(height*ratio)
        canvas = np.full((512, 768, 3), 255, np.uint8)
        canvas[:h, :w] = np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR))
        xy, scores = detect(model, canvas, float(state['threshold']), device)
        inside = ((xy >= 0) & (xy <= [w-1, h-1])).all(1)
        xy, scores = xy[inside], scores[inside]
        topology = organize_lattice(xy, scores)
        original_xy = (xy+.5)/[w/width, h/height]-.5
        np.savez_compressed(out/(row['image_id']+'-points.npz'), xy=original_xy, scores=scores)
        report['cases'].append({'id': row['image_id'], 'patient_id': row['patient_id'],
                                'image_type': row['image_type'], 'image_sha256': row['image_sha256'],
                                'point_count': len(xy), 'topology': topology, 'accuracy': None})
        _write_json(out/'physical-checks.json', report)
        print(json.dumps({'id': row['image_id'], 'points': len(xy), 'topology': topology['status']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'checkpoint', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    torch.set_num_threads(4)
    main(**vars(parser.parse_args()))
