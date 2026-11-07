"""Evaluate learned points and inferred topology before frozen ECG extraction.

Topology never receives grid truth. Ground-truth correspondence is used only
after inference to measure relative identity consistency on held-out sources.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.runner import _write_json
from ecg_gpu.grid_benchmark import GridUNet
from ecg_gpu.grid_point_learning import decode_grid_points
from ecg_gpu.grid_evaluation import match_grid_points
from examples.digitization_grid_rectification import grid_remap, save_rgb
from examples.digitization_npj_profiles import run
from examples.train_grid_rectifier import verified_captures


TRANSFER_CASES = ('standard', 'six-rows', 'offline-flat', 'offline-fold')

def infer_points(model, image, threshold):
    """Letterbox RGB to detector size and invert pixel-center resizing exactly."""
    height, width = image.shape[:2]
    ratio = min(512/width, 384/height)
    w, h = round(width*ratio), round(height*ratio)
    resized = Image.fromarray(image).resize((w, h), Image.Resampling.BILINEAR)
    canvas = np.full((384, 512, 3), 255, dtype=np.uint8)
    canvas[:h, :w] = np.asarray(resized)
    tensor = torch.from_numpy(canvas.copy()).permute(2, 0, 1)[None].float()/255
    with torch.inference_mode():
        heat, offsets = model(tensor)
        prediction = decode_grid_points(heat.sigmoid(), offsets, threshold=threshold,
                                         max_points=384*512)
    valid = prediction['valid'][0].numpy()
    xy = prediction['xy'][0].numpy()[valid]
    score = prediction['scores'][0].numpy()[valid]
    inside = ((xy >= 0).all(axis=1) & (xy <= [w-1, h-1]).all(axis=1))
    xy, score = xy[inside], score[inside]
    return (xy+.5)/[w/width, h/height]-.5, score


def topology_identity_score(predicted, topology, sample):
    """Evaluation only, modulo unknown origin and square-lattice symmetries."""
    if topology['status'] != 'accepted':
        return None
    selected = np.asarray(topology['selected_indices'])
    match = match_grid_points(predicted[selected], sample['xy'][0].numpy(), radius=1.5)
    inferred = np.asarray(topology['ids'])[match['prediction_indices']]
    truth = sample['ids'].numpy()[match['reference_indices']]
    best = 0
    for swap in (False, True):
        p = inferred[:, ::-1] if swap else inferred
        for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            if len(p):
                _, counts = np.unique(truth-p*signs, axis=0, return_counts=True)
                best = max(best, int(counts.max()))
    return {'matched_nodes': len(inferred), 'consistent_nodes': best,
            'relative_identity_consistency': best/len(inferred) if len(inferred) else None,
            'matching_uses_truth_for_evaluation_only': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--npj-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Choose a new output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = GridUNet().eval()
    model.load_state_dict(checkpoint['state_dict'])
    threshold = float(checkpoint['threshold'])
    report = {'schema': 'ecg-learned-grid-topology-v1', 'status': 'running',
              'checkpoint_sha256': sha256_file(args.checkpoint), 'threshold': threshold,
              'topology_uses_generator_ids': False, 'source_test': [], 'transfer': []}
    for sample in verified_captures(args.cache):
        if sample['split'] != 'test':
            continue
        image = (sample['image'][0].permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
        points, scores = infer_points(model, image, threshold)
        topology = organize_lattice(points, scores)
        report['source_test'].append({'key': sample['key'], 'backend': sample['backend'],
            'source_id': sample['source_id'], 'group_id': sample['group_id'],
            'topology': topology, 'identity_score': topology_identity_score(points, topology, sample)})
        _write_json(args.out/'report.json', report)
    manifest = json.loads((args.inputs/'manifest.json').read_text())
    if sha256_file(args.inputs/'source.npz') != manifest['source_sha256']:
        raise ValueError('Transfer source checksum mismatch')
    prepared = dict(manifest, cases=[])
    directory = args.out/'rectified-inputs'
    directory.mkdir()
    shutil.copy2(args.inputs/'source.npz', directory/'source.npz')
    for case in manifest['cases']:
        if case['name'] not in TRANSFER_CASES:
            continue
        record = {'name': case['name'], 'image_sha256': case['image_sha256']}
        try:
            if sha256_file(args.inputs/case['image']) != case['image_sha256']:
                raise ValueError('Transfer RGB checksum mismatch')
            image = np.asarray(Image.open(args.inputs/case['image']).convert('RGB'))
            points, scores = infer_points(model, image, threshold)
            topology = organize_lattice(points, scores)
            record['topology'] = topology
            np.savez_compressed(args.out/(case['name']+'-points.npz'), xy=points, scores=scores)
            if topology['status'] != 'accepted' or len(topology['cells']) < 16:
                raise ValueError('Insufficient reliable cells for waveform extraction')
            ids = np.asarray(topology['ids'])
            # Output density is an explicit display convention, not image truth.
            canonical = ids[:, ::-1]*40.
            size = np.maximum(512, (canonical.max(axis=0)+1).astype(int))[::-1]
            if max(size) > 4096:
                raise ValueError('Inferred extent exceeds diagnostic canvas limit')
            warped, valid = grid_remap(image, points[topology['selected_indices']], canonical,
                                       ids, tuple(size), allowed_cells=topology['cells'])
            digest = save_rgb(directory/case['image'], warped)
            entry = copy.deepcopy(case)
            entry.update(image_sha256=digest, rectification_arm='learned-relative-lattice')
            prepared['cases'].append(entry)
            record.update(status='rectified', supported_pixel_fraction=float(valid.mean()),
                          output_shape=size.tolist(), declared_pixels_per_mm=8.)
        except (ValueError, RuntimeError) as error:
            record.update(status='rejected', error=str(error))
        report['transfer'].append(record)
        _write_json(args.out/'report.json', report)
        print(json.dumps({'transfer': record['name'], 'status': record['status']}), flush=True)
    _write_json(directory/'manifest.json', prepared)
    if prepared['cases']:
        report['digitization'] = run(directory, args.out/'npj', args.npj_source,
                                      arms=('configured',), dewarping=False)
    else:
        report['digitization'] = None
    report['status'] = 'completed'
    _write_json(args.out/'report.json', report)
