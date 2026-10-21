"""Separate known-grid geometry from an RGB-only learned-grid control.

The oracle uses generator grid correspondences. The detected control uses
upstream grid probabilities and its own graph/TPS optimizer, without those
correspondences. Both remap RGB before a fresh frozen digitization pass, keeping
printed text and waveform in one coordinate system. Paper gains remain supplied.
"""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu import ECGImageLayer, capture_scenario
from ecg_gpu.gpu_renderer import RenderConfig
from ecg_gpu.pathtrace import PathTraceConfig, pathtrace_capture
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.runner import _write_json
from examples.digitization_npj_probe import load_upstream
from examples.digitization_npj_profiles import apply_profile, run


CASES = ('offline-flat', 'offline-fold')


def grid_remap(rgb, observed, canonical, ids, size):
    """Piecewise affine inverse sampling, without crossing missing grid cells.

    Coordinates denote pixel centers. Only triangles joining immediate lattice
    neighbors are supported. Outside the supported mesh stays white and false.
    This offline diagnostic is deliberately outside the torch training hot path.
    """
    import cv2
    from scipy.spatial import Delaunay
    arrays = [np.asarray(value, dtype=np.float64) for value in (observed, canonical, ids)]
    observed, canonical, ids = arrays
    if (any(a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all() for a in arrays)
            or not observed.shape == canonical.shape == ids.shape or len(ids) < 3):
        raise ValueError('Matching finite [N,2] correspondences required')
    if len(np.unique(ids, axis=0)) != len(ids) or not np.equal(ids, np.round(ids)).all():
        raise ValueError('Unique integer lattice identities required')
    height, width = size
    triangulation = Delaunay(canonical)
    vertices = ids[triangulation.simplices]
    supported = (np.ptp(vertices, axis=1) <= 1).all(axis=1)
    mapping = np.full((height, width, 2), -1., dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)
    for start in range(0, height, 64):
        y, x = np.mgrid[start:min(start+64, height), :width]
        query = np.stack((x.ravel(), y.ravel()), -1)
        simplex = triangulation.find_simplex(query)
        keep = (simplex >= 0) & supported[np.maximum(simplex, 0)]
        selected = simplex[keep]
        affine = triangulation.transform[selected]
        first = np.einsum('nij,nj->ni', affine[:, :2], query[keep]-affine[:, 2])
        weights = np.column_stack((first, 1-first.sum(axis=1)))
        sampled = np.einsum('ni,nij->nj', weights, observed[triangulation.simplices[selected]])
        inside = ((sampled >= 0).all(axis=1)
                  & (sampled <= [rgb.shape[1]-1, rgb.shape[0]-1]).all(axis=1))
        indices = np.flatnonzero(keep)[inside]
        mapping[start:start+len(y)].reshape(-1, 2)[indices] = sampled[inside]
        valid[start:start+len(y)].reshape(-1)[indices] = True
    warped = cv2.remap(rgb, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return warped, valid


def prepare(inputs, out, source):
    manifest = json.loads((inputs/'manifest.json').read_text())
    if sha256_file(inputs/'source.npz') != manifest['source_sha256']:
        raise ValueError('Reference checksum mismatch')
    with np.load(inputs/'source.npz') as data:
        values, fs = data['values_mv'], int(data['fs'])
    selected = [c for c in manifest['cases'] if c['name'] in CASES]
    if len(selected) != len(CASES):
        raise ValueError('Both declared offline cases are required')
    model, _, commit = load_upstream(source)
    apply_profile(model, model.identifier, model.identifier.layouts, True)
    model.apply_dewarping = True
    manifests = {arm: dict(manifest, cases=[]) for arm in ('oracle-grid', 'detected-grid', 'canonical')}
    evidence = {'schema': 'ecg-grid-rectification-inputs-v1', 'upstream_commit': commit,
                'detector_receives_generator_coordinates': False, 'cases': []}
    for arm in manifests:
        (out/arm).mkdir(parents=True)
        shutil.copy2(inputs/'source.npz', out/arm/'source.npz')
    for case in selected:
        if sha256_file(inputs/case['image']) != case['image_sha256']:
            raise ValueError('RGB checksum mismatch')
        rgb = np.asarray(Image.open(inputs/case['image']).convert('RGB'))
        # This branch receives only RGB. Known geometry is constructed later.
        torch.manual_seed(20261021)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].float()
        with torch.no_grad():
            prediction = model(tensor, layout_should_include_substring=None)
            aligned = prediction['aligned']['image']
            grid = model.dewarper.grid[None].to(aligned.device)
            rgb_warp = F.grid_sample(aligned, grid, align_corners=False)
            support = F.grid_sample(torch.ones_like(aligned[:, :1]), grid, align_corners=False)
            rgb_warp = torch.where(support >= .999, rgb_warp, torch.ones_like(rgb_warp))
        detected = (rgb_warp[0].permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
        print(json.dumps({'detected_grid': case['name'], 'points': len(model.dewarper.final_local_maxima)}), flush=True)

        _, printing = capture_scenario('clean_print', width=case['width'], height=case['height'], severity=.35)
        printing.update(layout=case['layout'], font=0, font_size=20., grid_phase_x=0., grid_phase_y=0.)
        layer = ECGImageLayer(fs=fs, config=RenderConfig(**case['config']), parameters=printing,
                              seed=case['seed'], randomize=False).eval()
        canonical = layer.render_batch(torch.from_numpy(values)[None], stage='all',
                                       photo=False, as_texture=True, return_instances=True)
        capture_config = PathTraceConfig(**case['pathtrace_config'])
        # Optical sampling changes cost, not the mesh or projected grid points.
        geometry = pathtrace_capture(canonical, config=replace(capture_config, samples_per_pixel=1),
                                    seed=case['seed'], return_grid_points=True)
        for key in ('mesh_sha256', 'canonical_texture_sha256', 'canonical_reference_sha256'):
            if geometry['metadata'][key] != case['capture_metadata'][key]:
                raise ValueError(f'Oracle regeneration mismatch: {key}')
        valid = geometry['grid_points_valid'][0].numpy()
        oracle, supported = grid_remap(rgb, geometry['grid_points_xy'][0].numpy()[valid],
            geometry['grid_points_canonical_xy'][0].numpy()[valid],
            geometry['grid_points_ids'][0].numpy()[valid], tuple(canonical.image.shape[-2:]))
        clean = (canonical.image[0].permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
        record = {'name': case['name'], 'original_sha256': case['image_sha256'],
                  'oracle_grid_nodes': int(valid.sum()), 'oracle_supported_pixel_fraction': float(supported.mean()),
                  'detected_grid_nodes': len(model.dewarper.final_local_maxima),
                  'canonical_is_new_clean_texture_not_recovered_rgb': True,
                  'mesh_sha256': geometry['metadata']['mesh_sha256']}
        for arm, image in (('oracle-grid', oracle), ('detected-grid', detected), ('canonical', clean)):
            path = out/arm/case['image']
            Image.fromarray(image).save(path)
            entry = copy.deepcopy(case)
            entry.update(image_sha256=sha256_file(path), rectification_arm=arm)
            manifests[arm]['cases'].append(entry)
        np.savez_compressed(out/'oracle-support-{}'.format(case['name']), valid=supported)
        evidence['cases'].append(record)
        _write_json(out/'geometry-evidence.json', evidence)
        del prediction, geometry, canonical, layer, aligned, rgb_warp
    for arm, content in manifests.items():
        _write_json(out/arm/'manifest.json', content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Choose a new output directory')
    torch.set_num_threads(2)
    prepare(args.inputs, args.out/'inputs', args.source)
    reports = {}
    for arm in ('original', 'oracle-grid', 'detected-grid', 'canonical'):
        location = args.inputs if arm == 'original' else args.out/'inputs'/arm
        reports[arm] = run(location, args.out/'results'/arm, args.source,
                           arms=('configured',), dewarping=False, case_names=CASES)
    _write_json(args.out/'report.json', {'schema': 'ecg-grid-rectification-v1',
                'neural_training_performed': False, 'arms': reports})
