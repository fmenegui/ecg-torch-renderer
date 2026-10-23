"""Read-only replay of saved npj geometry and RGB, with no neural inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as nnf

from npj_adapter import DEFAULT_ROOT, NPJMap, module_from_file, tv_forward_matrix


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root):
    root = Path(root)
    repository = root/'ecg-torch-renderer'
    comparison = repository/'build/m3-comparison'
    bank = repository/'build/nextgen-bank'
    manifest = {r['key']: r for r in json.loads((bank/'manifest.json').read_text())}
    keys = ['220288-0', '208777-0', '220288-2', '208777-2', '220288-6', '208777-6']
    cropper_class = module_from_file('npj_saved_cropper', DEFAULT_ROOT/'src/model/cropper.py').Cropper
    rows = []
    for key in keys:
        source_path = bank/manifest[key]['file']
        with np.load(source_path) as archive:
            original_rgb = archive['image'].copy()
            ids = archive['ids'].copy()
        with np.load(comparison/'queries'/f'{key}.npz') as archive:
            queries = {k: archive[k].copy() for k in archive.files}
        for method in ['npj-generic', 'npj-george-moody-2024']:
            folder = comparison/'cases'/key/method
            result_path = folder/'result.json'
            result = json.loads(result_path.read_text())
            meta = result['metadata']
            cropper = cropper_class(**meta['geometry_parameters'])
            tensor = torch.from_numpy(original_rgb.copy()).permute(2, 0, 1)[None].float()
            tensor = (tensor-tensor.min())/(tensor.max()-tensor.min())
            h, w = original_rgb.shape[:2]
            rh, rw = meta['resized_shape']
            if (rh, rw) != (h, w):
                tensor = nnf.interpolate(tensor, size=(rh, rw), mode='bilinear',
                    align_corners=False, antialias=min(h, w) >= 512)
            corners = torch.tensor(meta['source_points_in_resized_pixel_indices'], dtype=torch.float32)
            destination = cropper._calculate_destination_points(rh, rw, corners)
            sx, sy = rw/w, rh/h
            resized_matrix = np.array([[sx, 0., (sx-1)/2], [0., sy, (sy-1)/2], [0., 0., 1.]])
            matrix = tv_forward_matrix(corners, destination)@resized_matrix
            # This calls the unchanged original sampler, not the exported map.
            replay = cropper.apply_perspective(tensor, corners, fill_value=0)
            if meta['clockwise_portrait_rotation']:
                old_h = replay.shape[2]
                replay = torch.rot90(replay, k=3, dims=(2, 3))
                matrix = np.array([[0., -1., old_h-1], [1., 0., 0.], [0., 0., 1.]])@matrix
            y1, y2 = meta['cropped_rows_in_rotated_canvas']
            replay = replay[:, :, y1:y2+1]
            matrix = np.array([[1., 0., 0.], [0., 1., -y1], [0., 0., 1.]])@matrix
            replay_rgb = np.clip(np.rint(replay[0].permute(1, 2, 0).numpy()*255), 0, 255).astype('uint8')
            saved_rgb = np.array(Image.open(folder/'rectified-native.png'))
            with np.load(folder/'native_geometry.npz') as archive:
                native = {k: archive[k].copy() for k in archive.files}
            mapping = NPJMap(matrix, meta['mm_per_pixel_xy'], meta['source_shape'], meta['output_shape'])
            with np.load(folder/'query-predictions.npz') as archive:
                predictions = {k: archive[k].copy() for k in archive.files}
            prediction_checks = {}
            for domain in ['anchors', 'grid', 'paper']:
                predicted, eligible = mapping.predict(queries[domain+'_source'])
                prediction_checks[domain] = {
                    'maximum_coordinate_difference_mm': float(np.max(np.abs(predicted-predictions[domain+'_coords']))),
                    'eligibility_exact': bool(np.array_equal(eligible, predictions[domain+'_claimed_eligible'])),
                    'native_eligible_queries': int(eligible.sum()),
                }
            reg = result['scores']['primary']['registration']
            primary = result['scores']['primary']['domains']['grid']
            secondary = result['scores']['secondary_D4']['domains']['grid']
            left = float(corners[0, 0]+corners[3, 0])
            right = float(corners[1, 0]+corners[2, 0])
            top = float(corners[0, 1]+corners[1, 1])
            bottom = float(corners[2, 1]+corners[3, 1])
            rows.append({
                'key': key, 'method': method, 'layout': result['layout'], 'backend': result['backend'],
                'official_preset': meta['official_preset'], 'official_config_sha256': meta['official_config_sha256'],
                'source_archive_hash_matches': digest(source_path) == result['input_archive_sha256'],
                'result_sha256': digest(result_path),
                'recomputed_destination_exact': bool(np.array_equal(destination.numpy(), meta['destination_points_corner_convention'])),
                'maximum_source_to_pixels_difference': float(np.max(np.abs(matrix-native['source_to_pixels']))),
                'maximum_source_to_mm_difference': float(np.max(np.abs(mapping.source_to_mm-native['source_to_mm']))),
                'native_RGB_replay_exact': bool(np.array_equal(replay_rgb, saved_rgb)),
                'native_RGB_maximum_difference_8bit': int(np.max(np.abs(replay_rgb.astype(int)-saved_rgb.astype(int)))),
                'predictions': prediction_checks,
                'source_quad_approximate_aspect': (right-left)/(bottom-top),
                'source_quad_vertical_extent_from_average_edges_px': (bottom-top)/2,
                'destination_x_range': [float(destination[:, 0].min()), float(destination[:, 0].max())],
                'mm_per_output_pixel_xy': meta['mm_per_pixel_xy'],
                'visible_major_grid_ID_min': ids.min(0).tolist(), 'visible_major_grid_ID_max': ids.max(0).tolist(),
                'primary_registration': reg['status'], 'reference_anchor_span_mm': reg['reference_span_mm'],
                'predicted_anchor_span_mm': reg['predicted_span_mm'],
                'primary_grid_G': primary['g'], 'primary_grid_C': primary['c'],
                'native_grid_C': primary['native_claimed_coverage'], 'primary_grid_RMSE_mm': primary['rmse_mm'],
                'D4_grid_RMSE_mm': secondary['rmse_mm'],
            })
    # Original quantile bounds can invert for probability concentrated in one bin.
    # This reproduces a failure family without claiming the cached NN had exactly this CDF.
    example = torch.zeros(80)
    example[20] = 1
    collapsed_bounds = cropper_class(granularity=80, percentiles=[.02, .98])._get_indices(example)
    passed = all(r['source_archive_hash_matches'] and r['recomputed_destination_exact']
        and r['maximum_source_to_pixels_difference'] == 0 and r['maximum_source_to_mm_difference'] == 0
        and r['native_RGB_replay_exact'] and all(p['maximum_coordinate_difference_mm'] == 0
            and p['eligibility_exact'] for p in r['predictions'].values()) for r in rows)
    return {'status': 'pass' if passed else 'fail', 'neural_inference_performed': False,
            'frozen_code_or_results_changed': False, 'selected_case_scope': 'six fixed diagnostic cases, not comparator selection',
            'adapter_sha256': digest(Path(__file__).with_name('npj_adapter.py')),
            'lock_sha256': digest(comparison/'lock.json'), 'cases': rows,
            'original_cropper_single_probability_bin_control': {
                'nonzero_bin': 20, 'returned_bounds': list(collapsed_bounds),
                'bounds_inverted': collapsed_bounds[0] > collapsed_bounds[1],
                'scope': 'demonstrates original bound degeneracy, does not prove exact cached feature-map distribution'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = run(args.root)
    target = Path(__file__).with_name('saved-fidelity-audit.json')
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'case_method_pairs': len(result['cases']),
                      'collapsed_bounds': result['original_cropper_single_probability_bin_control']}))
