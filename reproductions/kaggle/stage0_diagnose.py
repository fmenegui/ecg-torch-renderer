"""Image-only diagnosis of pinned native stage 0, never a waveform benchmark."""

import argparse
import contextlib
import io
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from reproductions.common import environment, sha256, write_json

SOURCE_ROOT = Path(__file__).parent
STAGE0_ASSETS = ['heng/stage0_common.py', 'heng/stage0_model.py',
                 'heng/640106434-0001.gridpoint_xy.npy', 'heng/weight/stage0-last.checkpoint.pth']


def native_source(assets):
    assets = Path(assets).resolve()
    pinned = json.loads((SOURCE_ROOT / 'asset-sources.json').read_text())
    entries = {r['path']: r for r in pinned['files']}
    for name in STAGE0_ASSETS:
        if sha256(assets / name) != entries[name]['sha256']:
            raise ValueError('Pinned stage-0 asset changed: ' + name)
    sys.path.insert(0, str(assets / 'heng'))
    import stage0_common
    return stage0_common, {name: entries[name]['sha256'] for name in STAGE0_ASSETS}


def geometry_control(native):
    """Independent noncollinear landmark fixture for the ORIGINAL postprocessor."""
    height, width = native.HEIGHT, native.WIDTH
    image = np.zeros((height, width, 3), np.uint8)
    marker = np.zeros((1, 14, height, width), np.float32)
    marker[:, 0] = 1.
    expected = native.REF_PT9.copy()
    for (x, y), label in zip(expected, [2, 3, 4, 6, 7, 8, 10, 11, 12]):
        x, y = round(float(x)), round(float(y))
        marker[0, :, y-2:y+3, x-2:x+3] = 0.
        marker[0, label, y-2:y+3, x-2:x+3] = 1.
    points, rotation = native.marker_to_keypoint(image, torch.tensor([[1.] + [0.] * 7]), torch.from_numpy(marker), 1.)
    _, matched, matrix = native.normalise_by_homography(image, points)
    source = np.array([p[:2] for p in matched], np.float32)
    reprojection = cv2.perspectiveTransform(source[None], matrix)[0]
    error = float(np.max(np.linalg.norm(reprojection - expected, axis=1)))
    if rotation != 0 or sum(int(p[-1]) for p in matched) != 9 or error > 1.5:
        raise AssertionError('Native postprocessor failed its independent landmark control')
    return {'schema': 'native-stage0-landmark-positive-control-v1', 'points': 9,
        'matched_points': 9, 'maximum_reprojection_error_px': error,
        'network_inference_performed': False, 'photographic_accuracy_claim': False}


def marker_observations(batch, output):
    """Exact native TTA arithmetic, saved for diagnostics without changing outputs."""
    marker, orientation = 0, 0
    for index, tta in enumerate(batch['tta']):
        mk, on = output['marker'][[index]], output['orientation'][index]
        if tta == 1:
            mk, on = torch.flip(mk, [2]).contiguous(), on[[4, 5, 6, 7, 0, 1, 2, 3]]
        elif tta == 2:
            mk, on = torch.flip(mk, [3]).contiguous(), on[[6, 7, 4, 5, 2, 3, 0, 1]]
        elif tta == 3:
            mk, on = torch.flip(mk, [2, 3]).contiguous(), on[[2, 3, 0, 1, 6, 7, 4, 5]]
        marker = marker + mk[..., :batch['sH'], :batch['sW']]
        orientation = orientation + on
    marker = marker / len(batch['tta'])
    return {'orientation_per_tta': output['orientation'].detach().cpu().numpy(),
        'orientation_native_average': (orientation / len(batch['tta'])).detach().cpu().numpy(),
        'marker_average_argmax': marker.argmax(1)[0].detach().cpu().numpy().astype(np.uint8),
        'marker_average_max_probability': marker.max(1).values[0].detach().cpu().numpy()}


def infer_tta_chunks(model, batch, chunk_size):
    """Retain all four native TTAs while bounding inference memory."""
    if type(chunk_size) is not int or not 1 <= chunk_size <= 4:
        raise ValueError('TTA chunk size must be in [1,4]')
    chunks = [model(dict(batch, image=batch['image'][i:i+chunk_size]))
              for i in range(0, len(batch['image']), chunk_size)]
    return {key: torch.cat([part[key] for part in chunks]) for key in ['marker', 'orientation']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True, help='Strict image-only manifest')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--tta-chunk-size', type=int, choices=[1, 2, 4], default=1)
    parser.add_argument('--resume', action='store_true', help='Keep every verified completed diagnostic case')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable, no substitution')
    if args.device == 'cpu' and args.tta_chunk_size != 1:
        raise ValueError('CPU diagnostic uses one TTA per forward after an 8 GiB full-batch OOM')
    images = json.loads(args.inputs.read_text())
    if not images or any(set(c) != {'image_id', 'image', 'image_sha256'} for c in images):
        raise ValueError('Only explicit image identities may enter stage-0 diagnosis')
    if len({c['image_id'] for c in images}) != len(images):
        raise ValueError('Unique image identities required')
    args.out.mkdir(parents=True, exist_ok=args.resume)
    native, hashes = native_source(args.assets)
    torch.set_num_threads(2)
    torch.manual_seed(20261030)
    model = native.Stage0Net(pretrained=False)
    state = torch.load(args.assets / STAGE0_ASSETS[-1], map_location='cpu', weights_only=True)
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval().to(args.device)
    model.output_type = ['infer']
    with torch.inference_mode():
        fixture = {'image': torch.randint(0, 256, (4, 3, 96, 128), dtype=torch.uint8)}
        batched, serial = model(fixture), infer_tta_chunks(model, fixture, 1)
        parity = {key: float((batched[key] - serial[key]).abs().max()) for key in batched}
        for key in batched:
            torch.testing.assert_close(batched[key], serial[key], atol=2e-5, rtol=2e-5)
    receipt = {'schema': 'native-stage0-diagnosis-v1', 'environment': environment(), 'device': args.device,
        'precision': 'float32', 'native_tta_count': 4, 'source_and_weights_modified': False,
        'tta_chunk_size': args.tta_chunk_size, 'small_batch_serial_maximum_absolute_difference': parity,
        'waveform_benchmark': False, 'gpu_fp16_failure_reproduction_claim': False,
        'reference_received': False, 'input_manifest_sha256': sha256(args.inputs),
        'assets': hashes, 'positive_geometry_control': geometry_control(native), 'cases': []}
    if args.resume:
        prior = json.loads((args.out / 'receipt.json').read_text())
        if any(prior[key] != receipt[key] for key in ['assets', 'device', 'precision', 'tta_chunk_size', 'input_manifest_sha256']):
            raise ValueError('Diagnostic resume profile changed')
        for old, expected in zip(prior['cases'], images):
            if (old['image_id'] != expected['image_id'] or old['image_sha256'] != expected['image_sha256']
                    or sha256(args.out / old['image_id'] / 'native-observations.npz') != old['observations_sha256']):
                raise ValueError('Partial diagnostic identity or observations changed')
        if len(prior['cases']) > len(images):
            raise ValueError('Partial receipt exceeds original schedule')
        receipt['cases'] = prior['cases']
        receipt['resumed_from_verified_partial_receipt'] = True
    write_json(args.out / 'receipt.json', receipt)
    for case in images[len(receipt['cases']):]:
        relative = Path(case['image'])
        if relative.is_absolute() or '..' in relative.parts or sha256(args.root / relative) != case['image_sha256']:
            raise ValueError('Original image identity changed')
        folder = args.out / case['image_id']
        folder.mkdir()
        image = cv2.imread(str(args.root / relative), cv2.IMREAD_COLOR_RGB)
        if image is None:
            raise ValueError('Unreadable original RGB')
        log, started = io.StringIO(), time.perf_counter()
        row = {'image_id': case['image_id'], 'image_sha256': case['image_sha256']}
        with contextlib.redirect_stdout(log), torch.inference_mode():
            batch = native.image_to_batch(image)
            output = infer_tta_chunks(model, batch, args.tta_chunk_size)
            observations = marker_observations(batch, output)
            np.savez_compressed(folder / 'native-observations.npz', **observations)
            row['orientation_class'] = int(observations['orientation_native_average'].argmax())
            row['native_orientation_distribution'] = observations['orientation_native_average'].tolist()
            rotated, points = native.output_to_predict(image, batch, output)
            row['native_landmarks'] = points
            row['nonzero_landmarks'] = sum(bool(x != 0 or y != 0) for x, y, *_ in points)
            coordinates = np.array([p[:2] for p in points])
            row['distinct_landmark_positions'] = len(np.unique(coordinates, axis=0))
            row['landmark_affine_rank'] = int(np.linalg.matrix_rank(np.column_stack([coordinates, np.ones(len(points))])))
            try:
                normalized, matches, matrix = native.normalise_by_homography(rotated, points)
                np.save(folder / 'native-homography.npy', matrix)
                cv2.imwrite(str(folder / 'native-normalized.png'), cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR))
                row.update(status='native_postprocessing_returned', native_matched_points=sum(int(p[-1]) for p in matches))
            except Exception as error:  # noqa: BLE001 - retain the exact native numerical exception
                row.update(status='native_postprocessing_failed', error=repr(error), traceback=traceback.format_exc())
        (folder / 'native.log').write_text(log.getvalue())
        row.update(seconds=time.perf_counter() - started,
                   observations_sha256=sha256(folder / 'native-observations.npz'))
        receipt['cases'].append(row)
        write_json(args.out / 'receipt.json', receipt)
        print(json.dumps({k: row[k] for k in ['image_id', 'status', 'orientation_class', 'nonzero_landmarks', 'seconds']}), flush=True)
        del output, observations, batch


if __name__ == '__main__':
    main()
