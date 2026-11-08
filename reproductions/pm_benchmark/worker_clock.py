"""Bounded GPU clock ablation. No waveform references are acquired or scored."""

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ecg_gpu.digitization.npj_backend import NativeNpjBackend
from ecg_gpu.digitization.npj_trace import physical_clock_proposal
from ecg_gpu.digitization.selective import SignalProposal
from examples.prepare_pm_ecg_external import fetch_member
from reproductions.common import environment, sha256, write_json

from .contracts import join_pages

ROOT = Path(__file__).parent
ARMS = ['npj_native', 'physical_x_clock']


def validate_inputs(document):
    if set(document) != {'schema', 'exams', 'metadata_sha256', 'zip_index_sha256', 'public_axes', 'independent_accuracy_validation', 'new_training'}:
        raise ValueError('Image-only clock protocol required')
    if document['schema'] != 'physical-clock-image-only-inputs-v1' or document['new_training'] or document['independent_accuracy_validation']:
        raise ValueError('This invocation is a bounded integration ablation')
    expected_axes = {'fs': 500, 'samples': 5000, 'paper_speed_mm_s': 25., 'gain_mm_mv': 10.,
                     'origin_x_px': 0., 'origin_basis': 'aligned image frame zero, independently unverified as print zero'}
    if document['public_axes'] != expected_axes:
        raise ValueError('Frozen clock calibration changed')
    cases = document['exams']
    if len(cases) != 12 or sum(len(e['pages']) for e in cases) != 22:
        raise ValueError('The frozen 12-exam, 22-page pilot is required')
    ids = set()
    for case in cases:
        if set(case) != {'exam_id', 'pages'}:
            raise ValueError('Exam references or labels cannot enter inference')
        for page in case['pages']:
            if set(page) != {'image_id', 'image', 'image_sha256', 'archive_member'}:
                raise ValueError('Page references or labels cannot enter inference')
            path = Path(page['image'])
            if path.is_absolute() or '..' in path.parts or page['archive_member']['name'] != page['image'] or page['image_id'] in ids:
                raise ValueError('Unique safe archive image identities required')
            ids.add(page['image_id'])
    return cases


def infer(images, data, out, backend, axes, deadline):
    if any(set(c) != {'image_id', 'image', 'image_sha256'} for c in images):
        raise ValueError('Native inference accepts only sealed image identities')
    records, proposals = [], {}
    parity = False
    for case in images:
        name, identity = case['image_id'], case['image_sha256']
        folder = out / name
        folder.mkdir()
        if sha256(data / case['image']) != identity:
            raise ValueError('Sealed image bytes changed')
        rgb = np.asarray(Image.open(data / case['image']).convert('RGB'))
        values, statuses, log = {}, {}, io.StringIO()
        begin = time.perf_counter()
        with contextlib.redirect_stdout(log):
            try:
                if time.time() > deadline - 210:
                    raise TimeoutError('Bounded invocation time exhausted before page inference')
                direct, observation = backend.predict_observed(rgb, fs=axes['fs'], samples=axes['samples'], source_id=identity)
                if not parity:
                    plain = backend.predict(rgb, fs=axes['fs'], samples=axes['samples'], source_id=identity)
                    np.testing.assert_array_equal(plain.values_mv, direct.values_mv)
                    parity = True
                values['npj_native'], statuses['npj_native'] = direct, 'returned'
                np.savez_compressed(folder / 'raster-observations.npz', **observation.arrays)
                write_json(folder / 'raster-observations.json', observation.report)
                try:
                    candidate = physical_clock_proposal(backend, observation, source_id=identity, **axes)
                    values['physical_x_clock'], statuses['physical_x_clock'] = candidate, 'returned'
                except Exception as error:  # noqa: BLE001 - preserve failed page without fabricating samples
                    values['physical_x_clock'] = SignalProposal(np.full((12, axes['samples']), np.nan), axes['fs'], identity, None,
                        metadata={'error': repr(error), 'backend': 'physical-x-clock'})
                    statuses['physical_x_clock'] = 'failed'
            except AssertionError:
                raise
            except Exception as error:  # noqa: BLE001 - every scheduled page remains in the denominator
                for arm in ARMS:
                    values[arm] = SignalProposal(np.full((12, axes['samples']), np.nan), axes['fs'], identity, None,
                                                metadata={'error': repr(error), 'backend': arm})
                    statuses[arm] = 'failed'
        elapsed = time.perf_counter() - begin
        (folder / 'inference.log').write_text(log.getvalue())
        for arm in ARMS:
            proposal = values[arm]
            path = folder / (arm + '.npz')
            np.savez_compressed(path, values_mv=proposal.values_mv, fs=proposal.fs)
            records.append({'image_id': name, 'source_id': identity, 'arm': arm, 'status': statuses[arm],
                'prediction': str(path.relative_to(out)), 'prediction_sha256': sha256(path),
                'metadata': proposal.metadata, 'shared_native_forward_and_readout_seconds': elapsed})
            proposals[name, arm] = proposal
        write_json(out / 'page-progress.json', {'pages_completed': len(records) // 2,
                                              'scheduled_pages': len(images), 'records': records})
    return records, proposals, parity


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    root = p.parse_args().workspace
    out = root / 'outputs'
    execution = json.loads((root / 'execution.json').read_text())
    document = json.loads((ROOT / 'clock-inputs.json').read_text())
    cases = validate_inputs(document)
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required, no CPU substitution')
    torch.set_num_threads(2)
    receipt = {'schema': 'physical-clock-gpu-ablation-v1', 'status': 'started', 'environment': environment(),
        'reference_inputs_received': False, 'reference_scoring_performed': False, 'new_training': False,
        'protocol_sha256': sha256(ROOT / 'clock-inputs.json'), 'scheduled_exams': len(cases), 'scheduled_pages': 22}
    write_json(out / 'receipt.json', receipt)
    try:
        with (out / 'source-bootstrap.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'reproductions.bootstrap', '--method', 'npj', '--dest', str(root / 'sources')],
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=max(1, min(360, int(execution['deadline'] - time.time() - 210))))
        data = root / 'clock-image-only'
        data.mkdir()
        pages = [page for case in cases for page in case['pages']]

        def acquire(page):
            path = fetch_member(data, page['archive_member'])
            digest = sha256(path)
            if page['image_sha256'] and digest != page['image_sha256']:
                raise ValueError('Pinned original photo changed')
            return {'image_id': page['image_id'], 'image': page['image'], 'image_sha256': digest}

        with ThreadPoolExecutor(max_workers=4) as pool:
            image_only = list(pool.map(acquire, pages))
        write_json(out / 'sealed-image-only-manifest.json', image_only)
        backend = NativeNpjBackend(root / 'sources/npj', device='cuda', max_side=3000)
        records, proposals, parity = infer(image_only, data, out, backend, document['public_axes'], execution['deadline'])
        if not parity:
            raise ValueError('No successful native observer parity control')
        exams = []
        for case in cases:
            identity = hashlib.sha256(json.dumps([p['image_sha256'] for p in image_only if p['image_id'] in {q['image_id'] for q in case['pages']}]).encode()).hexdigest()
            for arm in ARMS:
                joined = join_pages([proposals[page['image_id'], arm] for page in case['pages']],
                    exam_source_id=identity, fs=500, samples=5000)
                path = out / (case['exam_id'] + '-' + arm + '.npz')
                np.savez_compressed(path, values_mv=joined.values_mv, fs=500)
                exams.append({'exam_id': case['exam_id'], 'arm': arm, 'prediction': path.name,
                    'prediction_sha256': sha256(path), 'metadata': joined.metadata,
                    'status': 'returned' if np.isfinite(joined.values_mv).any() else 'failed'})
        write_json(out / 'sealed-predictions.json', {'schema': 'physical-clock-prediction-seal-v1',
            'reference_used': False, 'records': records, 'exams': exams, 'native_observer_parity_bit_exact': parity})
        receipt.update(status='completed_inference', prediction_seal_sha256=sha256(out / 'sealed-predictions.json'),
                       native_observer_parity_bit_exact=parity)
    except BaseException as error:
        receipt.update(status='failed_with_preserved_outputs', error=repr(error))
        raise
    finally:
        write_json(out / 'receipt.json', receipt)


if __name__ == '__main__':
    main()
