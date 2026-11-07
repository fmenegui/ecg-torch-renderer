"""Repeat a locked image-only CPU comparison and retain partial-support facts.

Image returns are engineering outcomes, not geometric or waveform accuracy.
Every adapter receives the same decoded RGB bytes. No annotation is provided
to inference. References remain in their separately verified source folders.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.runner import _write_json
from examples.rectification_references import main as run_references
from examples.rectify_denoising_grid import rectify


def run(images, checkpoint, source, assets, out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty output directory')
    rows = json.loads(images.read_text())
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Nonempty, unique case identities required')
    for row in rows:
        if Path(row['id']).name != row['id'] or row['id'] in ('.', '..'):
            raise ValueError('Case identity must be a single safe path component')
        path = (images.parent/row['file']).resolve()
        if not path.is_relative_to(images.parent.resolve()):
            raise ValueError('Image paths must stay beneath the input manifest')
        if sha256_file(path) != row['sha256']:
            raise ValueError('Input bytes differ from locked manifest')
    out.mkdir(parents=True, exist_ok=True)
    ours = []
    for row in rows:
        started = time.perf_counter()
        record = {'id': row['id'], 'status': 'failed'}
        try:
            record.update(rectify(checkpoint, images.parent/row['file'],
                                  out/'ours'/row['id'], device='cpu', pixels_per_mm=2.))
        except Exception as exc:
            record['error'] = type(exc).__name__+': '+str(exc)
        record['wall_seconds'] = time.perf_counter()-started
        ours.append(record)
        _write_json(out/'ours.json', ours)
    references = run_references(images, out/'references', source, assets, device='cpu')
    summary = {'schema': 'cpu-rectification-support-audit-v1', 'device': 'cpu',
               'manifest_sha256': sha256_file(images), 'cases': len(rows),
               'ours_partial_outputs': sum(r['status'] == 'partial-rectification' for r in ours),
               'ours_full_frame_proposals': sum(r['status'] == 'rectification-proposal' for r in ours),
               'ours_original_preserved': sum(r['status'] == 'original-preserved' for r in ours),
               'reference_image_returns': {name: sum(r['method'] == name and r['status'] == 'returned'
                                                   for r in references) for name in ('kaggle', 'npj')},
               'complete_rectification_accuracy_measured': False,
               'waveform_accuracy_measured': False,
               'limits': ['No full-page gold annotation is present in this image manifest',
                          'Cell support measures the observed lattice extent only',
                          'Image-return counts cannot establish method superiority']}
    _write_json(out/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('images', 'checkpoint', 'source', 'assets', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    torch.set_num_threads(4)
    print(json.dumps(run(**vars(parser.parse_args()))))
