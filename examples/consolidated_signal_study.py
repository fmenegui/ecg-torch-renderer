"""Frozen signal extraction on matched originals, oracle and selected RGB model."""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.field_rectifier import GridFieldNet, chart_lattice
from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.runner import _write_json
from examples.consolidated_grid_study import load_samples, predict, progress
from examples.digitization_grid_rectification import grid_remap, save_rgb
from examples.digitization_npj_profiles import run


def main(root, source, *, prepare_only=False):
    selected = json.loads((root/'selected-model.json').read_text())
    state = torch.load(root/selected['checkpoint'], map_location='cpu', weights_only=True)
    model = GridFieldNet(contrast=selected['variant'] != 'raw-point').cuda().eval()
    model.load_state_dict(state['state_dict'])
    samples = load_samples(root, json.loads((root/'capture-manifest.json').read_text()))
    cases = []
    pairs = json.loads((root/'signal-inputs.json').read_text())
    for patient in pairs:
        original = root/patient['original']
        learned = original.parent/'learned'
        learned.mkdir()
        shutil.copy2(original/'source.npz', learned/'source.npz')
        manifest = json.loads((original/'manifest.json').read_text())
        prepared = dict(manifest, cases=[])
        for entry in manifest['cases']:
            sample = next(s for s in samples if s['source_id'] == patient['source_id'] and s['view'] == 0 and s['backend'] == entry['name'])
            record = {'source_id': patient['source_id'], 'group_id': patient['group_id'], 'backend': entry['name']}
            xy, scores, chart, _ = predict(model, sample, threshold=selected['validation']['threshold'])
            topology = chart_lattice(xy, scores, chart) if selected['variant'] == 'contrast-global' else organize_lattice(xy, scores)
            record['topology'] = topology
            np.savez_compressed(learned/(entry['name']+'-points.npz'), xy=xy, scores=scores, chart=chart)
            try:
                if topology['status'] != 'accepted' or len(topology['cells']) < 16:
                    raise ValueError('Insufficient inferred cells')
                ids = np.asarray(topology['ids'])
                canonical = ids[:, ::-1]*40.
                size = np.maximum(512, (canonical.max(0)+1).astype(int))[::-1]
                if max(size) > 4096:
                    raise ValueError('Inferred extent exceeds canvas bound')
                with np.load(root/sample['file']) as data:
                    rgb = data['image']
                warped, support = grid_remap(rgb, xy[topology['selected_indices']], canonical, ids,
                    tuple(size), allowed_cells=topology['cells'])
                new = copy.deepcopy(entry)
                new['image_sha256'] = save_rgb(learned/entry['image'], warped)
                prepared['cases'].append(new)
                record.update(status='rectified', supported_pixel_fraction=float(support.mean()))
            except (ValueError, RuntimeError) as exc:
                record.update(status='rejected', error=str(exc))
            cases.append(record)
        _write_json(learned/'manifest.json', prepared)
    del model
    torch.cuda.empty_cache()
    _write_json(root/'learned-rectification.json', cases)
    if prepare_only:
        progress(root, 'learned-inputs-prepared')
        return
    outputs = []
    for i, patient in enumerate(pairs):
        for arm in ('original', 'oracle', 'learned'):
            inputs = root/patient['original'] if arm == 'original' else (root/patient['oracle'] if arm == 'oracle' else (root/patient['original']).parent/'learned')
            manifest = json.loads((inputs/'manifest.json').read_text())
            if not manifest['cases']:
                outputs.append(dict(patient, arm=arm, cases=[], status='no accepted geometry'))
                continue
            out = root/'signal-results'/patient['source_id']/arm
            result = run(inputs, out, source, device='cuda', arms=('configured',), dewarping=False)
            outputs.append(dict(patient, arm=arm, cases=result['cases'], status='completed'))
            _write_json(root/'signal-results.json', outputs)
            progress(root, 'signal-evaluation', patient=i+1, total=len(pairs), arm=arm)
    _write_json(root/'signal-results.json', outputs)
    progress(root, 'consolidated-study-completed')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    main(a.root, a.source, prepare_only=a.prepare_only)
