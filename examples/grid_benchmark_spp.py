"""Paired render-sampling sensitivity on one held-out capture, without tuning."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.grid_benchmark import (BenchmarkConfig, GridUNet, _capture, _decode,
                                    _overlay, _predict, load_sources)
from ecg_gpu.grid_evaluation import match_grid_points, summarize_grid_matches


def compare(cohort, original, evaluated, out, *, index=20, view=0, spp=128):
    import drjit as dr
    dr.set_thread_count(2)
    torch.set_num_threads(2)
    original, evaluated, out = Path(original), Path(evaluated), Path(out)
    previous = json.loads((evaluated/'report.json').read_text())
    if hashlib.sha256(Path(cohort).read_bytes()).hexdigest() != previous['signals_sha256']:
        raise ValueError('Different source cohort')
    records, fs = load_sources(cohort)
    if not 0 <= index < len(records) or records[index]['split'] != 'test':
        raise ValueError('Choose a held-out test record')
    base = BenchmarkConfig(**previous['configuration'])
    if not 0 <= view < base.views or spp <= base.spp:
        raise ValueError('Valid view and increased samples per pixel required')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new output directory')
    low = torch.load(original/'captures'/f'{index:05d}-{view}-offline.pt', weights_only=True)
    high = _capture(records[index], view, index, fs, replace(base, spp=spp), 'offline')
    torch.testing.assert_close(low['xy'], high['xy'], atol=1e-6, rtol=0)
    torch.testing.assert_close(low['valid'], high['valid'], atol=0, rtol=0)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for model_report in previous['models']:
        if not model_report['name'].startswith('baseline-'):
            continue
        model = GridUNet()
        saved = torch.load(evaluated/(model_report['name']+'.pt'), weights_only=True, map_location='cpu')
        model.load_state_dict(saved['state_dict'])
        model.eval()
        for sample, rays in ((low, base.spp), (high, spp)):
            heat, delta = _predict(model, sample, 'cpu', True)
            predicted, truncated = _decode(heat, delta, model_report['threshold'])
            metrics = summarize_grid_matches([match_grid_points(predicted, sample['xy'][0].numpy(), radius=base.match_radius)])
            results.append({'model': model_report['name'], 'spp': rays, 'truncated': truncated, **metrics})
            _overlay(sample, predicted, out/(model_report['name']+f'-spp{rays}.png'))
    result = {'study': 'paired sampling sensitivity, one held-out capture, no threshold retuning',
              'source_id': low['source_id'], 'group_id': low['group_id'], 'view': view,
              'same_point_coordinates_and_validity': True, 'results': results,
              'limitation': 'One image, one geometry and one trained seed. Not an aggregate benchmark'}
    (out/'spp.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cohort', 'original', 'evaluated', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    for key, default in (('index', 20), ('view', 0), ('spp', 128)):
        parser.add_argument('--'+key, type=int, default=default)
    print(json.dumps(compare(**vars(parser.parse_args()))))
