"""Frozen-model inference on a preselected external annotation queue.

Unannotated captures receive predictions, never invented quantitative accuracy.
Overlays are local review artifacts and must not be redistributed by default.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.grid_benchmark import (
    BenchmarkConfig,
    GridUNet,
    evaluate_external,
    load_external,
)


def review(evaluated, external, cohort_sources, out):
    evaluated, external, cohort_sources, out = map(Path, (evaluated, external, cohort_sources, out))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    previous = json.loads((evaluated/'report.json').read_text())
    if previous['status'] != 'completed':
        raise ValueError('Completed parent benchmark required')
    config = BenchmarkConfig(**previous['configuration'])
    samples = load_external(external, config)
    excluded = {r['group_id'] for r in json.loads(cohort_sources.read_text())['records']}
    if any(sample['group_id'] in excluded for sample in samples):
        raise ValueError('External patient leakage')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    report = {'status': 'running', 'training_or_tuning': False, 'images': len(samples),
        'patients': len({s['group_id'] for s in samples}),
        'manifest_sha256': hashlib.sha256(external.read_bytes()).hexdigest(),
        'parent_report_sha256': hashlib.sha256((evaluated/'report.json').read_bytes()).hexdigest(),
        'annotation_scope': 'Unannotated convenience sample. Counts are not accuracy metrics', 'models': []}
    for row in previous['models']:
        name = row['name']
        model = GridUNet()
        checkpoint = evaluated/(name+'.pt')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row['checkpoint_sha256']:
            raise ValueError('Parent checkpoint SHA256 mismatch')
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu')['state_dict'])
        model.eval()
        trial = BenchmarkConfig(**row['target'])
        result = evaluate_external(model, samples, trial, row['threshold'], out, name)
        report['models'].append({'name': name, 'threshold': row['threshold'],
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'external': result})
        print(json.dumps({'model': name, 'captures': len(result)}), flush=True)
    report['status'] = 'completed'
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('evaluated', 'external', 'cohort-sources', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    review(**vars(parser.parse_args()))
