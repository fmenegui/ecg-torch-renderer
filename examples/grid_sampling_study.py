"""Paired cohort-wide offline sampling study, frozen models and thresholds.

CPU research driver, outside the rendering/training hot path. Saves captures,
predictions, per-image metrics, hashes and patient-paired bootstrap intervals.
"""
import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.grid_benchmark import (
    BenchmarkConfig,
    GridUNet,
    _capture,
    _decode,
    _overlay,
    _predict,
    load_external,
    load_sources,
)
from ecg_gpu.grid_evaluation import (
    evaluate_grid_tolerances,
    match_grid_points,
    summarize_grid_matches,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def aggregate(cases, radii, *, repeats=2000, seed=71):
    """Micro metrics and paired differences, resampling whole patient groups."""
    groups = sorted({case['group_id'] for case in cases})
    if not groups or type(repeats) is not int or repeats < 1:
        raise ValueError('Nonempty cases and positive bootstrap repeats required')
    draws = np.random.default_rng(seed).integers(len(groups), size=(repeats, len(groups)))
    expected_keys = {(case['group_id'], case['key']) for case in cases}
    summaries, bootstrap = [], {}
    for model in sorted({case['model'] for case in cases}):
        for spp in sorted({case['spp'] for case in cases}):
            subset = [c for c in cases if c['model'] == model and c['spp'] == spp]
            keys = [(c['group_id'], c['key']) for c in subset]
            if set(keys) != expected_keys or len(set(keys)) != len(keys):
                raise ValueError('Incomplete patient-paired comparison')
            for radius in radii:
                matches = [match_grid_points(c['predicted'], c['reference'], radius=radius) for c in subset]
                metrics = summarize_grid_matches(matches)
                counts = np.array([[sum(m[k] for c, m in zip(subset, matches) if c['group_id'] == g)
                                    for k in ('tp', 'fp', 'fn')] for g in groups])
                sampled = counts[draws].sum(1)
                denominator = 2*sampled[:, 0]+sampled[:, 1:].sum(-1)
                # Test pages have reference points. Undefined F1 is an error here.
                if (denominator == 0).any():
                    raise ValueError('Undefined F1 in a bootstrap draw')
                scores = 2*sampled[:, 0]/denominator
                bootstrap[model, spp, radius] = scores
                delta = np.concatenate([c['predicted'][m['prediction_indices']]-c['reference'][m['reference_indices']]
                                        for c, m in zip(subset, matches)])
                summaries.append({'model': model, 'spp': spp, 'radius_px': radius, **metrics,
                    'mean_dx_px': float(delta[:, 0].mean()) if len(delta) else None,
                    'mean_dy_px': float(delta[:, 1].mean()) if len(delta) else None,
                    'f1_patient_bootstrap_95': np.quantile(scores, [.025, .975]).tolist()})
    baseline_spp = min(c['spp'] for c in cases)
    paired = []
    for (model, spp, radius), scores in bootstrap.items():
        if spp == baseline_spp:
            continue
        low = bootstrap[model, baseline_spp, radius]
        high_metric = next(r['f1'] for r in summaries if (r['model'], r['spp'], r['radius_px']) == (model, spp, radius))
        low_metric = next(r['f1'] for r in summaries if (r['model'], r['spp'], r['radius_px']) == (model, baseline_spp, radius))
        paired.append({'model': model, 'spp': spp, 'reference_spp': baseline_spp, 'radius_px': radius,
                       'delta_f1': high_metric-low_metric,
                       'patient_bootstrap_95': np.quantile(scores-low, [.025, .975]).tolist()})
    return {'by_model_spp_tolerance': summaries, 'paired_differences': paired,
            'bootstrap': {'unit': 'patient, all views and SPP levels kept paired', 'groups': len(groups),
                          'repeats': repeats, 'seed': seed,
                          'limitation': 'Exploratory sampling uncertainty from four test patients, not training-seed or device uncertainty'}}


def run(cohort, original, evaluated, out, *, levels=(16, 32, 64, 128), external=None):
    import drjit as dr
    import mitsuba as mi
    dr.set_thread_count(2)
    torch.set_num_threads(2)
    cohort, original, evaluated, out = map(Path, (cohort, original, evaluated, out))
    previous = json.loads((evaluated/'report.json').read_text())
    if previous.get('status') != 'completed' or digest(cohort) != previous['signals_sha256']:
        raise ValueError('Completed benchmark with identical source cohort required')
    config = BenchmarkConfig(**previous['configuration'])
    levels = tuple(levels)
    if (len(levels) < 2 or len(set(levels)) != len(levels)
            or any(type(s) is not int or s < config.spp for s in levels) or config.spp not in levels):
        raise ValueError('Unique integer SPP levels including the original baseline required')
    levels = sorted(levels)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    records, fs = load_sources(cohort)
    selected = [(i, v) for i, r in enumerate(records) if r['split'] == 'test' for v in range(config.views)]
    if not selected:
        raise ValueError('Held-out test captures required')
    manifest = {r['key']: r for r in json.loads((original/'manifest.json').read_text())}
    for index, view in selected:
        key = f'{index:05d}-{view}-offline'
        entry = manifest[key]
        if digest(original/'captures'/(key+'.pt')) != entry['file_sha256']:
            raise ValueError('Original capture SHA256 mismatch')
        if any(entry[field] != records[index][field] for field in ('source_id', 'group_id', 'split', 'signal_sha256')):
            raise ValueError('Original capture source mismatch')
    external_samples = load_external(external, config) if external else []
    if any(s['group_id'] in {r['group_id'] for r in records} for s in external_samples):
        raise ValueError('External patient appears in source cohort')
    models, checksums = {}, {}
    for row in previous['models']:
        name = row['name']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in models:
            raise ValueError('Unique safe model names required')
        checkpoint = evaluated/(name+'.pt')
        checksums[name] = digest(checkpoint)
        if checksums[name] != row['checkpoint_sha256']:
            raise ValueError('Parent checkpoint SHA256 mismatch')
        model = GridUNet()
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu')['state_dict'])
        models[name] = (model.eval(), row['threshold'], row['target']['offsets'])
    out.mkdir(parents=True, exist_ok=True)
    (out/'captures').mkdir()
    (out/'predictions').mkdir()
    report = {'schema': 'grid-sampling-study-1', 'status': 'running',
        'study': 'paired evaluation sampling sensitivity, frozen weights and validation thresholds',
        'signals_sha256': digest(cohort), 'parent_report_sha256': digest(evaluated/'report.json'),
        'checkpoint_sha256': checksums,
        'code_sha256': {p: digest(Path(__file__).resolve().parents[1]/p) for p in
                        ('examples/grid_sampling_study.py', 'ecg_gpu/grid_benchmark.py', 'ecg_gpu/grid_evaluation.py')},
        'levels': levels, 'radii_px': [.25, .5, 1., 1.5], 'configuration': previous['configuration'],
        'thresholds': {name: threshold for name, (_, threshold, _) in models.items()},
        'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'mitsuba': mi.__version__, 'drjit': dr.__version__, 'platform': platform.platform(),
                    'processor': platform.processor(), 'device': 'cpu', 'threads': 2, 'cuda_available': torch.cuda.is_available()},
        'protocol': {'training_or_tuning': False, 'spp_schedule': 'ascending per source/view',
                     'timing': 'wall seconds for _capture, warm process, 1 discarded warmup, excludes serialization and inference. One timing per setting, not a throughput benchmark',
                     'reference': 'geometric output-pixel centers, same signal, camera, mesh, seed, validity and lattice IDs',
                     'matching': 'independent maximum-cardinality minimum-distance assignment at each tolerance'},
        'limitations': ['Four test patients, two views each, one training seed, one renderer seed per geometry',
                        'All models were trained at 16 SPP. This studies evaluation sampling, not retraining',
                        'Subpixel synthetic geometric error does not establish physical annotation accuracy',
                        'No waveform reconstruction or clinical generalization measured', 'No GPU timings'],
        'captures': [], 'per_image': []}
    write_json(out/'report.json', report)
    first_index, first_view = selected[0]
    warm_start = time.perf_counter()
    _capture(records[first_index], first_view, first_index, fs, config, 'offline')
    report['discarded_warmup_seconds'] = time.perf_counter()-warm_start
    cases = []
    for index, view in selected:
        key = f'{index:05d}-{view}-offline'
        original_path = original/'captures'/(key+'.pt')
        baseline = torch.load(original_path, weights_only=True, map_location='cpu')
        for spp in levels:
            started = time.perf_counter()
            sample = _capture(records[index], view, index, fs, replace(config, spp=spp), 'offline')
            seconds = time.perf_counter()-started
            for field in ('xy', 'canonical', 'valid', 'ids'):
                torch.testing.assert_close(sample[field], baseline[field], atol=0, rtol=0)
            sample['key'] = key
            capture_path = out/'captures'/(key+f'-spp{spp}.pt')
            torch.save(sample, capture_path)
            capture = {'key': key, 'source_id': sample['source_id'], 'group_id': sample['group_id'],
                       'spp': spp, 'generation_seconds': seconds, 'accepted_points': sample['xy'].shape[1],
                       'geometry_and_validity_identical': True, 'original_capture_sha256': digest(original_path),
                       'file': str(capture_path.relative_to(out)), 'sha256': digest(capture_path)}
            if spp == config.spp:
                capture['baseline_max_image_difference'] = float((sample['image']-baseline['image']).abs().max())
            report['captures'].append(capture)
            for name, (model, threshold, offsets) in models.items():
                heat, delta = _predict(model, sample, 'cpu', offsets)
                predicted, truncated = _decode(heat, delta, threshold)
                if truncated:
                    raise RuntimeError('Decoder truncation invalidates the study')
                reference = sample['xy'][0].numpy()
                prediction_path = out/'predictions'/(name+'-'+key+f'-spp{spp}.npz')
                np.savez_compressed(prediction_path, xy=predicted, reference_xy=reference)
                case = {'model': name, 'key': key, 'group_id': sample['group_id'], 'spp': spp,
                        'predicted': predicted, 'reference': reference}
                cases.append(case)
                for score in evaluate_grid_tolerances(predicted, reference):
                    report['per_image'].append({k: case[k] for k in ('model', 'key', 'group_id', 'spp')}
                        | score | {'predictions_truncated': False, 'prediction_file': str(prediction_path.relative_to(out)),
                                   'prediction_sha256': digest(prediction_path)})
                if index == first_index and view == first_view:
                    _overlay(sample, predicted, out/(name+f'-spp{spp}.png'))
            write_json(out/'report.json', report)
            print(json.dumps({'capture': key, 'spp': spp, 'seconds': round(seconds, 3),
                              'completed': len(report['captures']), 'total': len(selected)*len(levels)}), flush=True)
    report.update(aggregate(cases, report['radii_px']))
    if external:
        report['external_manifest_sha256'] = digest(external)
        report['external_tolerance_sweep'] = []
        for sample in external_samples:
            if not sample['xy'].shape[1]:
                report['external_tolerance_sweep'].append({'id': sample['key'], 'metrics': None,
                    'reason': 'No complete annotated ROI. Quantitative accuracy not estimated'})
                continue
            region = (np.asarray(sample['metadata']['annotation_region']).reshape(2, 2)+.5)*sample['scale']-.5
            for name, (model, threshold, offsets) in models.items():
                heat, delta = _predict(model, sample, 'cpu', offsets)
                predicted, truncated = _decode(heat, delta, threshold)
                if truncated:
                    raise RuntimeError('External decoder truncated')
                inside = ((predicted >= region[0]) & (predicted <= region[1])).all(-1)
                report['external_tolerance_sweep'].append({'id': sample['key'], 'model': name,
                    'annotation': 'Existing silver ROI, uncertainty uncalibrated, not independent expert gold',
                    'coordinate_units': 'resized output pixels', 'scale_xy': sample['scale'].tolist(),
                    'metrics': evaluate_grid_tolerances(predicted[inside], sample['xy'][0].numpy())})
    report['status'] = 'completed'
    write_json(out/'report.json', report)
    with (out/'metrics.csv').open('w', newline='') as stream:
        fields = list(report['per_image'][0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report['per_image'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cohort', 'original', 'evaluated', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--external', type=Path)
    parser.add_argument('--levels', type=int, nargs='+', default=[16, 32, 64, 128])
    run(**vars(parser.parse_args()))
