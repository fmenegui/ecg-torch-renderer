"""Read-only independent audit of saved M3 predictions and fixed denominators.

This never reruns a neural model and never changes a prediction or source.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

DOMAINS = ('anchors', 'grid', 'paper')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []

    def check(self, condition, description):
        self.checks += 1
        if not condition:
            self.failures.append(description)

    def nested_equal(self, expected, actual, location):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict) and set(expected) == set(actual), location+': keys')
            if not isinstance(actual, dict):
                return
            for key in expected.keys() & actual.keys():
                self.nested_equal(expected[key], actual[key], location+'/'+key)
        elif isinstance(expected, list):
            self.check(isinstance(actual, list) and len(expected) == len(actual), location+': list size')
            if isinstance(actual, list):
                for i, (x, y) in enumerate(zip(expected, actual)):
                    self.nested_equal(x, y, location+'/'+str(i))
        elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
            self.check(actual is not None and np.isclose(expected, actual, rtol=1e-11, atol=1e-12), location)
        else:
            self.check(expected == actual, location)


def origin(reference, predicted, claimed):
    selected = np.asarray(claimed, bool) & np.isfinite(predicted).all(axis=1)
    if selected.sum() < 3 or (np.max(reference[selected], axis=0)-np.min(reference[selected], axis=0) < 1).any():
        return None
    return np.median(reference[selected]-predicted[selected], axis=0)


def physical_metric(reference, predicted, claimed, offset):
    n = len(reference)
    if offset is None:
        return {'g': 0., 'w': 0., 'c': 0., 'r': None}
    finite = np.isfinite(predicted).all(axis=1)
    keep = np.asarray(claimed, bool) & finite
    delta = predicted[keep]+offset-reference[keep]
    distance = np.sqrt(np.sum(delta*delta, axis=1))
    correct = int(np.count_nonzero(distance <= .5))
    coverage = int(np.count_nonzero(claimed))
    wrong = coverage-correct
    return {'g': correct/n, 'w': wrong/n, 'c': coverage/n,
            'r': wrong/coverage if coverage else None}


def aggregate(rows):
    groups = sorted({r['group_id'] for r in rows})
    backends = sorted({r['backend'] for r in rows})
    patient_values = []
    for group in groups:
        values = []
        for backend in backends:
            matching = [r for r in rows if r['group_id'] == group and r['backend'] == backend]
            if not matching:
                raise ValueError('Missing scheduled patient/backend stratum')
            values.append(np.mean([[r[k] for k in ('g', 'w', 'c')] for r in matching], axis=0))
        patient_values.append(np.mean(values, axis=0))
    g, w, c = np.mean(patient_values, axis=0)
    return {'g': float(g), 'w': float(w), 'c': float(c), 'r': float(w/c) if c else None,
            'cases': len(rows), 'patients': len(groups), 'backends': backends}


def run(execution, split, destination, allow_partial=False):
    audit = Audit()
    lock_path = execution/'lock.json'
    lock = json.loads(lock_path.read_text())
    schedule = json.loads((execution/'schedule.json').read_text())['cases']
    cases = [r for r in schedule if split == 'all' or r['split'] == split]
    methods = lock['methods']
    for group in ('source_sha256', 'weight_sha256'):
        for name, expected in lock[group].items():
            audit.check(Path(name).exists() and digest(name) == expected, 'unchanged '+group+': '+name)
    for name, expected in lock.get('owned_file_sha256', {}).items():
        audit.check(digest(execution/name) == expected, 'owned file: '+name)
    for name, expected in lock['query_sha256'].items():
        audit.check(digest(execution/'queries'/name) == expected, 'query archive: '+name)
    core_path = Path(lock['runtime_paths']['metric_core'])
    spec = importlib.util.spec_from_file_location('independent_result_core_replay', core_path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    bank = Path(lock['runtime_paths']['bank'])
    rows, missing, overrun, native_overrun, images = [], [], [], [], []
    for scheduled in cases:
        key = scheduled['key']
        capture = bank/scheduled['file']
        audit.check(digest(capture) == scheduled['sha256'], 'capture SHA '+key)
        query_path = execution/'queries'/(key+'.npz')
        with np.load(query_path, allow_pickle=False) as data:
            queries = {d: {'source': data[d+'_source'].copy(), 'reference': data[d+'_reference'].copy()} for d in DOMAINS}
        source_image = None
        for method in methods:
            folder = execution/'cases'/key/method
            path = folder/'result.json'
            label = key+'/'+method
            if not path.exists():
                missing.append(label)
                continue
            result = json.loads(path.read_text())
            for field in ('key', 'group_id', 'split', 'backend', 'layout', 'view'):
                audit.check(result[field] == scheduled[field], label+': identity '+field)
            audit.check(result['method'] == method, label+': method')
            audit.check(result['lock_sha256'] == digest(lock_path), label+': execution lock')
            audit.check(result['input_archive_sha256'] == scheduled['sha256'], label+': input lineage')
            audit.check(result['queries_archive_sha256'] == lock['query_sha256'][key+'.npz'], label+': query lineage')
            predictions_path = folder/'query-predictions.npz'
            audit.check(digest(predictions_path) == result['predictions_sha256'], label+': prediction archive')
            with np.load(predictions_path, allow_pickle=False) as data:
                predictions = {d: {kind: data[d+'_'+kind].copy() for kind in
                                  ('coords', 'claimed_eligible', 'native_footprint')} for d in DOMAINS}
            for domain in DOMAINS:
                audit.check(predictions[domain]['coords'].shape == queries[domain]['reference'].shape, label+': '+domain+' shape')
                audit.check(predictions[domain]['claimed_eligible'].dtype == np.dtype(bool), label+': '+domain+' confidence boolean')
                audit.check(predictions[domain]['native_footprint'].dtype == np.dtype(bool), label+': '+domain+' footprint boolean')
            replay = core.score_predictions(predictions, queries)
            audit.nested_equal(replay, result['scores'], label+': scorer replay')
            anchor = predictions['anchors']
            offset = origin(queries['anchors']['reference'], anchor['coords'], anchor['claimed_eligible'])
            audit.check((offset is None) == (replay['primary']['registration']['status'] == 'registration_unavailable'), label+': independent origin availability')
            if offset is not None:
                audit.check(np.allclose(offset, replay['primary']['registration']['offset_mm'], rtol=0, atol=1e-12), label+': independent origin offset')
            for domain in ('grid', 'paper'):
                p, reference = predictions[domain], queries[domain]['reference']
                independent = physical_metric(reference, p['coords'], p['claimed_eligible'], offset)
                actual = replay['primary']['domains'][domain]
                audit.nested_equal(independent, {k: actual[k] for k in independent}, label+': independent primary '+domain)
                unrestricted = replay['unrestricted_finite_field']['domains'][domain]
                rgb_metric = {k: unrestricted[k] for k in ('g', 'w', 'c', 'r')} if result['rgb_resampled'] else {'g': 0., 'w': 0., 'c': 0., 'r': None}
                rows.append({**{k: scheduled[k] for k in ('key', 'group_id', 'backend', 'split', 'layout', 'view')},
                             'method': method, 'domain': domain, 'status': result['status'],
                             'rgb_resampled': result['rgb_resampled'], 'primary': independent,
                             'operational_RGB': rgb_metric, 'secondary_D4': {k: replay['secondary_D4']['domains'][domain][k] for k in ('g', 'w', 'c', 'r')},
                             'unrestricted_finite_field': {k: unrestricted[k] for k in ('g', 'w', 'c', 'r')}})
            elapsed = result['wall_seconds_including_forward_query_evaluation']
            audit.check(np.isfinite(elapsed) and elapsed >= 0, label+': recorded wall time')
            if elapsed > lock['case_timeout_seconds']:
                overrun.append({'case': label, 'seconds': elapsed, 'status': result['status']})
            native_elapsed = result['metadata'].get('inference_and_RGB_seconds')
            if native_elapsed is not None:
                audit.check(np.isfinite(native_elapsed) and native_elapsed >= 0, label+': recorded native pipeline time')
                if native_elapsed > lock['case_timeout_seconds']:
                    native_overrun.append({'case': label, 'seconds': native_elapsed, 'status': result['status']})
            if method.startswith('our-') and 'native_output' in result['metadata']:
                native = result['metadata']['native_output']
                image_path = folder/native['output_image']
            else:
                image_path = folder/'rectified-native.png'
            if result['rgb_resampled']:
                audit.check(image_path.exists(), label+': actual RGB image exists')
                if image_path.exists():
                    with Image.open(image_path) as image:
                        audit.check(image.width*image.height <= lock['RGB_output_budget_pixels'], label+': common RGB pixel budget')
                        images.append({'case': label, 'width': image.width, 'height': image.height, 'sha256': digest(image_path)})
            elif method.startswith('our-') and result['status'] == 'original_preserved_partial_coordinates':
                if source_image is None:
                    with np.load(capture, allow_pickle=False) as data:
                        source_image = data['image'].copy()
                with Image.open(image_path) as preserved:
                    audit.check(np.array_equal(np.asarray(preserved.convert('RGB')), source_image), label+': preserved original exact decoded RGB')
                for domain in DOMAINS:
                    audit.check(not predictions[domain]['native_footprint'].any(), label+': partial coordinates do not claim RGB footprint '+domain)
    if not allow_partial:
        audit.check(not missing, 'all scheduled case/method outputs exist')
    complete = not missing
    aggregates = {}
    if complete:
        for method in methods:
            selection = [r for r in rows if r['method'] == method and r['domain'] == 'paper']
            aggregates[method] = {'status_counts': dict(Counter(r['status'] for r in selection)),
                                  'RGB_actually_resampled': sum(r['rgb_resampled'] for r in selection)}
            for track in ('primary', 'secondary_D4', 'unrestricted_finite_field', 'operational_RGB'):
                aggregates[method][track] = aggregate([{**r, **r[track]} for r in selection])
    report = {'schema': 'M3-independent-saved-result-audit-v1',
              'status': 'fail' if audit.failures else 'pass' if complete else 'partial_pending',
              'checks': audit.checks, 'failures': audit.failures, 'execution': str(execution),
              'split': split, 'lock_sha256': digest(lock_path), 'protocol_sha256': digest(execution/'protocol.json'),
              'expected_case_method_results': len(cases)*len(methods),
              'actual_case_method_results': len(rows)//2, 'missing': missing,
              'wall_overruns_cooperative_deadline': overrun,
              'native_pipeline_overruns_including_shared_CNN_cost': native_overrun,
              'actual_resampled_image_count': len(images),
              'image_receipts': images, 'aggregates_primary_paper': aggregates,
              'scope': 'Exact source/weight/input/query/prediction lineage, every archived score replayed, primary registration and G/W/C/R independently computed, RGB file/budget and original-fallback checks. No neural rerun, real-camera truth or waveform-fidelity certification.',
              'reviewer_source_sha256': digest(__file__)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'checks', 'expected_case_method_results', 'actual_case_method_results', 'missing', 'failures')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--split', choices=('test', 'validation', 'all'), default='test')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = run(args.execution, args.split, args.out, args.allow_partial)
    sys.exit(1 if result['status'] == 'fail' else 0)
