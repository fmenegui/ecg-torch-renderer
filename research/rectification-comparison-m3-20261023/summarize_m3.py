"""Frozen descriptive postprocessing of a completed M3 comparison.

No inference, adaptation, model selection or gold-assisted quality tuning.
All scheduled rows must exist. Development and diagnostic panels stay separate.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


CORE_SHA = '8ed4cf91bc5b52c1c4bb1d0803d0f2644dbdc06573f799df8a73712b6f883e51'
BASELINE = 'our-B0'
EXPECTED_METHODS = {'our-B0', 'our-M2', 'npj-generic', 'npj-george-moody-2024', 'kaggle-third'}
TRACKS = ('primary', 'secondary_D4', 'unrestricted_finite_field', 'operational_RGB')
DOMAINS = ('grid', 'paper')
REPETITIONS, SEED = 20000, 1973
SUPPORTED_KAGGLE_LAYOUT = '3x4+r1'
REGRESSION_G, REGRESSION_W = -.01, .01


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_core(path):
    if digest(path) != CORE_SHA:
        raise ValueError('Independent-QA-approved metric core changed')
    spec = importlib.util.spec_from_file_location('m3_summary_metric_core', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def zero_operational(metric):
    result = dict(metric)
    for key in ('correct_queries', 'wrong_queries', 'claimed_queries', 'finite_claimed_queries',
                'claimed_nonfinite_queries'):
        result[key] = 0
    result.update(g=0., w=0., c=0., r=None, unclaimed_queries=result['queries'],
                  status='RGB_not_resampled_operational_failure', rmse_mm=None,
                  p50_mm=None, p95_mm=None, p99_mm=None, max_mm=None)
    result['coverage_scope'] = 'actual RGB dewarping, scheduled non-resampled case retained'
    return result


def metric_for(result, track, domain):
    if track == 'operational_RGB':
        metric = metric_for(result, 'unrestricted_finite_field', domain)
        return metric if result['rgb_resampled'] else zero_operational(metric)
    block = result['scores'][track]
    if track == 'unrestricted_finite_field' and block.get('status') != 'available':
        raise ValueError('Cannot silently invent missing unrestricted native footprint')
    return block['domains'][domain]


def registration_for(result, track):
    name = 'unrestricted_finite_field' if track == 'operational_RGB' else track
    return result['scores'][name]['registration']


def transformed_predictions(result, arrays, track, domain):
    coords = np.asarray(arrays[domain+'_coords'], dtype=float)
    finite = np.isfinite(coords).all(1)
    if track in ('unrestricted_finite_field', 'operational_RGB'):
        accepted = np.asarray(arrays[domain+'_native_footprint'], bool) & finite
        if track == 'operational_RGB' and not result['rgb_resampled']:
            accepted = np.zeros(len(coords), bool)
    else:
        accepted = np.asarray(arrays[domain+'_claimed_eligible'], bool) & finite
    registration = registration_for(result, track)
    if registration['status'] == 'registration_unavailable':
        return coords, np.zeros(len(coords), bool)
    matrix = np.asarray(registration['matrix'], dtype=float)
    offset = np.asarray(registration['offset_mm'], dtype=float)
    transformed = coords@matrix.T+offset
    return transformed, accepted & np.isfinite(transformed).all(1)


def common_support(result, baseline, arrays, baseline_arrays, query_reference, track, domain):
    current, accepted = transformed_predictions(result, arrays, track, domain)
    base, base_accepted = transformed_predictions(baseline, baseline_arrays, track, domain)
    common = accepted & base_accepted
    if not common.any():
        return {'common_queries': 0, 'common_rmse_mm': None, 'B0_common_rmse_mm': None}
    reference = np.asarray(query_reference)
    a = current[common]-reference[common]
    b = base[common]-reference[common]
    def rmse(residual):
        values = np.hypot(residual[:, 0], residual[:, 1])
        scale = float(values.max())
        return float(scale*np.sqrt(np.mean((values/scale)**2))) if scale else 0.
    return {'common_queries': int(common.sum()), 'common_rmse_mm': rmse(a),
            'B0_common_rmse_mm': rmse(b)}


def numeric_summary(values):
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {'count': len(valid), 'median': float(np.median(valid)) if valid else None,
            'p95': float(np.quantile(valid, .95)) if valid else None}


def case_diagnostics(result):
    meta = result.get('metadata', {})
    inverse = meta.get('inverse_map', {})
    complete = meta.get('native_output', {}).get('complete_geometry', {})
    return {
        'inference_and_RGB_seconds': meta.get('inference_and_RGB_seconds'),
        'study_wall_seconds': result.get('wall_seconds_including_forward_query_evaluation'),
        'execution_error': meta.get('execution_error', meta.get('error')),
        'zero_grid_nodes': meta.get('grid_after_interpolation_zero_nodes'),
        'observed_grid_nodes': meta.get('grid_before_interpolation_observed_nodes'),
        'invalid_inverse_cells': inverse.get('nonconvex_or_nonfinite_cells'),
        'convex_inverse_cells': inverse.get('convex_cells'),
        'native_output_shape': meta.get('output_shape', meta.get('native_rectified_output')),
        'native_geometry_status': complete.get('status', meta.get('status')),
        'native_geometry_reason': complete.get('reason'),
        'our_complete_map_proposal': complete.get('status') == 'full-frame-rectification',
    }


def load_results(root, schedule, methods, lock_hash):
    results, arrays, missing, receipts = {}, {}, [], []
    for row in schedule:
        key = row['key']
        results[key], arrays[key] = {}, {}
        for method in methods:
            folder = root/'cases'/key/method
            path = folder/'result.json'
            if not path.exists():
                missing.append({'key': key, 'method': method})
                continue
            result = read(path)
            for name in ('key', 'group_id', 'split', 'backend', 'layout', 'view'):
                if result.get(name) != row[name]:
                    raise ValueError('Case identity differs from frozen schedule: '+str(path))
            if result.get('method') != method or result.get('lock_sha256') != lock_hash:
                raise ValueError('Result belongs to a different method or execution lock: '+str(path))
            prediction_path = folder/'query-predictions.npz'
            if digest(prediction_path) != result['predictions_sha256']:
                raise ValueError('Archived predictions changed: '+str(path))
            with np.load(prediction_path, allow_pickle=False) as z:
                arrays[key][method] = {name: z[name].copy() for name in z.files}
            results[key][method] = result
            receipts.append({'path': str(path.relative_to(root)), 'sha256': digest(path),
                             'prediction_sha256': result['predictions_sha256']})
    if missing:
        raise ValueError('Incomplete panel, no full-panel statistics produced: '+json.dumps(missing))
    return results, arrays, receipts


def supported_description(rows):
    """Descriptive supported-layout subset, no success-conditioned filtering."""
    if not rows:
        return {'cases': 0, 'patients': 0, 'metrics': None,
                'scope': 'No scheduled cases meet the predeclared supported layout'}
    groups = sorted({r['group_id'] for r in rows})
    values, availability = [], {}
    for group in groups:
        subset = [r for r in rows if r['group_id'] == group]
        backends = sorted({r['backend'] for r in subset})
        availability[group] = backends
        backend_means = [np.mean([[r[k] for k in ('g', 'w', 'c')] for r in subset if r['backend'] == b], axis=0)
                         for b in backends]
        values.append(np.mean(backend_means, axis=0))
    g, w, c = np.mean(values, axis=0)
    return {'cases': len(rows), 'patients': len(groups), 'patient_backend_availability': availability,
            'metrics': {'g': float(g), 'w': float(w), 'c': float(c), 'r': float(w/c) if c else None},
            'scope': 'Patient means over available backends within predeclared supported layout, descriptive only, no bootstrap CI',
            'support_not_selected_from_success': True}


def run(root, out, metric_core, split='all'):
    core = load_core(metric_core)
    lock_path = root/'lock.json'
    lock, schedule_doc = read(lock_path), read(root/'schedule.json')
    methods = list(lock['methods'])
    full_schedule = schedule_doc['cases']
    schedule = full_schedule
    if set(methods) != EXPECTED_METHODS or len(methods) != 5 or len(schedule) != 56:
        raise ValueError('Expected frozen five-method56case panel with our-B0')
    if {r['view'] for r in schedule} != {0, 1, 2, 6}:
        raise ValueError('Unexpected schedule views')
    for panel_split, count, patients in [('validation', 24, 6), ('test', 32, 8)]:
        selected = [r for r in full_schedule if r['split'] == panel_split]
        if len(selected) != count or len({r['group_id'] for r in selected}) != patients:
            raise ValueError('Incomplete or changed split panel')
    locked_sources = set(lock['source_sha256'].values())
    if digest(__file__) not in locked_sources:
        raise ValueError('Summarizer source was not frozen in the execution lock')
    if digest(root/'schedule.json') not in locked_sources or digest(root/'protocol.json') not in locked_sources:
        raise ValueError('Copied schedule/protocol differ from locked source bytes')
    for name, expected in lock['query_sha256'].items():
        if digest(root/'queries'/name) != expected:
            raise ValueError('Frozen queries changed before postprocessing')
    if split not in ('all', 'validation', 'test'):
        raise ValueError('Unknown execution split')
    schedule = [r for r in full_schedule if split == 'all' or r['split'] == split]
    lock_hash = digest(lock_path)
    results, arrays, receipts = load_results(root, schedule, methods, lock_hash)
    records = []
    for case in schedule:
        key = case['key']
        with np.load(root/'queries'/(key+'.npz'), allow_pickle=False) as z:
            references = {d: z[d+'_reference'].copy() for d in DOMAINS}
        for method in methods:
            result, baseline = results[key][method], results[key][BASELINE]
            diag = case_diagnostics(result)
            for track in TRACKS:
                registration = registration_for(result, track)
                for domain in DOMAINS:
                    metric = metric_for(result, track, domain)
                    base_metric = metric_for(baseline, track, domain)
                    common = common_support(result, baseline, arrays[key][method], arrays[key][BASELINE],
                                            references[domain], track, domain)
                    record = {k: case[k] for k in ('key', 'group_id', 'split', 'backend', 'layout', 'view')}
                    record.update(method=method, track=track, domain=domain,
                                  status=result['status'], rgb_resampled=result['rgb_resampled'],
                                  registration_status=registration['status'],
                                  **{k: metric.get(k) for k in ('queries', 'g', 'w', 'c', 'r', 'rmse_mm', 'p50_mm',
                                                             'p95_mm', 'p99_mm', 'max_mm', 'claimed_nonfinite_queries',
                                                             'native_claimed_coverage', 'native_RGB_footprint_coverage',
                                                             'nonfinite_inside_native_RGB_footprint')},
                                  delta_g=metric['g']-base_metric['g'], delta_w=metric['w']-base_metric['w'],
                                  delta_c=metric['c']-base_metric['c'],
                                  delta_r=metric['r']-base_metric['r'] if metric['r'] is not None and base_metric['r'] is not None else None,
                                  **common, **diag)
                    record['diagnostic_regression_over_1pp'] = record['delta_g'] < REGRESSION_G or record['delta_w'] > REGRESSION_W
                    records.append(record)
    summary = {'date': '2026-10-23', 'schema': 'M3-descriptive-summary-v1',
               'scope': 'Retrospective synthetic diagnostic comparison, no new untouched holdout or SOTA claim',
               'baseline': BASELINE, 'methods': methods, 'bootstrap_repetitions': REPETITIONS,
               'bootstrap_seed': SEED, 'primary_domain': 'paper',
               'primary_geometry_gauge': 'translation-only with native scale and orientation',
               'secondary_D4': 'Diagnostic only, never primary ranking',
               'operational_RGB': 'No actual resampling means G=W=C0 operational, case retained. Finite inverse-field metrics do not certify folded nativeRGB regions.',
               'NaN_policy': 'Claimed-eligible nonfinite prediction is wrong in primary and D4 metrics',
               'supported_layout': SUPPORTED_KAGGLE_LAYOUT,
               'operating_points': 'Native trusted and unrestricted-finite operating points, not a calibrated continuous risk-coverage curve',
               'diagnostic_regression_thresholds': {'delta_g_below': REGRESSION_G, 'delta_w_above': REGRESSION_W, 'not_used_for_selection': True},
               'source_sha256': {'summarizer': digest(__file__), 'metric_core': digest(metric_core)},
               'input_sha256': {'lock': lock_hash, 'schedule': digest(root/'schedule.json')},
               'execution_split': split, 'frozen_pool_cases': len(full_schedule),
               'scored_cases': len(schedule),
               'unscored_pool_splits': [s for s in ('validation', 'test') if not any(r['split'] == s for r in schedule)],
               'development_feasibility': 'Any smoke-prefix results outside the selected complete split are not pooled into a development aggregate',
               'splits': {}}
    for panel_split in ('validation', 'test'):
        if not any(r['split'] == panel_split for r in schedule):
            continue
        split_records = [r for r in records if r['split'] == panel_split]
        part = {'role': 'development' if panel_split == 'validation' else 'diagnostic_already_inspected_M2_patients',
                'cases': len([c for c in schedule if c['split'] == panel_split]),
                'patients': len({r['group_id'] for r in split_records}), 'tracks': {}, 'method_diagnostics': {}}
        for track in TRACKS:
            part['tracks'][track] = {}
            for domain in DOMAINS:
                selected = [r for r in split_records if r['track'] == track and r['domain'] == domain]
                by_method = {m: [r for r in selected if r['method'] == m] for m in methods}
                bootstrap = core.paired_patient_bootstrap(by_method, BASELINE, repetitions=REPETITIONS, seed=SEED)
                diagnostic = {}
                for method, rows in by_method.items():
                    finite_tail = [r for r in rows if r['rmse_mm'] is not None]
                    diagnostic[method] = {
                        'aggregation': core.aggregate_patient_backend(rows),
                        'undefined_R_cases': sum(r['r'] is None for r in rows),
                        'registration_unavailable_cases': sum(r['registration_status'] == 'registration_unavailable' for r in rows),
                        'worst_finite_tail': sorted(finite_tail, key=lambda r: r['rmse_mm'], reverse=True)[:5],
                        'regressions_vs_B0_over_1pp': [r for r in rows if r['diagnostic_regression_over_1pp']],
                        'largest_G_regressions_vs_B0': sorted(rows, key=lambda r: r['delta_g'])[:5],
                        'largest_W_increases_vs_B0': sorted(rows, key=lambda r: r['delta_w'], reverse=True)[:5],
                        'predeclared_Kaggle_supported_layout': supported_description([r for r in rows if r['layout'] == SUPPORTED_KAGGLE_LAYOUT]),
                    }
                part['tracks'][track][domain] = {'bootstrap': bootstrap, 'methods': diagnostic}
        for method in methods:
            native = [results[c['key']][method] for c in schedule if c['split'] == panel_split]
            statuses = sorted({r['status'] for r in native})
            diagnostics = [case_diagnostics(r) for r in native]
            part['method_diagnostics'][method] = {
                'cases': len(native), 'status_counts': {s: sum(r['status'] == s for r in native) for s in statuses},
                'RGB_actually_resampled': sum(r['rgb_resampled'] for r in native),
                'RGB_not_resampled': sum(not r['rgb_resampled'] for r in native),
                'native_inference_and_RGB_seconds': numeric_summary([d['inference_and_RGB_seconds'] for d in diagnostics]),
                'study_wall_seconds_including_query_evaluation': numeric_summary([d['study_wall_seconds'] for d in diagnostics]),
                'timing_caution': 'Our native pipeline includes the shared fresh CNN time for each arm. Study wall reflects execution sharing and must not be substituted as standalone algorithm latency. Artifact I/O differs across native implementations.',
                'zero_grid_node_counts': [d['zero_grid_nodes'] for d in diagnostics if d['zero_grid_nodes'] is not None],
                'invalid_inverse_cell_counts': [d['invalid_inverse_cells'] for d in diagnostics if d['invalid_inverse_cells'] is not None],
                'output_shapes': [d['native_output_shape'] for d in diagnostics if d['native_output_shape'] is not None],
                'execution_errors': [d['execution_error'] for d in diagnostics if d['execution_error']],
            }
        summary['splits'][panel_split] = part
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    with (out/'cases.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    receipt = {'summarizer_sha256': digest(__file__), 'metric_core_sha256': digest(metric_core),
               'lock_sha256': lock_hash, 'result_files': receipts,
               'output_sha256': {name: digest(out/name) for name in ('summary.json', 'cases.csv')},
               'expected_case_method_results': len(schedule)*len(methods), 'actual_case_method_results': len(receipts),
               'case_method_track_domain_rows': len(records), 'selection_or_inference_changed': False}
    (out/'postprocessing-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--metric-core', type=Path, required=True)
    parser.add_argument('--split', choices=('all', 'validation', 'test'), default='all')
    args = parser.parse_args()
    receipt = run(args.root, args.out, args.metric_core, args.split)
    print(json.dumps({k: receipt[k] for k in ('expected_case_method_results', 'actual_case_method_results', 'case_method_track_domain_rows', 'output_sha256')}))
