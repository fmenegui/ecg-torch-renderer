"""Descriptive patient-grouped summary of the frozen next-generation study.

The default opens completed development results only. --include-test must be
explicitly requested after inference and selection are frozen. This program
never selects strengths, changes inference, or reads capture gold archives.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_REPETITIONS = 20000
BOOTSTRAP_SEED = 1973
DOMAINS = ('grid', 'paper')
SEVERE_DIAGNOSTIC_THRESHOLDS = {
    'delta_g_below': -.05, 'delta_w_above': .02,
    'delta_r_above': .05, 'delta_common_rmse_mm_above': .5,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def ratio(wrong, coverage):
    wrong, coverage = np.asarray(wrong), np.asarray(coverage)
    return np.divide(wrong, coverage, out=np.full(wrong.shape, np.nan), where=coverage > 0)


def metrics(gwc):
    return {'g': float(gwc[0]), 'w': float(gwc[1]), 'c': float(gwc[2]),
            'r': number(ratio(gwc[1], gwc[2]))}


def interval(values):
    valid = np.asarray(values)[np.isfinite(values)]
    return {'lower': float(np.quantile(valid, .025)) if len(valid) else None,
            'upper': float(np.quantile(valid, .975)) if len(valid) else None,
            'defined_bootstrap_replicates': len(valid)}


def patient_tensor(cases, methods, domains=DOMAINS):
    """Patient/backend means first, retaining every case within its group."""
    patients = sorted({case['group_id'] for case in cases})
    backends = sorted({case['backend'] for case in cases})
    values = np.empty((len(patients), len(backends), len(methods), len(domains), 3))
    for p, patient in enumerate(patients):
        for b, backend in enumerate(backends):
            subset = [c for c in cases if c['group_id'] == patient and c['backend'] == backend]
            if not subset:
                raise ValueError('Every study patient must contribute to every backend')
            for m, method in enumerate(methods):
                for d, domain in enumerate(domains):
                    rows = [c['methods'][method]['domains'][domain] for c in subset]
                    if any(not np.isfinite([v[k] for k in ('g', 'w', 'c')]).all() for v in rows):
                        raise ValueError('Nonfinite G/W/C in case metrics')
                    if any(abs(v['c']-v['g']-v['w']) > 1e-12 for v in rows):
                        raise ValueError('Coverage is not G+W')
                    values[p, b, m, d] = np.mean([[v[k] for k in ('g', 'w', 'c')] for v in rows], axis=0)
    return values, patients, backends


def common_error(cases, method, domain):
    """Squared error pooled within patient/backend, then equal group weights."""
    backends = sorted({c['backend'] for c in cases})
    result = {}
    backend_means = []
    for backend in backends:
        patient_mse = []
        paired = []
        total_queries, total_common = 0, 0
        groups = sorted({c['group_id'] for c in cases if c['backend'] == backend})
        for patient in groups:
            rows = [c['methods'][method]['domains'][domain] for c in cases
                    if c['backend'] == backend and c['group_id'] == patient]
            common = sum(v['common_with_B0_queries'] for v in rows)
            total_queries += sum(v['queries'] for v in rows)
            total_common += common
            if common:
                current = sum(v['common_with_B0_queries']*v['common_rmse_mm']**2 for v in rows
                              if v['common_with_B0_queries'])/common
                baseline = sum(v['common_with_B0_queries']*v['B0_common_rmse_mm']**2 for v in rows
                               if v['common_with_B0_queries'])/common
                patient_mse.append([current, baseline])
                paired.append({'group_id': patient, 'common_queries': common,
                               'method_rmse_mm': float(np.sqrt(current)),
                               'B0_rmse_mm_on_same_queries': float(np.sqrt(baseline))})
        mean = np.mean(patient_mse, axis=0) if patient_mse else np.array([np.nan, np.nan])
        if patient_mse:
            backend_means.append(mean)
        result[backend] = {
            'method_rmse_mm': number(np.sqrt(mean[0])),
            'B0_rmse_mm_on_same_queries': number(np.sqrt(mean[1])),
            'patients_with_common_support': len(patient_mse), 'total_patients': len(groups),
            'common_queries': total_common, 'fixed_queries': total_queries,
            'patient_values': paired,
        }
    overall = np.mean(backend_means, axis=0) if backend_means else np.array([np.nan, np.nan])
    return {'aggregation': 'pool squared residuals within each patient/backend, then equally weight patients and backends',
            'scope': 'conditional on nonempty common support, with missing groups explicitly counted',
            'overall': {'method_rmse_mm': number(np.sqrt(overall[0])),
                        'B0_rmse_mm_on_same_queries': number(np.sqrt(overall[1])),
                        'backends_with_common_support': len(backend_means)},
            'by_backend': result}


def method_counts(cases, method):
    rows = [c['methods'][method] for c in cases]
    refinements = [r['refinement'] for r in rows]
    timing = [r['geometry_seconds'] for r in rows]
    return {
        'cases': len(rows), 'full_mesh_proposals_before_RGB_sampling': sum(r['full_map'] for r in rows),
        'full_mesh_unavailable': sum(not r['full_map'] for r in rows),
        'partial_coordinates_available': {
            d: sum(not r['full_map'] and r['domains'][d]['c'] > 0 for r in rows) for d in DOMAINS},
        'no_eligible_coordinates': {d: sum(r['domains'][d]['c'] == 0 for r in rows) for d in DOMAINS},
        'refinement_applied': sum(r['refined_coordinates'] for r in rows),
        'detection_coordinates_preserved_exactly': sum(r['detection_coordinates_preserved_exactly'] for r in rows),
        'input_only_construction_fallback_to_B0': sum(r['construction_fallback_to_B0'] for r in rows),
        'construction_fallback_after_refinement_attempt': sum(
            r['construction_fallback_to_B0'] and r['refinement']['status'] == 'refined' for r in rows),
        'construction_fallback_with_refiner_unchanged': sum(
            r['construction_fallback_to_B0'] and r['refinement']['status'] == 'unchanged' for r in rows),
        'fallback_count_note': 'Runner fallback includes repeating a rejected B0 construction when the refiner already left points unchanged. Total is not the number of new refinement failures.',
        'crease_guard_triggered': sum(r.get('crease_guard_triggered', False) for r in refinements),
        'iteration_limit_reached': sum(r.get('iteration_limit_reached', False) for r in refinements),
        'budget_exhausted': sum(r.get('budget_exhausted', False) for r in refinements),
        'irls_converged': sum(r.get('converged') is True for r in refinements),
        'irls_not_run': sum(r.get('converged') is None for r in refinements),
        'geometry_seconds_median': float(np.median(timing)),
        'geometry_seconds_p95': float(np.quantile(timing, .95)),
        'timing_scope': 'geometry construction including refinement, excludes detector and RGB sampling',
    }


def paired_diagnostics(cases, method, domain):
    regressions = []
    deltas = []
    for case in cases:
        current = case['methods'][method]['domains'][domain]
        baseline = case['methods']['B0']['domains'][domain]
        delta = {k: current[k]-baseline[k] for k in ('g', 'w', 'c')}
        delta['r'] = current['r']-baseline['r'] if current['r'] is not None and baseline['r'] is not None else None
        delta['common_rmse_mm'] = (current['common_rmse_mm']-current['B0_common_rmse_mm']
                                   if current['common_rmse_mm'] is not None else None)
        deltas.append({'key': case['key'], **delta})
        reasons = []
        if delta['g'] < SEVERE_DIAGNOSTIC_THRESHOLDS['delta_g_below']:
            reasons.append('G regression greater than 5 percentage points')
        if delta['w'] > SEVERE_DIAGNOSTIC_THRESHOLDS['delta_w_above']:
            reasons.append('W increase greater than 2 percentage points')
        if delta['r'] is not None and delta['r'] > SEVERE_DIAGNOSTIC_THRESHOLDS['delta_r_above']:
            reasons.append('R increase greater than 5 percentage points')
        if (delta['common_rmse_mm'] is not None
                and delta['common_rmse_mm'] > SEVERE_DIAGNOSTIC_THRESHOLDS['delta_common_rmse_mm_above']):
            reasons.append('common-support RMSE increase greater than 0.5 mm')
        if reasons:
            regressions.append({'key': case['key'], 'group_id': case['group_id'],
                                'backend': case['backend'], 'reasons': reasons, 'delta': delta})
    return {'diagnostic_thresholds_only': SEVERE_DIAGNOSTIC_THRESHOLDS,
            'not_used_for_model_selection': True, 'severe_regressions': regressions,
            'cases_G_improves_over_1pp': sum(v['g'] > .01 for v in deltas),
            'cases_G_regresses_over_1pp': sum(v['g'] < -.01 for v in deltas),
            'worst_delta_G': min(deltas, key=lambda v: v['g']) if deltas else None,
            'worst_delta_W': max(deltas, key=lambda v: v['w']) if deltas else None}


def summarize_split(report):
    cases = report['cases']
    methods = list(cases[0]['methods'])
    if methods[0] != 'B0' or any(list(c['methods']) != methods for c in cases):
        raise ValueError('Every case must have the same methods with B0 first')
    tensor, patients, backends = patient_tensor(cases, methods)
    expected_patients = 6 if report['split'] == 'validation' else 8
    if len(patients) != expected_patients or len(cases) != expected_patients*8:
        raise ValueError('Study split is not complete')
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, len(patients), (BOOTSTRAP_REPETITIONS, len(patients)))
    by_scope = {'overall': tensor.mean(axis=1)}
    by_scope.update({b: tensor[:, i] for i, b in enumerate(backends)})
    output = {m: {'counts': method_counts(cases, m), 'domains': {}} for m in methods}
    for scope, patient_values in by_scope.items():
        point = patient_values.mean(axis=0)
        draws = patient_values[indices].mean(axis=1)
        draw_r = ratio(draws[..., 1], draws[..., 2])
        for m, method in enumerate(methods):
            for d, domain in enumerate(DOMAINS):
                current, baseline = metrics(point[m, d]), metrics(point[0, d])
                delta = {k: current[k]-baseline[k] for k in ('g', 'w', 'c')}
                delta['r'] = current['r']-baseline['r'] if current['r'] is not None and baseline['r'] is not None else None
                delta_ci = {k: interval(draws[:, m, d, k_index]-draws[:, 0, d, k_index])
                            for k_index, k in enumerate(('g', 'w', 'c'))}
                delta_ci['r'] = interval(draw_r[:, m, d]-draw_r[:, 0, d])
                absolute_ci = {k: interval(draws[:, m, d, k_index])
                               for k_index, k in enumerate(('g', 'w', 'c'))}
                absolute_ci['r'] = interval(draw_r[:, m, d])
                value = {'metrics': current, 'absolute_95_descriptive_interval': absolute_ci,
                         'delta_vs_B0': delta, 'delta_95_descriptive_interval': delta_ci}
                target = output[method]['domains'].setdefault(domain, {})
                if scope == 'overall':
                    target['overall'] = value
                    target['common_support_error'] = common_error(cases, method, domain)
                    target['paired_case_diagnostics'] = paired_diagnostics(cases, method, domain)
                else:
                    target.setdefault('by_backend', {})[scope] = value
                # Independent postprocessing must reproduce the evaluator's
                # published aggregation, without changing its selection.
                published = report['summary'][method][domain]
                expected = published['overall'] if scope == 'overall' else published['by_backend'][scope]
                for key, actual in current.items():
                    if actual is None:
                        if expected[key] is not None:
                            raise ValueError('Undefined risk differs from evaluator summary')
                    elif abs(actual-expected[key]) > 1e-11:
                        raise ValueError('Aggregation differs from evaluator: '+method+'/'+domain+'/'+scope+'/'+key)
    return {'cases': len(cases), 'patients': patients, 'independent_patient_count': len(patients),
            'backends': backends, 'methods': output,
            'cohort_role': ('previously inspected development cohort' if report['split'] == 'validation'
                            else 'new patient assessment with same renderer and frozen detector')}


def case_rows(report):
    output = []
    for case in report['cases']:
        for method, details in case['methods'].items():
            refinement = details['refinement']
            for domain, current in details['domains'].items():
                baseline = case['methods']['B0']['domains'][domain]
                row = {k: case[k] for k in ('key', 'group_id', 'backend', 'split', 'layout')}
                row.update(method=method, domain=domain)
                row.update({k: current.get(k) for k in ('queries', 'g', 'w', 'c', 'r', 'p95_mm', 'p99_mm',
                    'rmse_mm', 'common_with_B0_queries', 'common_rmse_mm', 'B0_common_rmse_mm')})
                row.update({'delta_'+k: current[k]-baseline[k] for k in ('g', 'w', 'c')})
                row['delta_r'] = current['r']-baseline['r'] if current['r'] is not None and baseline['r'] is not None else None
                row.update(full_mesh_proposal=details['full_map'],
                           construction_fallback_to_B0=details['construction_fallback_to_B0'],
                           construction_fallback_after_refinement_attempt=(
                               details['construction_fallback_to_B0'] and refinement['status'] == 'refined'),
                           construction_fallback_with_refiner_unchanged=(
                               details['construction_fallback_to_B0'] and refinement['status'] == 'unchanged'),
                           refinement_applied=details['refined_coordinates'],
                           refinement_status=refinement['status'], refinement_reason=refinement['reason'],
                           crease_guard_triggered=refinement.get('crease_guard_triggered', False),
                           budget_exhausted=refinement.get('budget_exhausted', False),
                           iteration_limit_reached=refinement.get('iteration_limit_reached', False),
                           irls_converged=refinement.get('converged'),
                           geometry_seconds=details['geometry_seconds'])
                output.append(row)
    return output


def run(root, out, include_test=False):
    selection_path, validation_path = root/'selection.json', root/'validation.json'
    selection = json.loads(selection_path.read_text())
    if digest(validation_path) != selection['validation_sha256']:
        raise ValueError('Validation differs from frozen selection')
    validation = json.loads(validation_path.read_text())
    if 'summary' not in validation:
        raise ValueError('Development evaluation is incomplete')
    for path, expected in selection['source_hashes'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Inference source changed after selection: '+path)
    reports = {'validation': validation}
    if include_test:
        # This is the sole assessment-data open in the summary program.
        test = json.loads((root/'test.json').read_text())
        if 'summary' not in test or len(test.get('cases', [])) != 64:
            raise ValueError('Assessment evaluation is incomplete')
        for key in ('source_hashes', 'checkpoint_sha256', 'data_protocol_sha256'):
            if test[key] != selection[key]:
                raise ValueError('Assessment differs from frozen selection: '+key)
        reports['test'] = test
    result = {
        'methodology': 'paired patient-cluster bootstrap, patient and backend balanced',
        'bootstrap_repetitions': BOOTSTRAP_REPETITIONS, 'bootstrap_seed': BOOTSTRAP_SEED,
        'bootstrap_unit': 'patient including every backend, image and correlated query from that patient',
        'interval_scope': 'descriptive conditional intervals, no multiplicity adjustment or population/clinical guarantee',
        'risk_definition': 'R is aggregate W/C, not mean of per-image or per-patient ratios',
        'delta_units': 'G/W/C/R in fractions, multiply by100 for percentage points',
        'selection': selection,
        'selection_changed_by_summary': False,
        'selected_method': ('B0' if selection['selected_strength'] == 0
                            else 'soft-'+str(selection['selected_strength'])),
        'new_neural_training': False, 'competitor_inference_executed': False,
        'data_limit': 'same renderer and fixed old detector, new patients do not establish external real-photo generalization',
        'input_sha256': {name+'.json': digest(root/(name+'.json')) for name in reports},
        'selection_sha256': digest(selection_path),
        'summary_source_sha256': digest(Path(__file__)),
        'splits': {name: summarize_split(report) for name, report in reports.items()},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    rows = [row for report in reports.values() for row in case_rows(report)]
    with (out/'cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--include-test', action='store_true')
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({'selected_method': result['selected_method'],
                      'splits': {k: {'cases': v['cases'], 'patients': v['independent_patient_count']}
                                 for k, v in result['splits'].items()}}, indent=2))
