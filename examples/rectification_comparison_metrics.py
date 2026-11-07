"""M3 origin-invariant physical-map evaluation, independent of inference.

Gold enters query preparation and scoring only. Primary registration removes one
arbitrary translation per image. D4 is a separately labelled diagnostic.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
from pathlib import Path

import numpy as np


D4 = np.asarray([
    [[1, 0], [0, 1]], [[-1, 0], [0, 1]],
    [[1, 0], [0, -1]], [[-1, 0], [0, -1]],
    [[0, 1], [1, 0]], [[0, -1], [1, 0]],
    [[0, 1], [-1, 0]], [[0, -1], [-1, 0]],
], dtype=float)
DOMAINS = ('grid', 'paper')


def array_hash(values, dtype='<f8'):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


@contextmanager
def _archive(source):
    if isinstance(source, (str, Path)):
        with np.load(source, allow_pickle=False) as data:
            yield data
    else:
        yield source


def prepare_case_queries(bank_npz, cached_query_npz, key):
    """Reuse only frozen gold/query arrays, create a disjoint paper anchor pool.

    Returned anchors/grid/paper each have source and reference N-by-2 arrays.
    Adapters may receive source arrays only, after their inference has finished.
    """
    with _archive(bank_npz) as bank, _archive(cached_query_npz) as cached:
        material = np.asarray(bank['reference_mm'])
        valid = np.asarray(bank['valid'])
        if material.ndim != 3 or material.shape[-1] != 2 or valid.shape != material.shape[:2]:
            raise ValueError('Expected HxWx2 material field and matching validity')
        physical = (valid >= .99) & np.isfinite(material).all(-1)
        height, width = physical.shape
        reserved = np.zeros(height*width, dtype=bool)
        result = {'key': str(key)}
        for domain in DOMAINS:
            source = np.asarray(cached[domain+'_query'], dtype=float).copy()
            reference = np.asarray(cached[domain+'_reference'], dtype=float).copy()
            if source.ndim != 2 or source.shape[1:] != (2,) or reference.shape != source.shape or not len(source):
                raise ValueError('Frozen query domain must be nonempty N by2')
            if not np.isfinite(source).all() or not np.isfinite(reference).all():
                raise ValueError('Frozen query/reference must be finite')
            integer = source.astype(np.int64)
            if not np.array_equal(source, integer):
                raise ValueError('M2 frozen queries must be source pixel centres')
            x, y = integer.T
            if ((x < 0) | (x >= width) | (y < 0) | (y >= height)).any():
                raise ValueError('Frozen query outside original source')
            if not physical[y, x].all() or not np.array_equal(reference, material[y, x].astype(float)):
                raise ValueError('Frozen query/reference differs from original material support')
            reserved[y*width+x] = True
            result[domain] = {'source': source, 'reference': reference,
                              'source_sha256': array_hash(source),
                              'reference_sha256': array_hash(reference),
                              'count': len(source)}
        candidates = np.flatnonzero(physical.ravel() & ~reserved)
        if len(candidates) < 64:
            raise ValueError('Fewer than64 material-support pixels remain for disjoint anchors')
        seed_bytes = hashlib.sha256(('20261023:'+str(key)).encode()).digest()[:8]
        seed = int.from_bytes(seed_bytes, 'little', signed=False)
        rng = np.random.Generator(np.random.PCG64(seed))
        chosen = rng.choice(candidates, size=64, replace=False)
        y, x = np.unravel_index(chosen, physical.shape)
        source = np.column_stack((x, y)).astype(float)
        reference = material[y, x].astype(float)
        assert not reserved[chosen].any()
        result['anchors'] = {'source': source, 'reference': reference,
                             'source_sha256': array_hash(source),
                             'reference_sha256': array_hash(reference),
                             'count': 64, 'seed': seed,
                             'selection': 'PCG64, SHA256(20261023:key) first8bytes little-endian'}
        result['contract'] = {
            'source_shape': [height, width],
            'material_support_pixels': int(physical.sum()),
            'disjoint_anchor_pool': True,
            'one_paper_anchor_offset_reused_both_domains': True,
            'measure': 'uniform source-image exported material-support pixels',
            'visibility': 'renderer-exported material support with recorded post-noise opaque occlusion removed, audited independently',
        }
        result['hashes'] = {name: {k: result[name][k] for k in ('source_sha256', 'reference_sha256')}
                            for name in ('anchors', 'grid', 'paper')}
        return result


def _prediction(item, expected_count):
    coords = np.asarray(item['coords'], dtype=float)
    eligible = np.asarray(item.get('claimed_eligible', item.get('eligible')), dtype=bool)
    if coords.shape != (expected_count, 2) or eligible.shape != (expected_count,):
        raise ValueError('Prediction needs coords(N,2) and claimed_eligible(N)')
    if 'claimed_eligible' in item and 'eligible' in item:
        if not np.array_equal(eligible, np.asarray(item['eligible'], dtype=bool)):
            raise ValueError('Conflicting eligibility fields')
    return coords, eligible


def _finite_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _error_statistics(errors):
    if not len(errors):
        return {k: None for k in ('rmse_mm', 'p50_mm', 'p95_mm', 'p99_mm', 'max_mm')}
    maximum = float(np.max(errors))
    rmse = maximum*np.sqrt(np.mean((errors/maximum)**2)) if maximum else 0.
    return {'rmse_mm': _finite_number(rmse),
            'p50_mm': _finite_number(np.quantile(errors, .5)),
            'p95_mm': _finite_number(np.quantile(errors, .95)),
            'p99_mm': _finite_number(np.quantile(errors, .99)),
            'max_mm': _finite_number(maximum)}


def score_coordinates(reference, predicted, claimed_eligible, threshold_mm=.5):
    """Fixed-denominator score. A claimed-eligible NaN is wrong, not abstention."""
    reference = np.asarray(reference, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    claimed = np.asarray(claimed_eligible, dtype=bool)
    if reference.ndim != 2 or reference.shape[1:] != (2,) or not len(reference):
        raise ValueError('Gold query domain must be nonempty N by2')
    if predicted.shape != reference.shape or claimed.shape != (len(reference),):
        raise ValueError('Prediction and eligibility do not match gold domain')
    if not np.isfinite(reference).all() or not np.isfinite(threshold_mm) or threshold_mm <= 0:
        raise ValueError('Gold and error threshold must be finite, threshold positive')
    finite = np.isfinite(predicted).all(1)
    returned_finite = claimed & finite
    residual = predicted[returned_finite]-reference[returned_finite]
    error = np.hypot(residual[:, 0], residual[:, 1])
    count = len(reference)
    correct = int(np.count_nonzero(error <= threshold_mm))
    coverage = int(claimed.sum())
    wrong = coverage-correct
    output = {'queries': count, 'correct_queries': correct, 'wrong_queries': wrong,
              'claimed_queries': coverage, 'finite_claimed_queries': int(returned_finite.sum()),
              'claimed_nonfinite_queries': int(np.count_nonzero(claimed & ~finite)),
              'unclaimed_queries': int(np.count_nonzero(~claimed)),
              'g': correct/count, 'w': wrong/count, 'c': coverage/count,
              'r': wrong/coverage if coverage else None,
              'error_statistic_population': 'finite claimed-eligible predictions only',
              **_error_statistics(error)}
    assert abs(output['g']+output['w']-output['c']) < 1e-12
    return output


def _registration(reference, predicted, claimed, matrix):
    transformed = predicted@matrix.T
    finite = np.isfinite(transformed).all(1)
    selected = claimed & finite
    count = int(selected.sum())
    reference_span = np.ptp(reference[selected], axis=0) if count else np.zeros(2)
    predicted_span = np.ptp(transformed[selected], axis=0) if count else np.zeros(2)
    record = {'eligible_finite_anchors': count,
              'claimed_nonfinite_anchors': int(np.count_nonzero(claimed & ~finite)),
              'reference_span_mm': reference_span.tolist(),
              'predicted_span_mm': predicted_span.tolist(),
              'matrix': matrix.astype(int).tolist(), 'offset_mm': None,
              'minimum_reference_span_mm': 1., 'minimum_anchors': 3,
              'reference_span_only_is_gate': True}
    if count < 3 or (reference_span < 1.).any():
        return {**record, 'status': 'registration_unavailable', 'anchor_median_error_mm': None}
    offset = np.median(reference[selected]-transformed[selected], axis=0)
    error = np.hypot(*(transformed[selected]+offset-reference[selected]).T)
    return {**record, 'status': 'registered_translation_only',
            'offset_mm': offset.tolist(), 'anchor_median_error_mm': float(np.median(error)),
            'anchor_mean_error_mm': float(np.mean(error))}


def _score_registered(predictions, queries, registration, matrix):
    result = {}
    for domain in DOMAINS:
        reference = np.asarray(queries[domain]['reference'], dtype=float)
        coords, claimed = _prediction(predictions[domain], len(reference))
        native_count = int(claimed.sum())
        native_finite_count = int(np.count_nonzero(claimed & np.isfinite(coords).all(1)))
        if registration['status'] == 'registration_unavailable':
            row = score_coordinates(reference, np.full_like(reference, np.nan), np.zeros(len(reference), bool))
            row['status'] = 'registration_unavailable'
            row['coverage_scope'] = 'operational registered-map availability, native map may exist'
        else:
            transformed = coords@matrix.T+np.asarray(registration['offset_mm'])
            row = score_coordinates(reference, transformed, claimed)
            row['status'] = 'scored'
            row['coverage_scope'] = 'native claimed eligibility after a permitted shared-origin offset'
        row['native_claimed_queries'] = native_count
        row['native_claimed_coverage'] = native_count/len(reference)
        row['native_finite_claimed_queries'] = native_finite_count
        row['native_finite_claimed_coverage'] = native_finite_count/len(reference)
        result[domain] = row
    return result


def _unrestricted_finite_field(predictions, queries, primary_registration):
    missing = [d for d in ('anchors', *DOMAINS)
               if 'native_footprint' not in predictions[d] and 'footprint' not in predictions[d]]
    if missing:
        return {'status': 'not_available_missing_native_footprint', 'missing_domains': missing,
                'scope': 'Never infer RGB footprint from a confidence mask'}
    fields, details = {}, {}
    for domain in ('anchors', *DOMAINS):
        coords, _ = _prediction(predictions[domain], len(queries[domain]['reference']))
        footprint = np.asarray(predictions[domain].get('native_footprint', predictions[domain].get('footprint')), dtype=bool)
        if footprint.shape != (len(coords),):
            raise ValueError('Native footprint must match query count')
        finite = np.isfinite(coords).all(1)
        fields[domain] = {'coords': coords, 'claimed_eligible': footprint & finite}
        details[domain] = {'native_RGB_footprint_queries': int(footprint.sum()),
                           'native_RGB_footprint_coverage': float(footprint.mean()),
                           'nonfinite_inside_native_RGB_footprint': int(np.count_nonzero(footprint & ~finite))}
    if primary_registration['status'] != 'registration_unavailable':
        registration = {**primary_registration, 'offset_source': 'same_primary_registration'}
    else:
        coords, claimed = _prediction(fields['anchors'], len(queries['anchors']['reference']))
        registration = {**_registration(np.asarray(queries['anchors']['reference']), coords, claimed, D4[0]),
                        'offset_source': 'unrestricted_finite_footprint_anchors_after_primary_registration_unavailable'}
    domains = _score_registered(fields, queries, registration, D4[0])
    for domain in DOMAINS:
        domains[domain].update(details[domain])
    return {'status': 'available', 'registration': registration, 'domains': domains,
            'scope': 'Finite native physical field inside actual RGB footprint, confidence gate removed. Not RGB-fidelity certification.',
            'nonfinite_footprint_queries_reported_separately': True}


def score_predictions(prediction_domains, queries):
    """Score anchors/grid/paper predictions. Inference must already be frozen.

    Prediction domains use coords and claimed_eligible (eligible alias allowed).
    One anchor-derived offset is used unchanged in both score domains. D4 is
    selected using anchor residuals only and appears solely in secondary output.
    """
    reference = np.asarray(queries['anchors']['reference'], dtype=float)
    if reference.ndim != 2 or reference.shape[1:] != (2,) or not np.isfinite(reference).all():
        raise ValueError('Anchor gold must be finite N by2')
    predicted, claimed = _prediction(prediction_domains['anchors'], len(reference))
    primary_registration = _registration(reference, predicted, claimed, D4[0])
    registrations = [_registration(reference, predicted, claimed, matrix) for matrix in D4]
    candidates = [(r['anchor_median_error_mm'], r.get('anchor_mean_error_mm', np.inf), i)
                  for i, r in enumerate(registrations) if r['status'] != 'registration_unavailable']
    chosen = min(candidates)[2] if candidates else 0
    diagnostic_registration = {**registrations[chosen], 'D4_index': chosen,
                               'selection_scope': 'anchor residuals only, not scoring queries'}
    raw = {d: score_coordinates(queries[d]['reference'], *_prediction(prediction_domains[d], len(queries[d]['reference'])))
           for d in DOMAINS}
    return {
        'primary': {'gauge': 'translation_only_native_orientation_and_scale',
                    'registration': primary_registration,
                    'domains': _score_registered(prediction_domains, queries, primary_registration, D4[0])},
        'secondary_D4': {'gauge': 'D4_plus_translation_diagnostic_only_not_primary_ranking',
                         'registration': diagnostic_registration,
                         'domains': _score_registered(prediction_domains, queries, diagnostic_registration, D4[chosen])},
        'unrestricted_finite_field': _unrestricted_finite_field(prediction_domains, queries, primary_registration),
        'raw_native_origin': {'scope': 'unregistered native origin, meaningful as absolute origin only when externally defined',
                              'domains': raw},
    }


def _metric_dict(gwc):
    g, w, c = [float(v) for v in gwc]
    return {'g': g, 'w': w, 'c': c, 'r': w/c if c else None}


def patient_backend_tensor(rows, patients=None, backends=None):
    """Rows have group_id, backend and g/w/c fields, or a metrics dictionary."""
    patients = sorted({r['group_id'] for r in rows}) if patients is None else list(patients)
    backends = sorted({r['backend'] for r in rows}) if backends is None else list(backends)
    if not patients or not backends:
        raise ValueError('At least one patient and backend are required')
    tensor = np.empty((len(patients), len(backends), 3), dtype=float)
    for i, patient in enumerate(patients):
        for j, backend in enumerate(backends):
            selected = [r.get('metrics', r) for r in rows
                        if r['group_id'] == patient and r['backend'] == backend]
            if not selected:
                raise ValueError('Every scheduled patient must contribute to each backend')
            values = np.asarray([[r[k] for k in ('g', 'w', 'c')] for r in selected], dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError('G/W/C must be finite fractions')
            if not np.allclose(values[:, 0]+values[:, 1], values[:, 2], rtol=0, atol=1e-12):
                raise ValueError('C must equal G+W')
            tensor[i, j] = np.mean(values, axis=0)
    return tensor, patients, backends


def aggregate_patient_backend(rows):
    tensor, patients, backends = patient_backend_tensor(rows)
    return {'overall': _metric_dict(tensor.mean(axis=(0, 1))),
            'by_backend': {b: _metric_dict(tensor[:, j].mean(0)) for j, b in enumerate(backends)},
            'by_patient': {p: _metric_dict(tensor[i].mean(0)) for i, p in enumerate(patients)},
            'patients': patients, 'backends': backends, 'independent_patients': len(patients),
            'cases': len(rows), 'aggregation': 'equal patient/backend means, ratio of aggregate W/C'}


def paired_patient_bootstrap(rows_by_method, baseline, repetitions=20000, seed=1973):
    """Conditional descriptive paired intervals, not confirmatory superiority."""
    if baseline not in rows_by_method or repetitions <= 0:
        raise ValueError('Baseline and positive bootstrap repetitions required')
    base, patients, backends = patient_backend_tensor(rows_by_method[baseline])
    methods = list(rows_by_method)
    tensors = []
    for method in methods:
        rows = rows_by_method[method]
        if {r['group_id'] for r in rows} != set(patients) or {r['backend'] for r in rows} != set(backends):
            raise ValueError('Paired methods must have identical patient and backend sets')
        base_rows = rows_by_method[baseline]
        if Counter((r['group_id'], r['backend']) for r in rows) != Counter((r['group_id'], r['backend']) for r in base_rows):
            raise ValueError('Paired methods must retain identical patient/backend case counts')
        if all('key' in r for r in rows+base_rows):
            if Counter((r['key'], r['group_id'], r['backend']) for r in rows) != Counter((r['key'], r['group_id'], r['backend']) for r in base_rows):
                raise ValueError('Paired methods must retain the same scheduled case keys')
        tensor, _, _ = patient_backend_tensor(rows, patients, backends)
        tensors.append(tensor)
    tensor = np.stack(tensors, axis=2)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(patients), size=(repetitions, len(patients)))
    draws = tensor[indices].mean(axis=(1, 2))
    values = np.concatenate((draws, np.divide(draws[:, :, 1:2], draws[:, :, 2:3],
                            out=np.full_like(draws[:, :, 1:2], np.nan), where=draws[:, :, 2:3] > 0)), axis=2)
    baseline_index = methods.index(baseline)
    output = {}
    for i, method in enumerate(methods):
        intervals = {}
        for k, name in enumerate(('g', 'w', 'c', 'r')):
            delta = values[:, i, k]-values[:, baseline_index, k]
            finite = delta[np.isfinite(delta)]
            intervals[name] = {'lower': float(np.quantile(finite, .025)) if len(finite) else None,
                               'upper': float(np.quantile(finite, .975)) if len(finite) else None,
                               'defined_draws': len(finite)}
        point = tensor[:, :, i].mean(axis=(0, 1))
        bpoint = base.mean(axis=(0, 1))
        pm, bm = _metric_dict(point), _metric_dict(bpoint)
        output[method] = {'metrics': pm,
                          'delta_vs_baseline': {k: pm[k]-bm[k] if pm[k] is not None and bm[k] is not None else None
                                                for k in ('g', 'w', 'c', 'r')},
                          'delta_95_descriptive_interval': intervals}
    return {'methods': output, 'baseline': baseline, 'repetitions': repetitions, 'seed': seed,
            'independent_patients': len(patients), 'patients': patients, 'backends': backends,
            'unit': 'paired patient clusters with all backends preserved',
            'scope': 'descriptive conditional intervals, no multiplicity adjustment or external-population guarantee'}
