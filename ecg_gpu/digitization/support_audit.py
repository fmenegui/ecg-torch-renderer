"""Decompose paired waveform error without rewarding unreported support loss.

This module is for evaluation only. It never chooses or modifies predictions.
The historical zero-missing score is preserved and its changes are attributed
to common support, dropped samples, gained samples and jointly missing samples.
"""

import numpy as np


def audit_support(reference, direct, candidate):
    y, a, b = [np.asarray(v, dtype=np.float64) for v in (reference, direct, candidate)]
    if y.ndim != 2 or y.shape[0] != 12 or a.shape != y.shape or b.shape != y.shape or not y.shape[1]:
        raise ValueError('Matching nonempty [12, time] arrays required')
    if any(np.isinf(v).any() for v in (y, a, b)) or not np.isfinite(y).any():
        raise ValueError('Printed reference support required, infinity forbidden')
    printed = np.isfinite(y)
    sa, sb = printed & np.isfinite(a), printed & np.isfinite(b)
    masks = {'common': sa & sb, 'dropped': sa & ~sb,
             'gained': sb & ~sa, 'both_missing': printed & ~sa & ~sb}
    n = int(printed.sum())
    power = float(np.sum(y[printed] ** 2))
    # These arrays are scoring operands only, never returned waveforms.
    ap, bp = np.where(sa, a, 0.), np.where(sb, b, 0.)
    ea, eb = (ap[printed] - y[printed]) ** 2, (bp[printed] - y[printed]) ** 2
    sse_a, sse_b = float(ea.sum()), float(eb.sum())
    nr_a, nr_b = (float(np.sqrt(s / power)) for s in (sse_a, sse_b)) if power > 0 else (None, None)
    denominator = power * (nr_a + nr_b) if power > 0 else None
    partitions = {}
    for name, mask in masks.items():
        delta = float(np.sum((bp[mask] - y[mask]) ** 2 - (ap[mask] - y[mask]) ** 2))
        partitions[name] = {'samples': int(mask.sum()), 'fraction_of_printed': float(mask.sum() / n),
                            'delta_squared_error_mv2': delta, 'delta_mse_mv2': delta / n,
                            'delta_nrmse_contribution': delta / denominator if denominator else (0. if power > 0 else None)}
    common = masks['common']
    count = int(common.sum())
    comparable = None
    if count:
        comparable = {'samples': count, 'direct_rmse_mv': float(np.sqrt(np.mean((a[common] - y[common]) ** 2))),
                      'candidate_rmse_mv': float(np.sqrt(np.mean((b[common] - y[common]) ** 2)))}
    return {'schema': 'paired-support-error-audit-v1', 'printed_samples': n,
            'direct_coverage': float(sa.sum() / n), 'candidate_coverage': float(sb.sum() / n),
            'direct_nrmse_zero_missing': nr_a, 'candidate_nrmse_zero_missing': nr_b,
            'delta_nrmse': nr_b - nr_a if nr_a is not None else None,
            'direct_all_printed_rmse_mv': float(np.sqrt(sse_a / n)),
            'candidate_all_printed_rmse_mv': float(np.sqrt(sse_b / n)),
            'partitions': partitions, 'common_support': comparable,
            'candidate_preserves_all_direct_samples': not bool(masks['dropped'].any()),
            'single_scalar_superiority_claim_supported': False,
            'reference_used_for_evaluation_only': True,
            'warning': 'Zero-missing error can improve by dropping erroneous predictions. Common-support error is conditional on retained samples, not full-image success.'}


def aggregate_patient_audits(rows):
    patients = sorted({r['patient_id'] for r in rows})

    def macro(values):
        by_patient = []
        for patient in patients:
            finite = [v for r, v in zip(rows, values) if r['patient_id'] == patient and v is not None]
            if finite:
                by_patient.append(float(np.mean(finite)))
        return {'value': float(np.mean(by_patient)) if by_patient else None,
                'images_with_defined_value': sum(v is not None for v in values),
                'scheduled_images': len(rows), 'patients_with_defined_value': len(by_patient),
                'scheduled_patients': len(patients)}

    result = {k: macro([r[k] for r in rows]) for k in ['direct_coverage', 'candidate_coverage', 'delta_nrmse']}
    result['partitions'] = {name: {key: macro([r['partitions'][name][key] for r in rows])
        for key in ['fraction_of_printed', 'delta_nrmse_contribution']}
        for name in ['common', 'dropped', 'gained', 'both_missing']}
    result['common_support'] = {key: macro([r['common_support'][key] if r['common_support'] else None for r in rows])
        for key in ['direct_rmse_mv', 'candidate_rmse_mv']}
    result['candidate_lost_direct_samples_in_images'] = sum(not r['candidate_preserves_all_direct_samples'] for r in rows)
    delta = result['delta_nrmse']['value']
    dropped = result['partitions']['dropped']['delta_nrmse_contribution']['value']
    result['dropped_partition_fraction_of_total_nrmse_reduction'] = dropped / delta if delta is not None and delta < 0 else None
    return result

