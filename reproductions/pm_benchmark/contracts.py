"""Gold-free page acquisition, conservative exam assembly and training guards."""

import math
from pathlib import Path

import numpy as np

from ecg_gpu.digitization.selective import SignalProposal
from examples.prepare_pm_ecg_external import fetch_member
from reproductions.common import sha256


def patient_key(value):
    text = str(value).removeprefix('ptbxl-patient-')
    number = float(text)
    if not math.isfinite(number) or not number.is_integer() or number < 1:
        raise ValueError('A verified PTB-XL patient identity is required')
    return 'ptbxl-patient-' + str(float(number))


def validate_training_patients(rows, exclusions):
    forbidden = {patient_key(v) for v in exclusions['patient_ids']}
    for row in rows:
        if patient_key(row['patient_id']) in forbidden:
            raise ValueError('Benchmark patient entered candidate training, including another recording or rendering')
    return {'training_rows_checked': len(rows), 'excluded_benchmark_patients': len(forbidden), 'passed': True}


def acquire_image_only(catalog, exam_ids, root):
    """Acquire selected bytes before sealing an inference manifest.

    This function is preprocessing. Its output contains no signal, patient,
    source layout, printed window or reference path.
    """
    root = Path(root)
    selected = [c for c in catalog['cases'] if c['exam_id'] in set(exam_ids)]
    if {c['exam_id'] for c in selected} != set(exam_ids) or len(set(exam_ids)) != len(exam_ids):
        raise ValueError('Exact unique scheduled exam identities required')
    result = []
    for case in selected:
        pages = []
        for page in case['pages']:
            path = fetch_member(root, page['archive_member'])
            digest = sha256(path)
            if page['image_sha256'] and digest != page['image_sha256']:
                raise ValueError('Previously sealed image bytes changed')
            pages.append({'image_id': page['image_id'], 'image': page['image'], 'image_sha256': digest})
        result.append({'exam_id': case['exam_id'], 'pages': pages})
    return result


def join_pages(pages, *, exam_source_id, fs, samples):
    """Union named observations, abstaining where pages disagree.

    Page identity/lead assignment must come from the digitizer. Scoring layout
    and reference waveforms never enter this function. Conflicting samples are
    not averaged, and later pages cannot silently overwrite earlier pages.
    """
    if not pages:
        raise ValueError('Scheduled page results required')
    values = np.full((12, samples), np.nan)
    conflicts = np.zeros((12, samples), bool)
    origins = set()
    for page in pages:
        if page.fs != fs or page.values_mv.shape != values.shape or page.source_id in origins:
            raise ValueError('Distinct page identities and matching physical axes required')
        origins.add(page.source_id)
        a = np.asarray(page.values_mv)
        shared = np.isfinite(values) & np.isfinite(a)
        conflicts |= shared & ~np.isclose(values, a, rtol=0, atol=1e-10)
        add = ~np.isfinite(values) & np.isfinite(a)
        values[add] = a[add]
    values[conflicts] = np.nan
    return SignalProposal(values, fs, exam_source_id, None, metadata={
        'backend': 'conservative-named-page-union', 'scheduled_pages': len(pages),
        'conflicting_samples': int(conflicts.sum()), 'reference_used': False,
        'missing_values_filled': False, 'page_values_averaged': False})
