"""Hash-verified post-seal scoring of the frozen Runpod clock experiment."""

import argparse
import json
from pathlib import Path

import numpy as np

from ecg_gpu.digitization.external_evaluation import score_external_signals
from ecg_gpu.digitization.selective import SignalProposal
from ecg_gpu.digitization.support_audit import aggregate_patient_audits, audit_support
from reproductions.common import sha256, write_json

from .contracts import join_pages
from .worker_clock import ARMS, ROOT, validate_inputs


def checked_array(root, record, fs=500):
    relative = Path(record['prediction'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Safe sealed prediction path required')
    path = root / relative
    if sha256(path) != record['prediction_sha256']:
        raise ValueError('Sealed prediction bytes changed')
    with np.load(path, allow_pickle=False) as archive:
        values, rate = archive['values_mv'].copy(), float(archive['fs'])
    if values.shape != (12, 5000) or rate != fs or np.isinf(values).any():
        raise ValueError('Canonical physical axes required')
    return values


def score(catalog_root, saved, *, catalog_sha256, protocol_path=ROOT / 'clock-inputs.json'):
    catalog_root, saved, protocol_path = Path(catalog_root), Path(saved), Path(protocol_path)
    if sha256(catalog_root / 'catalog.json') != catalog_sha256:
        raise ValueError('Previously verified reference catalog changed')
    protocol = json.loads(protocol_path.read_text())
    cases = validate_inputs(protocol)
    catalog = json.loads((catalog_root / 'catalog.json').read_text())
    if (catalog['metadata_sha256'] != protocol['metadata_sha256']
            or catalog['archive_index_sha256'] != protocol['zip_index_sha256']):
        raise ValueError('Image schedule and reference catalog release disagree')
    receipt = json.loads((saved / 'receipt.json').read_text())
    seal_path = saved / 'sealed-predictions.json'
    if (receipt['status'] != 'completed_inference' or receipt['reference_inputs_received']
            or receipt['reference_scoring_performed'] or receipt['new_training']
            or receipt['protocol_sha256'] != sha256(protocol_path)
            or receipt['prediction_seal_sha256'] != sha256(seal_path)):
        raise ValueError('Reference-free completed inference seal required before scoring')
    seal = json.loads(seal_path.read_text())
    if seal['reference_used'] or seal['native_observer_parity_bit_exact'] is not True:
        raise ValueError('Unchanged native observer parity must pass')
    image_manifest = json.loads((saved / 'sealed-image-only-manifest.json').read_text())
    pages = {p['image_id']: p for case in cases for p in case['pages']}
    images = {p['image_id']: p for p in image_manifest}
    if len(images) != len(image_manifest) or set(images) != set(pages):
        raise ValueError('All scheduled pages must remain in the denominator')
    for identity, page in pages.items():
        actual = images[identity]
        if actual['image'] != page['image'] or (page['image_sha256'] and page['image_sha256'] != actual['image_sha256']):
            raise ValueError('Original image identity changed')
    expected = {(name, arm) for name in pages for arm in ARMS}
    records = {(r['image_id'], r['arm']): r for r in seal['records']}
    if len(records) != len(seal['records']) or set(records) != expected:
        raise ValueError('Every scheduled page and arm must be sealed exactly once')
    page_values = {}
    for (identity, arm), record in records.items():
        if record['source_id'] != images[identity]['image_sha256']:
            raise ValueError('Prediction belongs to another image')
        page_values[identity, arm] = checked_array(saved, record)
    scheduled = {case['exam_id'] for case in cases}
    exams = {(r['exam_id'], r['arm']): r for r in seal['exams']}
    if len(exams) != len(seal['exams']) or set(exams) != {(name, arm) for name in scheduled for arm in ARMS}:
        raise ValueError('Failed exams cannot disappear from the schedule')
    references = {c['exam_id']: c for c in catalog['cases']}
    if len(references) != len(catalog['cases']) or not scheduled <= references.keys():
        raise ValueError('Unique corresponding reference exam required')
    rows, details = [], []
    for name in sorted(scheduled):
        case = references[name]
        reference = catalog_root / case['reference']
        if sha256(reference) != case['reference_sha256']:
            raise ValueError('Reference bytes changed')
        with np.load(reference, allow_pickle=False) as archive:
            truth = archive['values_mv'].copy()
            if float(archive['fs']) != 500:
                raise ValueError('Reference physical time scale changed')
        predictions = {arm: checked_array(saved, exams[name, arm]) for arm in ARMS}
        for arm in ARMS:
            assembled = join_pages([SignalProposal(page_values[p['image_id'], arm], 500,
                                                  images[p['image_id']]['image_sha256'], None)
                                    for p in case['pages']], exam_source_id=name, fs=500, samples=5000)
            if not np.array_equal(assembled.values_mv, predictions[arm], equal_nan=True):
                raise ValueError('Sealed exam assembly disagrees with its page observations')
        row = audit_support(truth, *(predictions[arm] for arm in ARMS))
        row.update(image_id=name, exam_id=name, patient_id=case['patient_id'], condition=case['condition'],
                   page_count=len(case['pages']), source_format=case['source_format_scoring_only'])
        rows.append(row)
        details.append({'exam_id': name, 'scores': {arm: score_external_signals(truth, predictions[arm], 500, max_shift_s=0)
                                                   for arm in ARMS}})
    return {'schema': 'physical-clock-post-seal-evaluation-v1', 'catalog_sha256': catalog_sha256,
        'inference_seal_sha256': sha256(seal_path), 'protocol_sha256': sha256(protocol_path),
        'scheduled_exams': len(scheduled), 'scheduled_pages': len(pages),
        'scheduled_patients': len({r['patient_id'] for r in rows}), 'reference_used_for_inference': False,
        'pages_with_any_observation': {arm: sum(np.isfinite(page_values[name, arm]).any().item() for name in pages) for arm in ARMS},
        'exams_with_any_observation': {arm: sum(np.isfinite(checked_array(saved, exams[name, arm])).any().item() for name in scheduled) for arm in ARMS},
        'raw_physical_axes_only': True, 'amplitude_or_shift_fitted': False,
        'secondary_zero_shift_dc_centered_diagnostics_present': True,
        'author_headline_reproduced': False, 'independent_accuracy_validation': False,
        'patient_macro_support_audit': aggregate_patient_audits(rows),
        'by_condition': {key: aggregate_patient_audits([r for r in rows if r['condition'] == key])
                         for key in sorted({r['condition'] for r in rows})},
        'by_page_count': {str(key): aggregate_patient_audits([r for r in rows if r['page_count'] == key])
                          for key in sorted({r['page_count'] for r in rows})},
        'exam_support_audits': rows, 'exam_scores': details,
        'warning': 'An aligned image frame origin is not validated print zero. Lower zero-missing error is insufficient for superiority.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--saved', type=Path, required=True)
    parser.add_argument('--catalog-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = score(args.catalog, args.saved, catalog_sha256=args.catalog_sha256)
    write_json(args.out, report)
    print(json.dumps(report['patient_macro_support_audit']))


if __name__ == '__main__':
    main()
