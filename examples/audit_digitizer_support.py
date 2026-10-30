"""Hash-verified attribution of score changes on a saved, paired image cohort."""

import argparse
import json
from pathlib import Path

import numpy as np

from ecg_gpu.digitization.support_audit import aggregate_patient_audits, audit_support
from reproductions.common import sha256, write_json


def run(inputs, saved, *, direct_arm='npj_native', candidate_arm='grid_adaptive'):
    inputs, saved = Path(inputs), Path(saved)
    manifest_path, report_path = inputs / 'manifest.json', saved / 'report.json'
    manifest, report = json.loads(manifest_path.read_text()), json.loads(report_path.read_text())
    cases = manifest['cases']
    if len({c['image_id'] for c in cases}) != len(cases):
        raise ValueError('Unique scheduled image identities required')
    records = {(r['image_id'], r['arm']): r for r in report['cases']}
    if len(records) != len(report['cases']):
        raise ValueError('Duplicate saved prediction identity')
    rows = []
    for c in cases:
        reference = inputs / c['reference']
        if sha256(reference) != c['reference_sha256']:
            raise ValueError('Reference hash changed')
        with np.load(reference, allow_pickle=False) as z:
            truth = z['values_mv'].copy()
        arrays = []
        rates = []
        for arm in [direct_arm, candidate_arm]:
            record = records[c['image_id'], arm]
            path = saved / record['prediction']
            if sha256(path) != record['prediction_sha256']:
                raise ValueError('Saved prediction hash changed')
            with np.load(path, allow_pickle=False) as z:
                arrays.append(z['values_mv'].copy())
                rates.append(float(z['fs']))
        if rates != [manifest['reference_fs']] * 2:
            raise ValueError('Explicit common time scale required')
        row = audit_support(truth, *arrays)
        row.update(image_id=c['image_id'], patient_id=c['patient_id'], condition=c['condition'])
        rows.append(row)
    return {'schema': 'patient-macro-support-audit-v1', 'manifest_sha256': sha256(manifest_path),
            'saved_report_sha256': sha256(report_path), 'direct_arm': direct_arm, 'candidate_arm': candidate_arm,
            'evaluation_only': True, 'original_predictions_modified': False, 'new_independent_validation': False,
            'patient_macro': aggregate_patient_audits(rows),
            'per_patient': {patient: aggregate_patient_audits([r for r in rows if r['patient_id'] == patient])
                            for patient in sorted({r['patient_id'] for r in rows})},
            'by_condition': {name: aggregate_patient_audits([r for r in rows if r['condition'] == name])
                             for name in sorted({r['condition'] for r in rows})},
            'per_image': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = run(args.inputs, args.run)
    write_json(args.out, result)
    print(json.dumps(result['patient_macro'], indent=2))


if __name__ == '__main__':
    main()
