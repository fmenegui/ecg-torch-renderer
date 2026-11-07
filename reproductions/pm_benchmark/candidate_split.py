"""Freeze patient-exclusive source signals for a new trace-reader candidate.

This prepares source identities, not acquired images or clinical validation.
Every recording and every future rendering inherits its patient's partition.
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from reproductions.common import sha256, write_json

from .contracts import patient_key, validate_training_patients

SEED = 'ecg-lead-reader-source-patients-v1/'


def split_rows(rows, exclusions):
    forbidden = {patient_key(value) for value in exclusions['patient_ids']}
    prepared, excluded, seen = [], [], set()
    for row in rows:
        identity, patient = int(row['ecg_id']), patient_key(row['patient_id'])
        if identity in seen or identity < 1 or not re.fullmatch(r'records500/\d{5}/\d{5}_hr', row['filename_hr']):
            raise ValueError('Unique verified PTB-XL record identities required')
        if int(Path(row['filename_hr']).stem.split('_')[0]) != identity:
            raise ValueError('Signal filename belongs to another ECG recording')
        seen.add(identity)
        if patient in forbidden:
            excluded.append(identity)
            continue
        bucket = int(hashlib.sha256((SEED + patient).encode()).hexdigest()[:8], 16) % 10000
        partition = 'train' if bucket < 8000 else 'validation' if bucket < 9000 else 'source_holdout'
        prepared.append({'ecg_id': identity, 'patient_id': patient,
            'filename_hr': row['filename_hr'], 'split': partition,
            'fs': 500, 'future_images_inherit_patient_split': True})
    prepared.sort(key=lambda row: row['ecg_id'])
    validate_training_patients(prepared, exclusions)
    patient_splits = {}
    for row in prepared:
        if patient_splits.setdefault(row['patient_id'], row['split']) != row['split']:
            raise AssertionError('Patient leaked across source partitions')
    return prepared, sorted(excluded)


def prepare(metadata, exclusions_path, out):
    metadata, exclusions_path, out = Path(metadata), Path(exclusions_path), Path(out)
    if out.exists():
        raise FileExistsError('A new frozen source-manifest path is required')
    exclusions = json.loads(exclusions_path.read_text())
    with metadata.open() as stream:
        rows, excluded = split_rows(list(csv.DictReader(stream)), exclusions)
    summary = {partition: {'records': sum(row['split'] == partition for row in rows),
        'patients': len({row['patient_id'] for row in rows if row['split'] == partition})}
        for partition in ['train', 'validation', 'source_holdout']}
    document = {'schema': 'patient-exclusive-trace-reader-source-plan-v1',
        'status': 'source_identities_prepared_raw_signal_acquisition_pending',
        'seed': SEED, 'metadata_sha256': sha256(metadata), 'exclusions_sha256': sha256(exclusions_path),
        'excluded_benchmark_patients': len(exclusions['patient_ids']), 'excluded_record_ids': excluded,
        'partition_summary': summary, 'records': rows,
        'new_training_executed': False, 'clinical_image_test_set_acquired': False,
        'source_holdout_is_independent_clinical_validation': False,
        'warning': 'Source waveform holdout and synthetic rendering do not replace an independent paired clinical image test.'}
    write_json(out, document)
    return {key: value for key, value in document.items() if key not in ['records', 'excluded_record_ids']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['metadata', 'exclusions', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.metadata, args.exclusions, args.out)))


if __name__ == '__main__':
    main()
