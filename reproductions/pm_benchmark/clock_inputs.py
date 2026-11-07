"""Select a small fixed integration cohort, covering all page-count profiles."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from reproductions.common import sha256, write_json

from .catalog import group_pages, recording_id

PROFILES = [('3x4', 4), ('3x4', 2), ('6x2', 2), ('12x1', 1), ('6x2', 1), ('3x4+1R', 1)]
CONDITIONS = ['photos_scans', 'photos_bents']


def prepare(root, ptb_metadata, out):
    root = Path(root)
    with (root / 'final_data/metadata.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    with Path(ptb_metadata).open() as stream:
        ptb = {int(r['ecg_id']): r for r in csv.DictReader(stream)}
    groups = group_pages(rows)
    index = {e['name']: e for e in json.loads((root / 'zip-index.json').read_text())}
    chosen, patients, exams = [], set(), []
    for layout, count in PROFILES:
        candidates = [pages for (_, condition), pages in groups.items() if condition == 'photos_scans'
                      and pages[0]['ECG format'] == layout and len(pages) == count]
        candidates.sort(key=lambda pages: hashlib.sha256(('physical-clock-integration-v1/' + pages[0]['Image ID']).encode()).hexdigest())
        for pages in candidates:
            row = pages[0]
            recording = recording_id(row['ECG ID'])
            patient = ptb[recording]['patient_id']
            if patient not in patients:
                patients.add(patient)
                chosen.append({'image_id': row['Image ID'], 'recording_id': recording,
                               'patient_id': patient, 'layout_scoring_only': layout, 'page_count': count})
                break
        else:
            raise ValueError('No patient-disjoint representative for profile')
    for item in chosen:
        for condition in CONDITIONS:
            pages = groups[item['image_id'], condition]
            entries = []
            for row in sorted(pages, key=lambda r: int(r['Image page'])):
                name = 'final_data/visual_data/' + row['Image relative path']
                path = root / name
                entries.append({'image_id': f'pm-{item["image_id"]}-{condition}-page-{row["Image page"]}',
                    'image': name, 'image_sha256': sha256(path) if path.exists() else None,
                    'archive_member': index[name]})
            exams.append({'exam_id': f'pm-{item["image_id"]}-{condition}', 'pages': entries})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = {'schema': 'physical-clock-image-only-inputs-v1', 'exams': exams,
        'metadata_sha256': sha256(root / 'final_data/metadata.csv'),
        'zip_index_sha256': sha256(root / 'zip-index.json'),
        'public_axes': {'fs': 500, 'samples': 5000, 'paper_speed_mm_s': 25., 'gain_mm_mv': 10.,
                        'origin_x_px': 0., 'origin_basis': 'aligned image frame zero, independently unverified as print zero'},
        'independent_accuracy_validation': False, 'new_training': False}
    write_json(out / 'clock-inputs.json', inputs)
    write_json(out / 'clock-selection-audit.json', {'schema': 'physical-clock-cohort-selection-v1',
        'selection': 'SHA256 seed, rare page-count profiles first, no prediction or reference scoring',
        'base_records': chosen, 'patients': len(patients), 'conditions': CONDITIONS,
        'exam_entries': len(exams), 'page_files': sum(len(e['pages']) for e in exams),
        'input_sha256': sha256(out / 'clock-inputs.json')})
    return inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'ptb-metadata', 'out']:
        p.add_argument('--' + name, required=True, type=Path)
    args = p.parse_args()
    result = prepare(args.root, args.ptb_metadata, args.out)
    print({'exams': len(result['exams']), 'pages': sum(len(e['pages']) for e in result['exams'])})


if __name__ == '__main__':
    main()
