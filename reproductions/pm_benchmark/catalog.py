"""Build an exam-level PM-ECG-ID catalog without loading predictions.

The 6,600 page files represent 6,000 exam/condition entries, 100 original
recordings and 97 patients. Signal-noise variants have their own references.
Reference time windows are verified against original PTB-XL, not guessed from
the number of columns on one page of a multipage recording.
"""

import argparse
import csv
import json
import re
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np

from ecg_gpu.digitization.data import read_wfdb16
from ecg_gpu.digitization.signals import LEADS
from reproductions.common import sha256, write_json

PTB_URL = 'https://physionet.org/files/ptb-xl/1.0.3/'
PTB_MIRROR = 'https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/'


def recording_id(name):
    match = re.fullmatch(r'(?:[a-z0-9_]+_)?LPAE_(\d{5})_hr', name)
    if not match:
        raise ValueError('Unknown PM waveform identity')
    return int(match[1])


def group_pages(rows):
    groups, paths = {}, set()
    for row in rows:
        path = PurePosixPath(row['Image relative path'])
        if path.is_absolute() or '..' in path.parts or len(path.parts) != 2 or str(path) in paths:
            raise ValueError('Unique safe condition/page paths required')
        paths.add(str(path))
        key = (row['Image ID'], path.parts[0])
        groups.setdefault(key, []).append(row)
    for pages in groups.values():
        for key in ['ECG ID', 'ECG number of pages', 'ECG number of columns per page',
                    'ECG number of rows per page', 'ECG number of rhythm leads', 'ECG format']:
            if len({r[key] for r in pages}) != 1:
                raise ValueError('Pages disagree about exam identity or format')
        count = int(pages[0]['ECG number of pages'])
        if count not in (1, 2, 4) or sorted(int(r['Image page']) for r in pages) != list(range(count)):
            raise ValueError('Missing or duplicated page')
    return groups


def infer_printed_windows(short, full, *, tolerance_mv=1e-6):
    short, full = np.asarray(short), np.asarray(full)
    if short.ndim != 2 or short.shape[1] != 12 or full.shape != (12, 5000) or len(short) not in (1250, 2500, 5000):
        raise ValueError('Expected canonical PM cutouts and ten-second PTB-XL source')
    if not np.isfinite(short).all() or not np.isfinite(full).all():
        raise ValueError('Finite source arrays required')
    starts, checks = [], []
    n = len(short)
    for lead in range(12):
        matches = []
        for start in range(0, 5000 - n + 1, n):
            difference = short[:, lead] - full[lead, start:start + n]
            offset = float(np.mean(difference))
            residual = float(np.max(np.abs(difference - offset)))
            if residual <= tolerance_mv:
                matches.append((start, offset, residual))
        if len(matches) != 1:
            raise ValueError('Ambiguous or inconsistent original time window for ' + LEADS[lead])
        start, offset, residual = matches[0]
        starts.append(start)
        checks.append({'lead': LEADS[lead], 'start_sample': start, 'samples': n,
                       'pm_minus_ptb_dc_mv': offset, 'maximum_residual_after_constant_dc_mv': residual})
    return starts, checks


def canonical_reference(short, rhythms, starts):
    short, rhythms = np.asarray(short), np.asarray(rhythms)
    if short.ndim != 2 or short.shape[1] != 12 or rhythms.ndim != 2 or rhythms.shape[0] != 5000 or len(starts) != 12:
        raise ValueError('Explicit cutouts, rhythm channels and source time windows required')
    if not np.isfinite(short).all() or not np.isfinite(rhythms).all():
        raise ValueError('Nonfinite PM reference')
    result, rhythm_leads = np.full((12, 5000), np.nan), []
    for lead, start in enumerate(starts):
        if type(start) is not int or not 0 <= start <= 5000 - len(short):
            raise ValueError('Invalid printed start')
        result[lead, start:start + len(short)] = short[:, lead]
    for channel in range(rhythms.shape[1]):
        matches = [lead for lead, start in enumerate(starts)
                   if np.array_equal(rhythms[start:start + len(short), channel], short[:, lead])]
        if len(matches) != 1 or matches[0] in rhythm_leads:
            raise ValueError('Ambiguous or duplicated rhythm lead')
        lead = matches[0]
        result[lead] = rhythms[:, channel]
        rhythm_leads.append(lead)
    return result, [LEADS[i] for i in rhythm_leads]


def fetch_originals(rows, ptb, out):
    out.mkdir(parents=True, exist_ok=True)

    def fetch(identity):
        stem = ptb[identity]['filename_hr']
        if not re.fullmatch(r'records500/\d{5}/\d{5}_hr', stem):
            raise ValueError('Unsafe PTB-XL source path')
        for suffix in ['.hea', '.dat']:
            path = out / (Path(stem).name + suffix)
            if not path.exists():
                for base in [PTB_MIRROR, PTB_URL]:
                    try:
                        with urllib.request.urlopen(base + stem + suffix, timeout=45) as response:
                            payload = response.read(200000)
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if base == PTB_URL:
                            raise
                if not payload or len(payload) == 200000:
                    raise ValueError('Unexpected source download size')
                temporary = path.with_suffix(path.suffix + '.part')
                temporary.write_bytes(payload)
                temporary.replace(path)
        # Byte count, header calibration, initial values and checksums checked.
        read_wfdb16(out / (Path(stem).name + '.hea'))
        return identity

    identities = sorted({recording_id(r['ECG ID']) for r in rows})
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(fetch, identities))


def build(root, ptb_metadata, originals, out, *, download_originals=False, require_complete=True):
    root, originals, out = Path(root), Path(originals), Path(out)
    if out.exists():
        raise FileExistsError('A new catalog directory is required')
    metadata = root / 'final_data/metadata.csv'
    with metadata.open() as stream:
        rows = list(csv.DictReader(stream))
    groups = group_pages(rows)
    with Path(ptb_metadata).open() as stream:
        ptb = {int(r['ecg_id']): r for r in csv.DictReader(stream)}
    index_path = root / 'zip-index.json'
    index = {r['name']: r for r in json.loads(index_path.read_text())}
    for name in ['metadata.csv', 'data/leads.npz', 'data/rhythms.npz']:
        path = root / 'final_data' / name
        entry = index['final_data/' + name]
        payload = path.read_bytes()
        if len(payload) != entry['size'] or zlib.crc32(payload) != entry['crc']:
            raise ValueError('Metadata/reference source CRC mismatch')
    if download_originals:
        fetch_originals(rows, ptb, originals)
    recordings = sorted({recording_id(r['ECG ID']) for r in rows})
    patients = sorted({'ptbxl-patient-' + str(float(ptb[i]['patient_id'])) for i in recordings})
    conditions = sorted({condition for _, condition in groups})
    if require_complete and (len(rows), len(groups), len(recordings), len(patients), len(conditions)) != (6600, 6000, 100, 97, 60):
        raise ValueError('This catalog does not match the complete fixed public PM release')
    for condition in conditions:
        if len({identity for identity, c in groups if c == condition}) != len(recordings):
            raise ValueError('Condition is missing an original exam')
    out.mkdir(parents=True)
    (out / 'references').mkdir()
    source_audits, windows, references = {}, {}, {}
    with np.load(root / 'final_data/data/leads.npz', allow_pickle=False) as leads, \
            np.load(root / 'final_data/data/rhythms.npz', allow_pickle=False) as rhythms:
        used = {r['ECG ID'] for r in rows}
        if set(leads.files) != used or set(rhythms.files) != used:
            raise ValueError('Reference bank has missing or unused identities')
        for identity in recordings:
            base = f'LPAE_{identity:05d}_hr'
            header = originals / f'{identity:05d}_hr.hea'
            full = read_wfdb16(header)
            if any(s.fs != 500 for s in full.values()):
                raise ValueError('Expected original 500 Hz source')
            starts, checks = infer_printed_windows(leads[base], np.stack([full[name].values for name in LEADS]))
            windows[identity] = starts
            source_audits[str(identity)] = {'patient_id': 'ptbxl-patient-' + str(float(ptb[identity]['patient_id'])),
                'header_sha256': sha256(header), 'data_sha256': sha256(header.with_suffix('.dat')), 'checks': checks}
        for name in sorted(used):
            values, rhythm_names = canonical_reference(leads[name], rhythms[name], windows[recording_id(name)])
            path = out / 'references' / (name + '.npz')
            np.savez_compressed(path, values_mv=values, fs=500)
            references[name] = {'reference': str(path.relative_to(out)), 'reference_sha256': sha256(path),
                'printed_start_samples_scoring_only': windows[recording_id(name)],
                'rhythm_leads_scoring_only': rhythm_names, 'printed_samples': int(np.isfinite(values).sum())}
    cases, acquired = [], 0
    for (identity, condition), pages in sorted(groups.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        first = pages[0]
        source = first['ECG ID']
        recording = recording_id(source)
        image_records = []
        for row in sorted(pages, key=lambda r: int(r['Image page'])):
            name = 'final_data/visual_data/' + row['Image relative path']
            entry, path = index[name], root / name
            image_hash = None
            if path.exists():
                payload = path.read_bytes()
                if len(payload) != entry['size'] or zlib.crc32(payload) != entry['crc']:
                    raise ValueError('Acquired image CRC mismatch')
                image_hash, acquired = sha256(path), acquired + 1
            image_records.append({'image_id': f'pm-{identity}-{condition}-page-{row["Image page"]}',
                'page_index': int(row['Image page']), 'image': name, 'image_sha256': image_hash,
                'archive_member': entry})
        cases.append({'exam_id': f'pm-{identity}-{condition}', 'recording_id': recording,
            'patient_id': source_audits[str(recording)]['patient_id'], 'condition': condition,
            'waveform_source_id': source, 'pages': image_records, **references[source],
            'source_format_scoring_only': first['ECG format']})
    catalog = {'schema': 'complete-pm-ecg-id-exam-catalog-v1', 'source': 'https://zenodo.org/records/13617673',
        'status': 'reference_catalog_complete_images_verified' if acquired == len(rows) else 'reference_catalog_complete_image_acquisition_partial',
        'exams': len(cases), 'page_files': len(rows),
        'base_recordings': len(recordings), 'patients': len(patients), 'conditions': conditions,
        'reference_variants': len(references), 'acquired_images_verified': acquired,
        'all_image_bytes_acquired': acquired == len(rows), 'fs': 500, 'samples': 5000, 'units': 'mV',
        'metadata_sha256': sha256(metadata), 'ptbxl_metadata_sha256': sha256(ptb_metadata),
        'archive_index_sha256': sha256(index_path), 'reference_source_sha256': {n: sha256(root / 'final_data/data' / n) for n in ['leads.npz', 'rhythms.npz']},
        'source_dc_offsets_preserved_in_scoring_references': True,
        'time_windows_verified_against_original_ptbxl': True,
        'whole_image_archive_md5_verified': False, 'independent_test_status': 'public benchmark with previously inspected development examples',
        'author_failure_and_alignment_protocol_exactly_reproduced': False, 'cases': cases}
    write_json(out / 'catalog.json', catalog)
    write_json(out / 'original-source-audit.json', {'schema': 'pm-original-cutout-time-audit-v1', 'source': PTB_URL,
        'recordings': source_audits, 'checks': len(recordings) * 12, 'passed': True})
    write_json(out / 'training-exclusions.json', {'schema': 'pm-patient-training-exclusions-v1',
        'recording_ids': recordings, 'patient_ids': patients,
        'exclude_all_recordings_and_generated_images_of_these_patients': True,
        'upstream_published_model_training_overlap': 'unknown', 'catalog_sha256': sha256(out / 'catalog.json')})
    return {k: v for k, v in catalog.items() if k not in ['cases', 'conditions']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'ptb-metadata', 'originals', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--download-originals', action='store_true')
    args = p.parse_args()
    print(json.dumps(build(args.root, args.ptb_metadata, args.originals, args.out, download_originals=args.download_originals)))


if __name__ == '__main__':
    main()
