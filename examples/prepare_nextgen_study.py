"""Frozen synthetic M2 cohort with old development and new source patients.

Selection is made from public metadata before any new image is rendered.
Renderer material coordinates and node identities are evaluation targets only.
Every scheduled case, including a failed read/render, remains in attempts.json.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.data import read_wfdb16
from ecg_gpu.schema_v05 import LEADS
from examples.diverse_grid_study import render_one


BASE = 'https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/'
SELECTION_SEED = 20261022
DEFAULT_PROTOCOL = (Path(__file__).resolve().parents[1]
                    / 'research/rectification-m2-2026-10-22/data-protocol.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def canonical_patient(value):
    text = str(value).removeprefix('ptbxl-patient-')
    return str(int(float(text)))


def download(url, path, limit):
    path = Path(path)
    if path.exists():
        if path.stat().st_size > limit:
            raise ValueError('Existing download exceeds fixed bound')
        return
    error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(limit+1)
            if len(data) > limit:
                raise ValueError('Public source exceeds fixed download bound')
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix+'.part')
            temporary.write_bytes(data)
            temporary.replace(path)
            return
        except (OSError, TimeoutError) as exc:
            error = exc
            if attempt == 2:
                break
    raise RuntimeError('Public source download failed') from error


def freeze_selection(sources, root, protocol_path):
    """Freeze groups and scheduled views without reading image observations."""
    sources, root, protocol_path = map(Path, (sources, root, protocol_path))
    original = json.loads(sources.read_text())
    metadata = root/'source/ptbxl_database.csv'
    download(BASE+'ptbxl_database.csv', metadata, 16000000)
    if digest(metadata) != original['metadata_sha256']:
        raise ValueError('Official metadata bytes changed from the frozen source cohort')
    previous_groups = {canonical_patient(r['group_id']) for r in original['records']}
    excluded = {canonical_patient(p) for p in original['excluded_patient_ids']}
    blocked = previous_groups | excluded
    rows = list(csv.DictReader(io.StringIO(metadata.read_text())))
    candidates = {}
    for row in sorted(rows, key=lambda r: int(r['ecg_id'])):
        patient = canonical_patient(row['patient_id'])
        if int(row['strat_fold']) == 10 and patient not in blocked:
            candidates.setdefault(patient, row)
    order = sorted(candidates, key=lambda p: (
        hashlib.sha256(f'{SELECTION_SEED}:{p}'.encode()).hexdigest(), int(p)))
    if len(order) < 8:
        raise ValueError('Fewer than eight unseen fold-10 patients')
    records = []
    for index in range(1024, 1030):
        row = original['records'][index]
        if row['split'] != 'validation':
            raise ValueError('Frozen development indices no longer denote validation')
        records.append({'source_index': index, **row,
                        'role': 'development', 'patient_group': canonical_patient(row['group_id'])})
    for patient in order[:8]:
        row = candidates[patient]
        ecg_id = int(row['ecg_id'])
        records.append({'source_index': 200000+ecg_id, 'source_id': f'ptbxl-{ecg_id}',
                        'group_id': f'ptbxl-patient-{patient}.0', 'patient_group': patient,
                        'split': 'test', 'role': 'new_patient_assessment',
                        'filename_lr': row['filename_lr'], 'strat_fold': 10})
    evaluation_groups = {r['patient_group'] for r in records if r['split'] == 'test'}
    if len(evaluation_groups) != 8 or evaluation_groups & blocked:
        raise ValueError('Patient overlap in new assessment')
    protocol = {
        'schema_version': 1, 'protocol_date': '2026-10-22',
        'source': original['source'], 'source_s3': BASE,
        'citation': original['citation'], 'license': original['license'],
        'previous_cohort_sha256': digest(sources),
        'official_metadata_sha256': digest(metadata),
        'selection_seed': SELECTION_SEED,
        'selection_rule': 'Fold10 patients absent from all 1280 previous cohort groups and all 77 exclusions. One earliest ECG per patient. Sort SHA256(seed:patient), take first eight.',
        'previous_cohort_groups': len(previous_groups),
        'previous_explicit_exclusions': len(excluded),
        'eligible_unseen_fold10_patients': len(candidates),
        'assessment_groups_overlap_previous': False,
        'selection_observed_new_images': False,
        'new_model_training': False, 'clinical_external_validation': False,
        'study_kind': 'synthetic CPU/LLVM observations with source patients unseen by the frozen trained model',
        'development': 'Six previously inspected validation groups, 48 captures',
        'assessment': 'Eight source-patient groups not in prior cohort, 64 captures',
        'views_per_patient': 8, 'expected_attempted_captures': len(records)*8,
        'renderer_variant': 'llvm_ad_rgb', 'render_one_unchanged': True,
        'render_source_sha256': digest(Path(__file__).with_name('diverse_grid_study.py')),
        'new_capture_index_rule': '200000+public_ecg_id, used only for deterministic geometry and noise seeds',
        'gold_contract': 'reference_mm, xy, ids and valid are renderer gold for evaluation only, never inference inputs',
        'failures': 'Every scheduled source/view remains in attempts.json. No replacement patient or hidden retry with different seed.',
        'records': records,
    }
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Frozen protocol would change. Use a new version explicitly.')
    else:
        save_json(protocol_path, protocol)
    save_json(root/'protocol.json', protocol)
    print(json.dumps({'selection_frozen': True, 'protocol_sha256': digest(protocol_path),
                      'assessment_groups': sorted(evaluation_groups), 'expected': len(records)*8}), flush=True)
    return protocol


def audit_manifest(root, manifest):
    """Check exported coordinate composition without an inferred rectifier.

    This is an export-contract diagnostic, not independent real-paper metrology.
    Bilinear material coordinates at rendered grid nodes should match their IDs.
    Nodes whose four interpolation neighbours are not visible are counted apart.
    No case is removed or selected using this diagnostic.
    """
    records = []
    for row in manifest:
        path = root/row['file']
        if digest(path) != row['sha256']:
            raise ValueError('Capture hash changed during final export audit')
        with np.load(path) as data:
            image, field, valid = data['image'], data['reference_mm'], data['valid']
            xy, ids = data['xy'].astype(float), data['ids']
            if image.shape != (704, 1024, 3) or field.shape != (704, 1024, 2):
                raise ValueError('Unexpected image/material shape')
            if valid.shape != image.shape[:2] or xy.shape != ids.shape:
                raise ValueError('Unaligned export arrays')
            lo = np.floor(xy).astype(int)
            hi = lo+1
            inside = ((lo >= 0) & (hi < [field.shape[1], field.shape[0]])).all(1)
            lo, hi, points, labels = lo[inside], hi[inside], xy[inside], ids[inside]
            visible = ((valid[lo[:, 1], lo[:, 0]] >= .99)
                       & (valid[hi[:, 1], lo[:, 0]] >= .99)
                       & (valid[lo[:, 1], hi[:, 0]] >= .99)
                       & (valid[hi[:, 1], hi[:, 0]] >= .99))
            wx, wy = (points-lo).T
            sampled = (field[lo[:, 1], lo[:, 0]]*((1-wx)*(1-wy))[:, None]
                       + field[hi[:, 1], lo[:, 0]]*((1-wx)*wy)[:, None]
                       + field[lo[:, 1], hi[:, 0]]*(wx*(1-wy))[:, None]
                       + field[hi[:, 1], hi[:, 0]]*(wx*wy)[:, None])
            error = np.linalg.norm(sampled-labels[:, ::-1]*5., axis=1)[visible]
            records.append({'key': row['key'], 'split': row['split'],
                            'nodes': len(xy), 'fully_visible_bilinear_nodes': len(error),
                            'excluded_boundary_or_visibility_nodes': len(xy)-len(error),
                            'median_mm': float(np.median(error)) if len(error) else None,
                            'p95_mm': float(np.quantile(error, .95)) if len(error) else None,
                            'maximum_mm': float(error.max()) if len(error) else None})
    audit = {'scope': 'Export chain consistency, not independent physical accuracy',
             'inferred_rectifier_used': False, 'cases_filtered': False,
             'capture_hashes_verified': len(records), 'records': records}
    save_json(root/'gold-node-consistency.json', audit)
    return audit


def prepare(sources, root, protocol_path=DEFAULT_PROTOCOL):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    protocol = freeze_selection(sources, root, protocol_path)
    raw = root/'wfdb'
    raw.mkdir(exist_ok=True)
    source_results = {}

    def prepare_source(row):
        name = Path(row['filename_lr']).name
        evidence = dict(row)
        for suffix, key in (('.hea', 'header_sha256'), ('.dat', 'dat_sha256')):
            path = raw/(name+suffix)
            download(BASE+row['filename_lr']+suffix, path, 200000)
            observed_hash = digest(path)
            if key in row and observed_hash != row[key]:
                raise ValueError('Known original source checksum mismatch: '+row['source_id'])
            evidence[key] = observed_hash
        signal = read_wfdb16(raw/(name+'.hea'))
        evidence['wfdb_header_and_channel_checksums_validated'] = True
        evidence['source_sha256_status'] = ('compared_with_prior_frozen_hashes'
                                          if row['split'] == 'validation' else 'recorded_from_publisher_bytes')
        if {s.fs for s in signal.values()} != {100}:
            raise ValueError('This protocol requires original 100 Hz records')
        return evidence

    with ThreadPoolExecutor(6) as pool:
        futures = [(r, pool.submit(prepare_source, r)) for r in protocol['records']]
        for row, future in futures:
            try:
                evidence = future.result()
                source_results[row['source_id']] = {'status': 'ready', **evidence}
                print(json.dumps({'source_ready': row['source_id'], 'split': row['split']}), flush=True)
            except Exception as exc:
                source_results[row['source_id']] = {'status': 'source-failed', **row,
                                                    'error': type(exc).__name__+': '+str(exc)}
    save_json(root/'source-records.json', list(source_results.values()))
    manifest_path, attempts_path = root/'manifest.json', root/'attempts.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    by_key = {r['key']: r for r in manifest}
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    attempted = {r['key']: r for r in attempts}
    bank = root/'bank'
    bank.mkdir(exist_ok=True)
    import drjit as dr
    dr.set_thread_count(4)
    for row in protocol['records']:
        source = source_results[row['source_id']]
        signal = None
        if source['status'] == 'ready':
            wfdb = read_wfdb16(raw/(Path(row['filename_lr']).name+'.hea'))
            values = np.stack([wfdb[name].values for name in LEADS]).astype(np.float32)
            signal = torch.from_numpy(values)[None]
        for view in range(8):
            key = f'{row["source_index"]:05d}-{view}'
            identity = {'key': key, 'source_id': row['source_id'], 'group_id': row['group_id'],
                        'split': row['split'], 'source_index': row['source_index'], 'view': view}
            if key in by_key:
                entry = by_key[key]
                path = root/entry['file']
                if not path.is_file() or digest(path) != entry['sha256']:
                    raise ValueError('Completed capture is missing or hash changed: '+key)
                continue
            if key in attempted and attempted[key]['status'] == 'failed':
                continue
            started = time.perf_counter()
            try:
                if signal is None:
                    raise RuntimeError(source['error'])
                capture = render_one(signal, 100., row['source_index'], view, row['split'],
                                     variant='llvm_ad_rgb', export_geometry=True)
                path = bank/(key+'.npz')
                temporary = path.with_suffix('.npz.part')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, **{k: v for k, v in capture.items() if k != 'metadata'})
                temporary.replace(path)
                Image.fromarray(capture['image']).save(bank/(key+'.png'))
                entry = {**identity, 'file': str(path.relative_to(root)), 'sha256': digest(path),
                         'observation': 'CPU float32 and LLVM, synthetic assessment',
                         **capture['metadata']}
                by_key[key] = entry
                attempted[key] = {**identity, 'status': 'rendered', 'file': entry['file'],
                                  'sha256': entry['sha256'], 'seconds': time.perf_counter()-started}
            except Exception as exc:
                attempted[key] = {**identity, 'status': 'failed',
                                  'error': type(exc).__name__+': '+str(exc),
                                  'seconds': time.perf_counter()-started}
            save_json(manifest_path, list(by_key.values()))
            save_json(attempts_path, list(attempted.values()))
            print(json.dumps({'case': key, 'split': row['split'],
                              'status': attempted[key]['status'], 'completed': len(by_key),
                              'attempted': len(attempted), 'expected': protocol['expected_attempted_captures'],
                              'seconds': attempted[key]['seconds'],
                              **({'error': attempted[key]['error']} if attempted[key]['status'] == 'failed' else {})}), flush=True)
    provenance = {k: protocol[k] for k in ('source', 'citation', 'license', 'gold_contract', 'study_kind')}
    provenance.update(protocol_sha256=digest(protocol_path), source_records_sha256=digest(root/'source-records.json'),
                      completed_captures=len(by_key), attempted_captures=len(attempted),
                      expected_captures=protocol['expected_attempted_captures'],
                      failed_captures=sum(a['status'] == 'failed' for a in attempted.values()),
                      cuda_execution=False, new_training=False)
    audit_manifest(root, list(by_key.values()))
    provenance['gold_export_audit_sha256'] = digest(root/'gold-node-consistency.json')
    provenance['manifest_sha256'] = digest(manifest_path)
    save_json(root/'provenance.json', provenance)
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-path', type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(prepare(**vars(args)), indent=2), flush=True)
