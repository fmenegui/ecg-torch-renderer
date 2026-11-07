"""Prepare a small PTB-XL 1.0.3 cohort from its downloaded metadata CSV.

Explicitly downloads selected public WFDB files, outside training. Example:
curl -o ptbxl_database.csv https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv
python examples/prepare_grid_benchmark.py --metadata ptbxl_database.csv --out build/cohort
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.schema_v05 import LEADS


def prepare(metadata, out, *, train=16, validation=4, test=4, seed=1973, wfdb_cache=None,
            excluded_patients=None):
    counts = {'train': train, 'validation': validation, 'test': test}
    if any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('Positive source counts required')
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty cohort directory')
    with Path(metadata).open() as stream:
        rows = list(csv.DictReader(stream))
    excluded = {str(int(float(p))) for p in (excluded_patients or [])}
    rows = [r for r in rows if str(int(float(r['patient_id']))) not in excluded]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    selected, used = [], set()
    for split, count in counts.items():
        candidates = [r for r in rows if ('train' if int(r['strat_fold']) <= 8 else 'validation' if int(r['strat_fold']) == 9 else 'test') == split]
        chosen = 0
        for row in candidates:
            patient = str(int(float(row['patient_id'])))
            if patient not in used:
                used.add(patient)
                selected.append((row, split))
                chosen += 1
                if chosen == count:
                    break
        if chosen != count:
            raise ValueError('Not enough independent patient groups')
    out.mkdir(parents=True, exist_ok=True)
    raw = out/'wfdb'
    raw.mkdir()

    def download(pair):
        row, split = pair
        stem = row['filename_lr']
        if not re.fullmatch(r'records100/[0-9]{5}/[0-9]{5}_lr', stem):
            raise ValueError('Unexpected PTB-XL filename')
        paths = []
        for suffix in ('.hea', '.dat'):
            path = raw/(Path(stem).name+suffix)
            url = 'https://physionet.org/files/ptb-xl/1.0.3/'+stem+suffix
            cached = Path(wfdb_cache)/path.name if wfdb_cache is not None else None
            if cached is not None and cached.is_file():
                shutil.copy2(cached, path)
            else:
                subprocess.run(['curl', '--fail', '--silent', '--show-error', '--retry', '3',
                                '--connect-timeout', '10', '--max-time', '35',
                                '--max-filesize', '200000', '-o', str(path), url], check=True, timeout=180)
            paths.append(path)
        lines = paths[0].read_text().splitlines()
        record_name, leads, fs, samples = lines[0].split()[:4]
        if record_name != Path(stem).name:
            raise ValueError('WFDB header record identity does not match selected source')
        if (int(leads), int(fs), int(samples)) != (12, 100, 1000):
            raise ValueError('Expected 12-lead, 100 Hz, 10-second PTB-XL record')
        values = np.frombuffer(paths[1].read_bytes(), dtype='<i2').reshape(1000, 12).T
        physical = []
        for index, line in enumerate(lines[1:13]):
            parts = line.split()
            gain = re.fullmatch(r'([0-9.]+)\((-?[0-9]+)\)/mV', parts[2])
            if parts[0] != Path(stem).name+'.dat' or parts[1] != '16' or gain is None or parts[-1].upper() != LEADS[index].upper():
                raise ValueError('Unexpected WFDB channel encoding')
            if (values[index, 0] != int(parts[5]) or int(values[index].astype(np.int64).sum()) % 65536 != int(parts[6]) % 65536
                    or (values[index] == -32768).any() or float(gain[1]) <= 0):
                raise ValueError('WFDB checksum, initial value or sample validity failed')
            physical.append((values[index].astype(np.float32)-int(gain[2]))/float(gain[1]))
        return np.stack(physical), {'source_id': 'ptbxl-'+row['ecg_id'], 'group_id': 'ptbxl-patient-'+row['patient_id'],
            'split': split, 'filename_lr': stem, 'strat_fold': int(row['strat_fold']),
            'header_sha256': hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            'dat_sha256': hashlib.sha256(paths[1].read_bytes()).hexdigest()}

    # Bound queued work. A denied download must not leave the whole cohort
    # scheduled or retry HTTP 403 responses as if they were transient failures.
    pool = ThreadPoolExecutor(6)
    pending, downloaded, cursor = {}, [None]*len(selected), 0
    try:
        while cursor < len(selected) or pending:
            while cursor < len(selected) and len(pending) < 6:
                pending[pool.submit(download, selected[cursor])] = cursor
                cursor += 1
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                index = pending.pop(future)
                downloaded[index] = future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    provenance = [item[1] for item in downloaded]
    np.savez_compressed(out/'signals.npz', signals=np.stack([item[0] for item in downloaded]), fs=100.,
        source_ids=np.array([r['source_id'] for r in provenance]), group_ids=np.array([r['group_id'] for r in provenance]),
        splits=np.array([r['split'] for r in provenance]), lead_names=np.array(LEADS), units='mV')
    report = {'source': 'https://physionet.org/content/ptb-xl/1.0.3/', 'citation': 'doi:10.1038/s41597-020-0495-6',
        'license': 'CC BY 4.0', 'metadata_sha256': hashlib.sha256(Path(metadata).read_bytes()).hexdigest(),
        'signals_sha256': hashlib.sha256((out/'signals.npz').read_bytes()).hexdigest(),
        'seed': seed, 'counts': counts, 'records': provenance,
        'excluded_patient_ids': sorted(excluded),
        'signal_processing': 'Original ADC-to-mV conversion only, no physiological augmentation'}
    (out/'sources.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wfdb-cache', type=Path)
    parser.add_argument('--excluded-patients', type=Path,
                        help='JSON list of PTB-XL patient IDs excluded before sampling')
    for name, default in (('train', 16), ('validation', 4), ('test', 4), ('seed', 1973)):
        parser.add_argument('--'+name, type=int, default=default)
    arguments = vars(parser.parse_args())
    excluded_path = arguments.pop('excluded_patients')
    arguments['excluded_patients'] = json.loads(excluded_path.read_text()) if excluded_path else None
    print(json.dumps(prepare(**arguments)['counts']))
