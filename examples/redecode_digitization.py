"""Auditably apply notebook-linear gaps to saved, losslessly upsampled rows.

No network is rerun and no threshold is selected using the reference signals.
The original report and prediction hashes remain recorded as parents.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.data import load_manifest
from ecg_gpu.digitization.evaluation import OfficialMetric
from ecg_gpu.digitization.runner import _summarize, _write_json, _write_prediction
from ecg_gpu.digitization.signals import replay_linear_gaps, split_rows


def redecode(run, manifest, assets, out):
    run, manifest, out = map(Path, (run, manifest, out))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    report = json.loads((run/'report.json').read_text())
    if report['status'] not in ('completed', 'completed_with_failures'):
        raise ValueError('Source evaluation is incomplete')
    if report['manifest_sha256'] != sha256_file(manifest):
        raise ValueError('Source manifest changed')
    cases = {c.image_id: c for c in load_manifest(manifest)}
    if set(cases) != {c['image_id'] for c in report['cases']}:
        raise ValueError('Source case membership disagrees with manifest')
    metric = OfficialMetric(assets)
    report = copy.deepcopy(report)
    report['status'] = 'running'
    report['replay'] = {'parent_report_sha256': sha256_file(run/'report.json'),
        'driver_sha256': sha256_file(__file__),
        'signal_code_sha256': sha256_file(Path(__file__).resolve().parents[1]/'ecg_gpu/digitization/signals.py'),
        'operation': 'Fourier-invert saved upsampled rows, linearly fill missing columns, resample',
        'network_rerun': False, 'changed_cases': []}
    out.mkdir(parents=True, exist_ok=True)
    (out/'predictions').mkdir()
    started = time.perf_counter()
    for index, record in enumerate(report['cases']):
        if record['status'] != 'success':
            continue
        source = run/record['prediction']
        if sha256_file(source) != record['prediction_sha256']:
            raise ValueError('Source prediction changed')
        with np.load(source, allow_pickle=False) as data:
            payload = {name: data[name].copy() for name in data.files}
        case = cases[record['image_id']]
        rows = replay_linear_gaps(payload['rows_mv'], payload['observed_columns'])
        if not payload['observed_columns'].all():
            report['replay']['changed_cases'].append(record['image_id'])
        leads = split_rows(rows, case.fs)
        payload.update({name: lead.values for name, lead in leads.items()})
        payload['rows_mv'] = rows
        prediction = out/record['prediction']
        _write_prediction(prediction, payload)
        record['provenance']['parent_prediction_sha256'] = record['prediction_sha256']
        record['provenance']['gap_interpolation'] = 'linear'
        record['prediction_sha256'] = sha256_file(prediction)
        if case.reference is not None:
            record['evaluation'] = metric.image(case.reference, leads, image_id=f'case{index}')
    _summarize(report, metric)
    report['status'] = 'completed' if report['failures'] == 0 else 'completed_with_failures'
    report['replay']['seconds'] = time.perf_counter()-started
    _write_json(out/'report.json', report)
    print(json.dumps({'changed_cases': report['replay']['changed_cases'],
                      'official_snr_db': report['official_snr_db']}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'manifest', 'assets', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    redecode(**vars(parser.parse_args()))
