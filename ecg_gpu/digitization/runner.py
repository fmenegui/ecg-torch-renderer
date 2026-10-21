"""Batch orchestration with persistent failures, provenance and signal scoring."""

import json
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Protocol

import numpy as np
from PIL import Image
import torch

from .artifacts import sha256_file
from .data import load_manifest
from .evaluation import OfficialMetric, zero_prediction
from .signals import DigitizationResult


class Digitizer(Protocol):
    def predict(self, image, *, fs, progress=None) -> tuple[DigitizationResult, dict]:
        ...


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _write_prediction(path, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **payload)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_manifest(manifest, out, digitizer: Digitizer, metric: OfficialMetric, *, progress=None):
    """Run every case, keeping a failed image in the evaluation denominator.

    Failed reconstructions produce no waveform artifact. If truth exists, the
    official score uses an explicitly reported zero-prediction failure penalty.
    Outputs contain no source images or rectified scans.
    """
    cases = load_manifest(manifest)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    (out/'predictions').mkdir()
    report = {'schema_version': 1, 'status': 'running',
        'manifest_sha256': sha256_file(manifest), 'cases_total': len(cases),
        'patients': len({case.patient_id for case in cases}),
        'metric': {'name': 'official Kaggle ECG metric v7', 'source_sha256': metric.source_sha256,
            'max_alignment_seconds': .2, 'power_ratio_cap': 1e6,
            'aggregation': 'mean capped linear image ratios, then 10*log10',
            'failure_policy': 'explicit zero-prediction penalty when reference is available'},
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(), 'cpu_threads': torch.get_num_threads(),
            'machine': platform.machine()},
        'limits': ['Development smoke evaluation, not the private Kaggle test',
            'Unknown overlap with upstream model training data',
            'Patient grouping does not establish distinct devices or sites',
            'Official alignment can mask timing and DC errors, inspect raw RMSE too'],
        'cases': []}
    _write_json(out/'report.json', report)
    started = time.perf_counter()
    for case in cases:
        if progress:
            progress({'stage': 'case', 'image_id': case.image_id, 'image_type': case.image_type})
        record = {'image_id': case.image_id, 'patient_id': case.patient_id,
            'image_type': case.image_type, 'image_sha256': case.image_sha256,
            'source': case.source, 'reference_available': case.reference is not None}
        t0 = time.perf_counter()
        try:
            with Image.open(case.image) as image:
                rgb = np.array(image.convert('RGB'))
            result, debug = digitizer.predict(rgb, fs=case.fs, progress=progress)
            if not isinstance(result, DigitizationResult):
                raise TypeError('Backend did not return a DigitizationResult')
            # Score before committing a successful prediction. A malformed lead
            # length/calibration is an inference failure, not a dropped score.
            if case.reference is not None:
                record['evaluation'] = metric.image(case.reference, result.leads,
                    image_id='case'+str(len(report['cases'])))
            payload = {name: lead.values for name, lead in result.leads.items()}
            payload.update(fs=np.asarray(case.fs), units=np.asarray('mV'),
                lead_names=np.asarray(list(result.leads)),
                start_s=np.asarray([lead.start_s for lead in result.leads.values()]))
            # Debug arrays use geometric and waveform information only.
            for name in ('grid_normalized_xy', 'homography', 'rows_mv', 'observed_columns'):
                if name in debug:
                    payload[name] = debug[name]
            prediction = out/'predictions'/f'{case.image_id}.npz'
            _write_prediction(prediction, payload)
            record.update(status='success', provenance=result.provenance,
                diagnostics=result.diagnostics, warnings=result.warnings,
                prediction=str(prediction.relative_to(out)), prediction_sha256=sha256_file(prediction))
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            if case.reference is not None:
                record['evaluation'] = metric.image(case.reference, zero_prediction(case.reference),
                    image_id='case'+str(len(report['cases'])))
                record['evaluation']['failure_penalty'] = True
        record['wall_seconds'] = time.perf_counter()-t0
        report['cases'].append(record)
        _summarize(report, metric)
        _write_json(out/'report.json', report)
        if progress:
            progress({'stage': 'result', 'image_id': case.image_id, 'status': record['status'],
                'snr_db': record.get('evaluation', {}).get('official_snr_db'),
                'wall_seconds': record['wall_seconds']})
    report['status'] = 'completed' if report['failures'] == 0 else 'completed_with_failures'
    report['wall_seconds'] = time.perf_counter()-started
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['process_peak_rss_bytes'] = rss if sys.platform == 'darwin' else rss*1024
    except ImportError:
        report['process_peak_rss_bytes'] = None
    _write_json(out/'report.json', report)
    return report


def _summarize(report, metric):
    cases = report['cases']
    report['successes'] = sum(case['status'] == 'success' for case in cases)
    report['failures'] = len(cases)-report['successes']
    report['scored_cases'] = sum('evaluation' in case for case in cases)
    report['official_snr_db'] = metric.aggregate([case['evaluation'] for case in cases if 'evaluation' in case])
    report['by_image_type'] = {}
    for kind in sorted({case['image_type'] for case in cases}):
        group = [case for case in cases if case['image_type'] == kind]
        report['by_image_type'][kind] = {'images': len(group),
            'failures': sum(case['status'] == 'failed' for case in group),
            'official_snr_db': metric.aggregate([case['evaluation'] for case in group if 'evaluation' in case])}
