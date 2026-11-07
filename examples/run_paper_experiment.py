"""Run matched-device frozen-model paper experiments with explicit setup paths.

Inputs, external weights and external source are prepared before this command.
This script does not rent machines, download weights or train a model.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.reference import HengTangDigitizer
from ecg_gpu.digitization.runner import _write_json, _write_prediction
from ecg_gpu.digitization.signals import LEADS
from examples.digitization_layout_probe import waveform_svg
from examples.digitization_npj_probe import evaluate_missing
from examples.digitization_npj_profiles import run as run_npj_profiles


def run_kaggle(inputs, out, assets, *, device):
    inputs, out = Path(inputs), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    manifest = json.loads((inputs/'manifest.json').read_text())
    if sha256_file(inputs/'source.npz') != manifest['source_sha256']:
        raise ValueError('Source signal checksum mismatch')
    with np.load(inputs/'source.npz') as source:
        values, fs = source['values_mv'], int(source['fs'])
    model = HengTangDigitizer(assets, device=device, precision='float32', profile='single')
    out.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'ecg-kaggle-paper-transfer-v1', 'status': 'running',
        'device': device, 'precision': 'float32', 'profile': 'single',
        'input_manifest_sha256': sha256_file(inputs/'manifest.json'),
        'weights_changed': False, 'training_performed': False,
        'assumed_layout': '3x4+r1', 'metric': 'same missing-aware +/-100 ms comparison as npj arms',
        'full_ensemble_limit': 'Stage0/1 geometry is shared by single and full profiles', 'cases': []}
    last_stage = [None]
    hooks = [getattr(model, f'net{i}').register_forward_pre_hook(
        lambda module, args, i=i: last_stage.__setitem__(0, i)) for i in range(3)]
    try:
        for case in manifest['cases']:
            name = case['name']
            record = {'name': name, 'image_sha256': case['image_sha256'],
                'layout_assumption_matches_input': case['layout'] == '3x4+r1'}
            print(json.dumps({'starting_kaggle': name}), flush=True)
            if sha256_file(inputs/case['image']) != case['image_sha256']:
                raise ValueError('Input PNG checksum mismatch')
            started = time.perf_counter()
            last_stage[0] = None
            log = io.StringIO()
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            try:
                with Image.open(inputs/case['image']) as image:
                    rgb = np.array(image.convert('RGB'))
                with contextlib.redirect_stdout(log):
                    result, debug = model.predict(rgb, fs=fs)
                prediction = np.full_like(values, np.nan)
                for i, lead in enumerate(LEADS):
                    signal = result.leads[lead]
                    start = round(signal.start_s*fs)
                    end = start+len(signal.values)
                    if end > prediction.shape[1] or signal.fs != fs:
                        raise ValueError('Reference prediction does not fit the requested acquisition')
                    prediction[i, start:end] = signal.values
                record.update(status='returned', diagnostics=result.diagnostics,
                    evaluation=evaluate_missing(values, prediction, fs, case['metadata']))
                _write_prediction(out/(name+'-prediction.npz'), {'values_mv': prediction,
                    'fs': np.asarray(fs), 'lead_names': np.array(LEADS), 'finite': np.isfinite(prediction)})
                display = SimpleNamespace(leads={lead: SimpleNamespace(values=prediction[i], fs=fs, start_s=0.)
                                                for i, lead in enumerate(LEADS)})
                waveform_svg(out/(name+'-waves.svg'), values, fs, case['metadata'], display)
                del result, debug
            except (ValueError, RuntimeError, model.cv2.error) as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            if device == 'cuda':
                torch.cuda.synchronize()
                record['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
                torch.cuda.empty_cache()
            record.update(last_network_stage=last_stage[0], wall_seconds=time.perf_counter()-started)
            (out/(name+'-upstream.log')).write_text(log.getvalue())
            report['cases'].append(record)
            _write_json(out/'report.json', report)
            print(json.dumps(record), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    report['status'] = 'completed'
    _write_json(out/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--npj-source', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    environment = {'torch': str(torch.__version__), 'cuda_available': torch.cuda.is_available(),
        'cuda_runtime': torch.version.cuda, 'device': args.device,
        'gpu_name': torch.cuda.get_device_name() if args.device == 'cuda' else None,
        'tf32_matmul': False, 'tf32_cudnn': False}
    _write_json(args.out/'environment.json', environment)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run_npj_profiles(args.inputs, args.out/'npj-profiles', args.npj_source, device=args.device)
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    run_kaggle(args.inputs, args.out/'kaggle', args.assets, device=args.device)
