"""Controlled paper-profile adaptation of the external npj reference.

Published neural weights stay frozen. The configured mode changes orientation,
grid-search bounds and the global candidate catalog. Gain is supplied explicitly
as a declared paper convention, never claimed to be inferred from RGB.
"""
import argparse
import base64
import contextlib
import copy
import html
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
from ecg_gpu.digitization.runner import _write_json, _write_prediction
from ecg_gpu.digitization.signals import LEADS
from examples.digitization_layout_probe import waveform_svg
from examples.digitization_npj_probe import EXPECTED_LAYOUT, evaluate_missing, load_upstream


CABRERA = {'description': 'Cabrera 3x4 with a wildcard rhythm strip',
    'layout': {'rows': 3, 'cols': 4},
    'leads': [['aVL', 'II', 'V1', 'V4'], ['I', 'aVF', 'V2', 'V5'], ['-aVR', 'III', 'V3', 'V6']],
    'rhythm_leads': ['Any'], 'total_rows': 4}


def apply_profile(model, identifier, layouts, configured):
    """The same candidate set is used for every image in a given arm."""
    model.rotate_on_resample = not configured
    model.pixel_size_finder.min_number_of_grid_lines = 15 if configured else 30
    model.pixel_size_finder.max_number_of_grid_lines = 120 if configured else 70
    identifier.layouts = copy.deepcopy(layouts)
    if configured:
        identifier.layouts['cabrera_3x4_with_r1'] = copy.deepcopy(CABRERA)


def write_page(out, inputs, manifest, report):
    def embed(path):
        mime = 'image/svg+xml' if path.suffix == '.svg' else 'image/png'
        return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()
    records = {(c['arm'], c['name']): c for c in report['cases']}
    page = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ECG · adaptação por perfis de papel</title><style>body{font:17px/1.6 system-ui;background:#0b1220;color:#e2e8f0;max-width:1240px;margin:auto;padding:30px}h1{line-height:1.2}article{background:#162235;border-radius:14px;padding:22px;margin:24px 0}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #334155}img{width:100%;height:auto}a,summary{color:#7dd3fc}summary{cursor:pointer}.notice{border-left:4px solid #fbbf24;padding-left:18px}</style>
<h1>O que muda ao configurar o papel?</h1><p>Open-ECG-Digitizer, pesos publicados congelados. Mesmas imagens e mesmo dispositivo nas duas condições.</p>
<p class="notice">O perfil configurado fornece o ganho nominal do papel. Não há reconhecimento automático das unidades. O catálogo é ampliado para Cabrera, a rotação automática de retrato é desativada e a busca da grade aceita 15–120 linhas maiores.</p>
<p>SNR médio por derivação nos trechos observados, alinhamento de até 100 ms e remoção de DC. A cobertura mantém os trechos ausentes no denominador. Layout incompatível, fallback e saída vazia falham explicitamente.</p>
<table><tr><th>Caso</th><th>Referência: SNR / cobertura</th><th>Configurado: SNR / cobertura</th><th>Layout configurado correto?</th></tr>'''
    for case in manifest['cases']:
        page += '<tr><td>'+html.escape(case['name'])+'</td>'
        for arm in ('published', 'configured'):
            c = records.get((arm, case['name']), {})
            e = c.get('evaluation', {})
            if e.get('mean_observed_lead_snr_db') is not None:
                cell = f'{e["mean_observed_lead_snr_db"]:.2f} dB / {e["printed_time_coverage"]:.1%}'
                if c.get('status') == 'failed':
                    cell = 'Rejeitado · diagnóstico: '+cell
                elif not c.get('layout_matches_expected', False):
                    cell = 'Layout incorreto · '+cell
            else:
                cell = html.escape(c.get('status', 'Pendente'))
            page += '<td>'+cell+'</td>'
        c = records.get(('configured', case['name']), {})
        page += '<td>'+str(c.get('layout_matches_expected', 'Pendente'))+'</td></tr>'
    page += '</table>'
    for case in manifest['cases']:
        name = case['name']
        page += f'<article><h2>{html.escape(name)} · {html.escape(case["layout"])}</h2>'
        page += f'<details><summary>Imagem de entrada</summary><img src="{embed(inputs/case["image"])}" alt="Entrada"></details>'
        for arm in ('published', 'configured'):
            c = records.get((arm, name), {})
            wave = out/arm/(name+'-waves.svg')
            page += f'<h3>{arm}</h3><p>{html.escape(c.get("error", c.get("predicted_layout", "Pendente")))}</p>'
            if wave.exists():
                page += f'<details><summary>Inspecionar ondas e lacunas</summary><img src="{embed(wave)}" alt="Original verde e reconstrução laranja"></details>'
        page += '</article>'
    (out/'index.html').write_text(page+'</html>')


def run(inputs, out, source, *, device='cpu', arms=('published', 'configured'),
        dewarping=None, case_names=None):
    inputs, out = Path(inputs).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    if dewarping is not None and type(dewarping) is not bool:
        raise ValueError('dewarping must be bool or None')
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('An available CPU or CUDA device is required')
    manifest = json.loads((inputs/'manifest.json').read_text())
    if sha256_file(inputs/'source.npz') != manifest['source_sha256']:
        raise ValueError('Source signal checksum mismatch')
    with np.load(inputs/'source.npz') as data:
        values, fs = data['values_mv'], int(data['fs'])
    if case_names is not None:
        requested = set(case_names)
        known = {case['name'] for case in manifest['cases']}
        if not requested or requested - known:
            raise ValueError('Select nonempty known case names')
        manifest['cases'] = [case for case in manifest['cases'] if case['name'] in requested]
    model, config, commit = load_upstream(source)
    if dewarping is not None:
        model.apply_dewarping = dewarping
    model.device = str(device)
    model.segmentation_model.to(device)
    identifier = model.identifier
    identifier.device = device
    identifier.unet.to(device)
    layouts = copy.deepcopy(identifier.layouts)
    out.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'ecg-paper-profile-adaptation-v1', 'status': 'running',
        'device': str(device), 'precision': 'float32', 'torch': str(torch.__version__),
        'cuda_runtime': torch.version.cuda, 'cpu_threads': torch.get_num_threads(),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'upstream_commit': commit, 'input_manifest_sha256': sha256_file(inputs/'manifest.json'),
        'apply_dewarping': model.apply_dewarping, 'selected_cases': [c['name'] for c in manifest['cases']],
        'arms': list(arms), 'weights_changed': False, 'training_performed': False,
        'configured_gain_source': 'explicit declared paper gain from generation configuration',
        'configured_gain_is_inferred': False,
        'configured_changes': ['disable portrait-to-landscape rotation',
            'grid search uses class default 15-120 major lines',
            'add Cabrera 3x4 + rhythm to the common candidate catalog',
            'pass declared gain as mV/mm', 'reject fallback, empty or inconsistent row/layout outputs'],
        'limits': ['One artificial signal and one capture per configuration',
                   'Configured mode supplies gain, so it is not fully automatic calibration',
                   'A correct layout and high conditional SNR do not guarantee complete waveforms'], 'cases': []}
    last_match = {}
    gain_mm_per_mv = [10.]

    def capture(*args, **kwargs):
        kwargs['mv_per_mm'] = 1/gain_mm_per_mv[0]
        match = identifier(*args, **kwargs)
        last_match.clear()
        last_match.update(match)
        return match

    model.identifier = capture
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    for arm in arms:
        configured = arm == 'configured'
        if arm not in ('published', 'configured'):
            raise ValueError('Unknown comparison arm')
        apply_profile(model, identifier, layouts, configured)
        directory = out/arm
        directory.mkdir()
        for case in manifest['cases']:
            name = case['name']
            gain_mm_per_mv[0] = float(case['metadata']['gain'][0]) if configured else 10.
            expected = EXPECTED_LAYOUT[case['layout']]
            if configured and case['layout'] == '3x4-cabrera+r1':
                expected = 'cabrera_3x4_with_r1'
            record = {'name': name, 'arm': arm, 'expected_layout': expected,
                'image_sha256': case['image_sha256'], 'provided_gain_mm_per_mv': gain_mm_per_mv[0],
                'gain_source': 'declared paper profile' if configured else 'published fixed assumption'}
            if sha256_file(inputs/case['image']) != case['image_sha256']:
                raise ValueError('Input PNG checksum mismatch')
            print(json.dumps({'starting': name, 'arm': arm}), flush=True)
            last_match.clear()
            log = io.StringIO()
            started = time.perf_counter()
            torch.manual_seed(20261021)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            try:
                with Image.open(inputs/case['image']) as image:
                    rgb = np.array(image.convert('RGB'))
                tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].float()
                with contextlib.redirect_stdout(log), torch.no_grad():
                    result = model(tensor, layout_should_include_substring=None)
                canonical = result['signal']['canonical_lines']
                if canonical is None:
                    raise ValueError('No canonical signal returned')
                prediction = canonical.detach().cpu().numpy()/1000.
                e = evaluate_missing(values, prediction, fs, case['metadata'])
                cost = float(result['signal']['layout_matching_cost'])
                layout = result['layout_name']
                row_count = int(last_match.get('rows_in_layout', 0))
                record.update(predicted_layout=layout, matching_cost=cost if np.isfinite(cost) else None,
                    detected_labels=int(last_match.get('n_detected', 0)), extracted_rows=row_count,
                    layout_matches_expected=bool(np.isfinite(cost)) and layout == expected, evaluation=e,
                    layout_fallback=not np.isfinite(cost),
                    pixel_spacing_mm={k: float(v) for k, v in result['pixel_spacing_mm'].items()})
                _write_prediction(directory/(name+'-prediction.npz'), {'values_mv': prediction,
                    'fs': np.asarray(fs), 'lead_names': np.array(LEADS), 'finite': np.isfinite(prediction)})
                display = SimpleNamespace(leads={lead: SimpleNamespace(values=prediction[i], fs=fs, start_s=0.)
                                                for i, lead in enumerate(LEADS)})
                waveform_svg(directory/(name+'-waves.svg'), values, fs, case['metadata'], display)
                record['status'] = 'returned'
                if configured and (not np.isfinite(cost) or e['printed_time_coverage'] == 0
                        or row_count != identifier.layouts[layout]['total_rows']):
                    raise ValueError('Rejected fallback, empty signal or row/layout inconsistency')
                del result, canonical, tensor
            except (ValueError, RuntimeError, IndexError, KeyError, ZeroDivisionError) as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                record['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                torch.cuda.empty_cache()
            record['wall_seconds'] = time.perf_counter()-started
            (directory/(name+'-upstream.log')).write_text(log.getvalue())
            report['cases'].append(record)
            _write_json(out/'report.json', report)
            write_page(out, inputs, manifest, report)
            print(json.dumps({k: record.get(k) for k in ('name', 'arm', 'status', 'predicted_layout', 'wall_seconds')}), flush=True)
    report['status'] = 'completed'
    _write_json(out/'report.json', report)
    write_page(out, inputs, manifest, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--arms', nargs='+', choices=('published', 'configured'), default=['published', 'configured'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.inputs, args.out, args.source, device=args.device, arms=tuple(args.arms))
