"""Run the published npj Open-ECG-Digitizer on the synthetic layout probe.

External code/weights remain in an explicit checkout. No external code is
vendored into the package and no model download occurs during inference.
"""
import argparse
import base64
import contextlib
import hashlib
import html
import importlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
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
from examples.digitization_layout_probe import printed_windows, support_mask, waveform_svg


UPSTREAM_COMMIT = '97a15087d4abcda843da8c58ee74b1d8f47e6f9a'
WEIGHTS = {
    'unet_weights_07072025.pt': '17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b',
    'lead_name_unet_weights_07072025.pt': '840bd6bf2433ee6c22db67f57c861d9d427f29e10a32eeb334f0bcf061b175a2',
}
EXPECTED_LAYOUT = {'3x4+r1': 'standard_3x4_with_r1', '3x4': 'standard_3x4',
                   '6x2': 'standard_6x2', '12x1': 'standard_12x1',
                   '3x4-cabrera+r1': None}


def evaluate_missing(values, predicted_mv, fs, metadata):
    """Missing-aware errors plus paper-style +/-100 ms, DC-centered SNR.

    No filling or fabricated zero leads. Report coverage before alignment, and
    never average a missing lead into an apparent success. The aligned score
    is conditional on observed printed samples and is not the Kaggle metric.
    """
    a, b = np.asarray(values, dtype=np.float64), np.asarray(predicted_mv, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != 12 or np.isinf(b).any():
        raise ValueError('Expected twelve matching channels, finite values or NaN only')
    times = np.arange(a.shape[1])/fs
    windows = printed_windows(metadata)
    per_lead = {}
    total_truth = total_covered = finite_total = unsupported = 0
    squared_error = 0.
    for i, name in enumerate(LEADS):
        printed = support_mask(times, windows[name])
        finite = np.isfinite(b[i])
        keep = printed & finite
        truth = np.where(printed, a[i], np.nan)
        count, covered = int(printed.sum()), int(keep.sum())
        total_truth += count
        total_covered += covered
        finite_total += int(finite.sum())
        unsupported += int((finite & ~printed).sum())
        error = b[i, keep]-a[i, keep]
        squared_error += float(error@error)
        best = None
        radius = round(.1*fs)
        # Maximize centered SNR with the same shift/crop definition as the paper.
        for shift in range(-radius, radius+1):
            gt = truth[shift:] if shift > 0 else truth[:shift] if shift < 0 else truth
            pred = b[i, :-shift] if shift > 0 else b[i, -shift:] if shift < 0 else b[i]
            valid = np.isfinite(gt) & np.isfinite(pred)
            if int(valid.sum()) < max(3, round(.25*fs)):
                continue
            x, y = gt[valid].copy(), pred[valid].copy()
            x -= x.mean()
            y -= y.mean()
            power, noise = float(x@x), float((x-y)@(x-y))
            if power <= 0:
                continue
            snr = float(10*np.log10(max(power, 1e-30)/max(noise, 1e-30)))
            if best is None or snr > best['snr_db']:
                best = {'snr_db': snr, 'rmse_mv': float(np.sqrt(noise/len(x))),
                        'shift_ms': shift/fs*1000, 'samples': len(x)}
        per_lead[name] = {'printed_samples': count, 'covered_samples': covered,
            'coverage': covered/count if count else None,
            'unsupported_output_samples': int((finite & ~printed).sum()),
            'raw_rmse_mv': float(np.sqrt(error@error/len(error))) if len(error) else None,
            'raw_bias_mv': float(error.mean()) if len(error) else None,
            'aligned': best}
    aligned = [v['aligned']['snr_db'] for v in per_lead.values() if v['aligned'] is not None]
    return {'printed_time_coverage': total_covered/total_truth if total_truth else 0.,
        'unsupported_output_fraction': unsupported/finite_total if finite_total else None,
        'raw_overlap_rmse_mv': float(np.sqrt(squared_error/total_covered)) if total_covered else None,
        'mean_observed_lead_snr_db': float(np.mean(aligned)) if aligned else None,
        'leads_with_aligned_score': len(aligned), 'per_lead': per_lead}


def load_upstream(source):
    source = Path(source).resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    changed = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                      cwd=source, text=True).strip()
    if commit != UPSTREAM_COMMIT or changed:
        raise ValueError('An unmodified checkout of the reviewed upstream commit is required')
    for filename, digest in WEIGHTS.items():
        if sha256_file(source/'weights'/filename) != digest:
            raise ValueError(f'Upstream weight SHA-256 mismatch: {filename}')
    sys.path.insert(0, str(source))
    from src.config.default import get_cfg
    from src.model.inference_wrapper import InferenceWrapper

    cfg = get_cfg(str(source/'src/config/inference_wrapper_george-moody-2024.yml'))
    cfg.MODEL.KWARGS.device = 'cpu'
    settings = cfg.MODEL.KWARGS.config
    settings.LAYOUT_IDENTIFIER.KWARGS.device = 'cpu'
    settings.LAYOUT_IDENTIFIER.KWARGS.target_num_samples = 5000
    settings.LAYOUT_IDENTIFIER.config_path = str(source/'src/config/lead_layouts_all.yml')
    settings.LAYOUT_IDENTIFIER.unet_config_path = str(source/'src/config/lead_name_unet.yml')
    settings.LAYOUT_IDENTIFIER.unet_weight_path = str(source/'weights/lead_name_unet_weights_07072025.pt')
    settings.SEGMENTATION_MODEL.weight_path = str(source/'weights/unet_weights_07072025.pt')
    model = InferenceWrapper(**cfg.MODEL.KWARGS)
    # Upstream individually sets both neural networks to eval during loading.
    # Calling wrapper.eval() traverses its unused Dewarper, which lacks Module
    # initialization upstream. Preserve their published CLI execution path.
    if model.segmentation_model.training or model.identifier.unet.training:
        raise RuntimeError('Both pretrained networks must be in evaluation mode')
    return model, cfg, commit


def write_page(out, inputs, manifest, report):
    def embedded(path):
        mime = 'image/svg+xml' if path.suffix == '.svg' else 'image/png'
        return f'data:{mime};base64,'+base64.b64encode(path.read_bytes()).decode('ascii')

    page = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ECG · avaliação do modelo da npj Digital Medicine</title><style>
body{background:#0b1220;color:#e2e8f0;font:17px/1.6 system-ui;max-width:1240px;padding:30px;margin:auto}
article{background:#162235;padding:24px;border-radius:15px;margin:25px 0}h1{font-size:36px;line-height:1.2}
img{width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #334155}
a,summary{color:#7dd3fc}summary{cursor:pointer}.notice{border-left:4px solid #fbbf24;padding-left:18px}
</style><h1>Open-ECG-Digitizer nas imagens do nosso gerador</h1>
<p>Stenhede, Bjørnstad e Ranjbar, npj Digital Medicine (2026). Código e pesos publicados, sem treino ou ajuste nos exemplos.</p>
<p class="notice">Oito imagens, um sinal artificial. O modelo recebe apenas RGB e uma duração de saída de dez segundos. Catálogo de 13 layouts fornecido pelos autores. Geometria verdadeira usada somente para avaliar. CPU float32.</p>
<p>A cobertura conta apenas instantes impressos com saída finita. O SNR permite até 100 ms de deslocamento e remove DC, somente nos trechos observados. Inspecione também o RMSE bruto e os dados ausentes. O SNR não é o do leaderboard Kaggle.</p>
<p><a href="https://www.nature.com/articles/s41746-025-02327-1">Artigo</a> · <a href="https://github.com/Ahus-AIM/Open-ECG-Digitizer">Código original</a></p>
<table><tr><th>Caso</th><th>Layout escolhido</th><th>Cobertura</th><th>SNR observado</th><th>RMSE bruto</th></tr>'''
    for case in report['cases']:
        e = case.get('evaluation', {})
        coverage = f'{e["printed_time_coverage"]:.1%}' if e else 'Falhou'
        snr = f'{e["mean_observed_lead_snr_db"]:.2f} dB' if e.get('mean_observed_lead_snr_db') is not None else 'Ausente'
        rmse = f'{e["raw_overlap_rmse_mv"]:.3f} mV' if e.get('raw_overlap_rmse_mv') is not None else 'Ausente'
        page += f'<tr><td>{html.escape(case["name"])}</td><td>{html.escape(case.get("predicted_layout", ""))}</td><td>{coverage}</td><td>{snr}</td><td>{rmse}</td></tr>'
    page += '</table>'
    lookup = {case['name']: case for case in manifest['cases']}
    for case in report['cases']:
        name = case['name']
        page += f'<article><h2>{html.escape(name)} · {html.escape(lookup[name]["layout"])}</h2>'
        if case['status'] == 'failed':
            page += f'<p>{html.escape(case["error"])}</p>'
        else:
            page += f'<p>Layout corresponde ao esperado: {case["layout_matches_expected"]}. '
            page += f'Fallback de layout: {case["layout_fallback"]}. '
            page += f'Rótulos detectados: {case["detected_labels"]}. '
            page += f'Duração: {case["wall_seconds"]:.1f} s.</p>'
        page += f'<details><summary>Imagem de entrada</summary><img src="{embedded(inputs/lookup[name]["image"])}" alt="Entrada {name}"></details>'
        if (out/f'{name}-waves.svg').exists():
            page += f'<img src="{embedded(out/(name+"-waves.svg"))}" alt="Original verde e reconstrução laranja">'
        if (out/f'{name}-segmentation.png').exists():
            page += f'<details><summary>Segmentação e grade após alinhamento</summary><img src="{embedded(out/(name+"-segmentation.png"))}" alt="Segmentação"></details>'
        page += '</article>'
    (out/'index.html').write_text(page+'</html>')


def run(inputs, out, source):
    inputs, out, source = Path(inputs).resolve(), Path(out).resolve(), Path(source).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    manifest = json.loads((inputs/'manifest.json').read_text())
    if len(manifest['cases']) != 8 or sha256_file(inputs/'source.npz') != manifest['source_sha256']:
        raise ValueError('Incomplete or altered input probe')
    with np.load(inputs/'source.npz') as data:
        values, fs = data['values_mv'], int(data['fs'])
    if fs != 500 or values.shape != (12, 5000):
        raise ValueError('This recorded protocol uses ten seconds at 500 Hz')
    model, config, commit = load_upstream(source)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out/'effective-config.json', json.loads(json.dumps(dict(config), default=dict)))
    environment = {name: importlib.metadata.version(name) for name in
        ('torch', 'torchvision', 'numpy', 'scipy', 'scikit-image', 'scikit-learn',
         'yacs', 'torch-tps', 'ray', 'matplotlib')}
    report = {'schema': 'ecg-npj-synthetic-probe-v1', 'status': 'running', 'cases_total': 8,
        'paper_doi': '10.1038/s41746-025-02327-1', 'upstream_commit': commit,
        'upstream_repo': 'https://github.com/Ahus-AIM/Open-ECG-Digitizer',
        'upstream_license': 'CC-BY-SA-4.0', 'weight_sha256': WEIGHTS,
        'manifest_sha256': sha256_file(inputs/'manifest.json'),
        'device': 'cpu', 'precision': 'float32', 'threads': torch.get_num_threads(),
        'environment': environment, 'python': platform.python_version(),
        'catalog': list(model.identifier.layouts),
        'config_overrides': ['device cpu for wrapper and name detector',
            'published lead_layouts_all.yml instead of competition-only catalog',
            '5000 output samples for requested ten seconds at 500 Hz', 'absolute file paths'],
        'limits': ['One artificial signal, no model training or tuning',
            'Published grid search assumes 30-70 major lines per axis',
            'Published portrait-to-landscape rotation remains enabled',
            'Published gain conversion is fixed at 0.1 mV/mm',
            'Cabrera 3x4 + rhythm is absent from the published all-layout catalog',
            'Aligned SNR is conditional on finite printed samples, not a full-coverage score'], 'cases': []}
    last_match = {}
    original = model.identifier

    def capture_match(*args, **kwargs):
        output = original(*args, **kwargs)
        last_match.clear()
        last_match.update(output)
        return output

    model.identifier = capture_match
    for case in manifest['cases']:
        name = case['name']
        record = {'name': name, 'expected_layout': EXPECTED_LAYOUT[case['layout']],
                  'input_sha256': case['image_sha256'], 'source_layout': case['layout']}
        if sha256_file(inputs/case['image']) != case['image_sha256']:
            raise ValueError('Input image hash mismatch')
        print(json.dumps({'starting': name}), flush=True)
        started = time.perf_counter()
        logs = io.StringIO()
        last_match.clear()
        torch.manual_seed(20261021)
        try:
            with Image.open(inputs/case['image']) as image:
                array = np.array(image.convert('RGB'))
            image_tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())[None].float()
            with contextlib.redirect_stdout(logs), torch.inference_mode():
                output = model(image_tensor, layout_should_include_substring=None)
            canonical = output['signal']['canonical_lines']
            if canonical is None:
                raise ValueError('Upstream returned no canonical signals')
            predicted_mv = canonical.detach().cpu().numpy()/1000.
            evaluation = evaluate_missing(values, predicted_mv, fs, case['metadata'])
            cost = float(output['signal']['layout_matching_cost'])
            fallback = not np.isfinite(cost)
            record.update(status='returned', predicted_layout=output['layout_name'],
                layout_fallback=fallback, detected_labels=int(last_match.get('n_detected', 0)),
                extracted_rows=int(last_match.get('rows_in_layout', 0)),
                matching_cost=cost if np.isfinite(cost) else None,
                layout_matches_expected=not fallback and output['layout_name'] == record['expected_layout'],
                evaluation=evaluation, pixel_spacing_mm={k: float(v) for k, v in output['pixel_spacing_mm'].items()},
                stage_seconds=dict(model.times))
            _write_prediction(out/f'{name}-prediction.npz',
                {'values_mv': predicted_mv, 'fs': np.asarray(fs), 'finite': np.isfinite(predicted_mv),
                 'lead_names': np.array(LEADS)})
            display_result = SimpleNamespace(leads={lead: SimpleNamespace(values=predicted_mv[i], fs=fs, start_s=0.)
                                                   for i, lead in enumerate(LEADS)})
            waveform_svg(out/f'{name}-waves.svg', values, fs, case['metadata'], display_result)
            aligned = output['aligned']
            grid = aligned['grid_prob'].squeeze().numpy()
            signal = aligned['signal_prob'].squeeze().numpy()
            text = aligned['text_prob'].squeeze().numpy()
            rgb = np.stack([signal, grid, text], -1).clip(0, 1)
            Image.fromarray((rgb*255).round().astype(np.uint8)).save(out/f'{name}-segmentation.png')
            del output, canonical
        except (ValueError, RuntimeError, IndexError, ZeroDivisionError, KeyError) as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        record['wall_seconds'] = time.perf_counter()-started
        (out/f'{name}-upstream.log').write_text(logs.getvalue())
        report['cases'].append(record)
        _write_json(out/'report.json', report)
        write_page(out, inputs, manifest, report)
        print(json.dumps(record), flush=True)
    report['status'] = 'completed'
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _write_json(out/'report.json', report)
    write_page(out, inputs, manifest, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.inputs, args.out, args.source)
