"""Probe the frozen Kaggle digitizer on declared synthetic paper configurations.

The predictor sees RGB only. Renderer metadata is used exclusively for scoring
and display. Outputs with a wrong layout remain visible and are never counted
as successful generalization merely because inference returned twelve arrays.
"""
import argparse
import base64
from dataclasses import asdict
import html
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu import ECGImageLayer, capture_scenario
from ecg_gpu.demo import demo_signals
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.runner import _write_json, _write_prediction
from ecg_gpu.digitization.signals import LEADS
from ecg_gpu.gpu_cli import plain


CASES = (
    ('standard', '3x4+r1', 'clean_print', 2048, 1280, 1, 1),
    ('without-rhythm', '3x4', 'clean_print', 2048, 1280, 1, 1),
    ('six-rows', '6x2', 'clean_print', 2048, 1280, 1, 1),
    ('portrait', '12x1', 'clean_print', 1280, 2048, 1, 1),
    ('cabrera', '3x4-cabrera+r1', 'clean_print', 2048, 1280, 1, 1),
    ('gain-5', '3x4+r1', 'clean_print', 2048, 1280, 1, 0),
    ('speed-50', '3x4+r1', 'clean_print', 2048, 1280, 2, 1),
    ('phone', '3x4+r1', 'phone_photo', 2048, 1280, 1, 1),
)


def printed_windows(metadata):
    """Named, positive-polarity time support, merging repeated rhythm strips."""
    windows = {name: [] for name in LEADS}
    for i, active in enumerate(metadata['active'][0]):
        if active:
            name = LEADS[int(metadata['lead_id'][0][i])]
            # Source references have standard polarity, including printed -aVR.
            windows[name].append((float(metadata['start_s'][0][i]),
                                  float(metadata['end_s'][0][i])))
    return windows


def support_mask(times, windows):
    mask = np.zeros(len(times), dtype=bool)
    for start, end in windows:
        mask |= (times >= start-1e-8) & (times < end-1e-8)
    return mask


def evaluate_overlap(values, fs, metadata, prediction):
    """Compare only physically printed times and report their coverage.

    No time alignment or DC removal. Coverage counts unique lead/time samples,
    so repeated II does not artificially inflate the denominator. A prediction
    of an unprinted rhythm strip is explicitly reported as unsupported output.
    """
    windows = printed_windows(metadata)
    source_t = np.arange(values.shape[1])/fs
    per_lead = {}
    total_truth = total_covered = total_output = total_unsupported = 0
    signal_power = error_power = 0.
    for i, name in enumerate(LEADS):
        lead = prediction[name]
        pred_t = lead.start_s+np.arange(len(lead.values))/lead.fs
        truth_mask = support_mask(source_t, windows[name])
        pred_mask = support_mask(pred_t, windows[name]) & (pred_t <= source_t[-1]+1e-8)
        covered = truth_mask & (source_t >= lead.start_s-1e-8)
        covered &= source_t < lead.start_s+len(lead.values)/lead.fs-1e-8
        truth = np.interp(pred_t[pred_mask], source_t, values[i])
        error = lead.values[pred_mask]-truth
        sp, ep = float(truth@truth), float(error@error)
        signal_power += sp
        error_power += ep
        count_truth, count_covered = int(truth_mask.sum()), int(covered.sum())
        unsupported = int((~pred_mask).sum())
        total_truth += count_truth
        total_covered += count_covered
        total_output += len(lead.values)
        total_unsupported += unsupported
        per_lead[name] = {
            'printed_samples': count_truth, 'covered_samples': count_covered,
            'unsupported_output_samples': unsupported,
            'raw_rmse_mv': float(np.sqrt(ep/len(error))) if len(error) else None,
            'raw_bias_mv': float(error.mean()) if len(error) else None,
            'printed_windows_s': windows[name], 'prediction_start_s': lead.start_s,
            'prediction_duration_s': len(lead.values)/lead.fs,
        }
    return {'printed_time_coverage': total_covered/total_truth if total_truth else 0.,
            'unsupported_output_fraction': total_unsupported/total_output,
            'raw_overlap_snr_db': float(10*np.log10(signal_power/max(error_power, 1e-30)))
                if signal_power > 0 else None,
            'raw_overlap_rmse_mv': float(np.sqrt(error_power/(total_output-total_unsupported)))
                if total_output > total_unsupported else None,
            'per_lead': per_lead}


def prepare(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    values, fs, _ = demo_signals()
    # A shared nonstationary envelope preserves all limb identities and makes
    # wrong temporal windows less likely to hide behind perfectly repeated beats.
    t = np.arange(values.shape[1])/fs
    values = (values*(.8+.18*np.sin(.63*t)+.02*t)).astype(np.float32)
    _write_prediction(out/'source.npz', {'values_mv': values, 'fs': np.asarray(fs)})
    manifest = {'schema': 'ecg-synthetic-layout-probe-v1', 'signal': 'artificial demo with shared envelope',
                'source_sha256': sha256_file(out/'source.npz'), 'cases': []}
    for name, layout, scenario, width, height, speed, gain in CASES:
        config, parameters = capture_scenario(scenario, width=width, height=height, severity=.35)
        parameters.update(layout=layout, speed_index=speed, gain_index=gain,
                          font=0, font_size=20., grid_phase_x=0., grid_phase_y=0.)
        layer = ECGImageLayer(fs=fs, config=config, parameters=parameters, seed=20261021,
                              randomize=False, photo=scenario != 'clean_print').eval()
        with torch.inference_mode():
            batch = layer.render_batch(torch.from_numpy(values)[None], stage='image')
        rgb = (batch.image[0].permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(out/f'{name}.png')
        manifest['cases'].append({'name': name, 'layout': layout, 'scenario': scenario,
            'width': width, 'height': height, 'config': asdict(config), 'seed': 20261021,
            'image': f'{name}.png', 'image_sha256': sha256_file(out/f'{name}.png'),
            'metadata': plain(batch.metadata)})
        _write_json(out/'manifest.json', manifest)
        print(json.dumps({'prepared': name, 'layout': layout}), flush=True)
        del layer, batch
    return manifest


def waveform_svg(path, values, fs, metadata, result):
    """Independent unaligned overlay. Dashed intervals were not printed."""
    windows = printed_windows(metadata)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 1100">',
             '<rect width="1100" height="1100" fill="#121b29"/>',
             '<text x="20" y="24" fill="#e2e8f0" font-size="17">Original: verde | Saída: laranja | mV, sem alinhamento</text>']
    for i, name in enumerate(LEADS):
        top, left, w = 42+i*87, 65, 1000
        row = values[i]
        scale = 25/max(1., float(np.abs(row).max()))
        parts.append(f'<text x="15" y="{top+34}" fill="white" font-size="15">{name}</text>')
        for start, end in windows[name]:
            a, b = left+w*start/10, left+w*end/10
            parts.append(f'<rect x="{a:.2f}" y="{top}" width="{b-a:.2f}" height="70" fill="#203442"/>')
        times = np.arange(len(row))/fs
        keep = support_mask(times, windows[name])
        # Each printed window is drawn separately so absent intervals stay empty.
        for start, end in windows[name]:
            ids = np.flatnonzero(keep & (times >= start) & (times < end))[::3]
            points = ' '.join(f'{left+w*times[j]/10:.2f},{top+40-scale*row[j]:.2f}' for j in ids)
            parts.append(f'<polyline points="{points}" fill="none" stroke="#6ee7b7" stroke-width="1.2"/>')
        if result is not None:
            pred = result.leads[name]
            times = pred.start_s+np.arange(len(pred.values))/pred.fs
            supported = support_mask(times, windows[name])
            for good, dash in ((True, ''), (False, ' stroke-dasharray="4 4"')):
                ids = np.flatnonzero((supported == good) & np.isfinite(pred.values))[::3]
                # Avoid long connections across disjoint printed windows.
                for chunk in np.split(ids, np.flatnonzero(np.diff(ids) > 4)+1):
                    points = ' '.join(f'{left+w*times[j]/10:.2f},{np.clip(top+40-scale*pred.values[j],top,top+76):.2f}' for j in chunk)
                    parts.append(f'<polyline points="{points}" fill="none" stroke="#fb923c" stroke-width="1"{dash}/>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts))


def write_page(out, report):
    def embed(path):
        mime = 'image/svg+xml' if path.suffix == '.svg' else 'image/png'
        return f'data:{mime};base64,'+base64.b64encode(path.read_bytes()).decode('ascii')

    cards = []
    for case in report['cases']:
        name = html.escape(case['name'])
        evaluation = case.get('evaluation')
        status = 'Erro explícito: '+html.escape(case.get('error', ''))
        if evaluation:
            status = (f"Inferência concluída. Cobertura temporal: {evaluation['printed_time_coverage']:.1%}. "
                      f"Saída sem suporte impresso: {evaluation['unsupported_output_fraction']:.1%}. "
                      f"RMSE na sobreposição: {evaluation['raw_overlap_rmse_mv']:.3f} mV. "
                      f"SNR bruto na sobreposição: {evaluation['raw_overlap_snr_db']:.2f} dB.")
        rect = f'<a href="{name}-rectified.png">Ver imagem retificada</a>' if case['status'] == 'returned' else ''
        cards.append(f'<article><h2>{name} · {html.escape(case["layout"])}</h2><p>{status}</p>'
                     f'<img src="{embed(out / (name+".png"))}" alt="ECG sintético {name}">'
                     f'<p>{rect}</p><details><summary>Ver sinais e intervalos impressos</summary>'
                     f'<img src="{embed(out / (name+"-waves.svg"))}" alt="Comparação sem alinhamento"></details></article>')
    page = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Digitalização de ECG · teste de layouts sintéticos</title><style>
body{background:#0b1220;color:#e2e8f0;font:17px/1.6 system-ui;margin:0 auto;max-width:1240px;padding:30px}
h1{font-size:36px;line-height:1.2}h2{font-size:25px}a{color:#7dd3fc}article{background:#162235;padding:24px;margin:30px 0;border-radius:15px}
img{display:block;width:100%;height:auto;border-radius:8px}p{max-width:1000px}.notice{border-left:4px solid #fbbf24;padding-left:18px}
table{border-collapse:collapse;width:100%;margin:24px 0}th,td{text-align:left;padding:10px;border-bottom:1px solid #334155}summary{cursor:pointer;color:#7dd3fc}
</style><h1>O mesmo sinal em oito configurações de papel</h1>
<p>Modelo Heng + Tang congelado, perfil single. O modelo recebe apenas RGB. Sinal e geometria exatos do gerador são usados somente na avaliação.</p>
<p class="notice">Teste exploratório com um sinal artificial e uma captura por configuração. Inferência concluída não significa layout reconhecido. Os pesos pressupõem 3×4 + II longa. Não é validação clínica nem comparação com o leaderboard.</p>
<p>A cobertura mede a fração dos instantes impressos por derivação representada na saída. O erro usa somente a sobreposição temporal, sem corrigir atraso ou nível DC. Trechos de saída que não estavam impressos aparecem tracejados. Curvas fora da faixa vertical são limitadas apenas na visualização.</p>
<p><a href="report.json">Resultados completos</a> · <a href="manifest.json">Configurações e hashes</a></p>'''
    failed = sum(case['status'] == 'failed' for case in report['cases'])
    page += f'<h2>Teste automático: {failed} falhas em {len(report["cases"])} imagens</h2>'
    page += '<table><tr><th>Configuração</th><th>Layout</th><th>Última rede executada</th><th>Resultado</th></tr>'
    for case in report['cases']:
        stage = case['last_network_stage']
        page += (f'<tr><td>{html.escape(case["name"])}</td><td>{html.escape(case["layout"])}</td>'
                 f'<td>Stage {stage}</td><td>{"Falhou" if case["status"] == "failed" else "Retornou saída"}</td></tr>')
    page += '</table><p>As falhas desta execução estão na localização inicial das referências impressas. A geração de uma saída automática utilizável não foi demonstrada nestes oito exemplos.</p>'
    oracle_path = out/'oracle-report.json'
    controls = ''
    if oracle_path.exists():
        oracle = json.loads(oracle_path.read_text())
        controls = '<h2>Controle com geometria fornecida pelo gerador</h2><p class="notice">Este controle usa posições e escalas verdadeiras. Ele testa somente a rede de extração da onda em imagens normalizadas. Não é digitalização automática nem detecção de layout.</p>'
        for case in oracle['cases']:
            name = case['name']
            controls += f'<article><h2>{html.escape(name)}</h2><p>'
            if case['status'] == 'returned':
                controls += (f'SNR bruto: {case["evaluation"]["raw_overlap_snr_db"]:.2f} dB. '
                             f'RMSE: {case["evaluation"]["raw_overlap_rmse_mv"]:.3f} mV. '
                             f'SNR oficial alinhado: {case["official_evaluation"]["official_snr_db"]:.2f} dB.')
                controls += f'</p><details><summary>Inspecionar curvas sem alinhamento</summary><img src="{embed(out / (name+"-oracle-waves.svg"))}" alt="Ondas no controle com geometria conhecida"></details>'
            else:
                controls += html.escape(case['error'])+'</p>'
            controls += '</article>'
    (out/'index.html').write_text(page+'\n'.join(cards)+controls+'</html>')


def normalize_known_rows(rgb, metadata, cv2):
    """Oracle control for clean standard 3x4 + II, never an RGB-only detector.

    Maps each printed row with known source baseline/time/gain into the frozen
    stage2 canvas. RGB is resampled, no source waveform is used to create pixels.
    This deliberately skips both learned geometry stages. Photo geometry is not
    supported here because canonical metadata is not observed-image geometry.
    """
    baseline = np.asarray(metadata['baseline'][0])
    row_ids = np.asarray(metadata['row'][0])
    active = np.asarray(metadata['active'][0], dtype=bool)
    x0 = np.asarray(metadata['x0'][0])
    x1 = np.asarray(metadata['x1'][0])
    output = np.full((1700, 4400, 3), 255, dtype=np.uint8)
    boundaries = (0, 846, 1130, 1402, 1700)
    zero_mv = (703.5, 987.5, 1271.5, 1531.5)
    sy = 79*float(metadata['canonical_millivolts_per_pixel'][0])
    for row in range(4):
        ids = np.flatnonzero(active & (row_ids == row))
        if len(ids) != (4 if row < 3 else 1):
            raise ValueError('Oracle control requires standard 3x4 + one rhythm row')
        sx = 3924/(float(x1[ids].max())-float(x0[ids].min()))
        transform = np.array([[sx, 0, 236-sx*float(x0[ids].min())],
                              [0, sy, zero_mv[row]-sy*float(baseline[ids[0]])]])
        warp = cv2.warpAffine(rgb, transform, (4400, 1700),
                              flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        start, end = boundaries[row:row+2]
        output[start:end] = warp[start:end]
    return output


def run_oracle_control(out, assets, precision):
    from ecg_gpu.digitization.data import sequential_reference
    from ecg_gpu.digitization.evaluation import OfficialMetric
    from ecg_gpu.digitization.reference import HengTangDigitizer
    from ecg_gpu.digitization.signals import DigitizationResult, LeadSignal, split_rows

    out = Path(out)
    if (out/'oracle-report.json').exists():
        raise FileExistsError('An oracle control report already exists')
    manifest = json.loads((out/'manifest.json').read_text())
    if [case['name'] for case in manifest['cases']] != [case[0] for case in CASES]:
        raise ValueError('The preparation phase did not finish all declared cases')
    if sha256_file(out/'source.npz') != manifest['source_sha256']:
        raise ValueError('Source signal hash mismatch')
    with np.load(out/'source.npz') as data:
        values, fs = data['values_mv'], int(data['fs'])
    model = HengTangDigitizer(assets, precision=precision)
    metric = OfficialMetric(assets)
    truth = sequential_reference({name: LeadSignal(values[i], fs) for i, name in enumerate(LEADS)})
    report = {'status': 'running', 'uses_ground_truth_geometry': True,
        'not_automatic_digitization': True, 'precision': precision, 'device': 'cpu',
        'predeclared_cases': ['standard', 'gain-5', 'speed-50'],
        'manifest_sha256': sha256_file(out/'manifest.json'), 'cases': []}
    for case in manifest['cases']:
        if case['name'] not in report['predeclared_cases']:
            continue
        name = case['name']
        if case['scenario'] != 'clean_print' or case['layout'] != '3x4+r1':
            raise ValueError('Control requires undeformed standard paper')
        print(json.dumps({'oracle_starting': name}), flush=True)
        record = {'name': name, 'layout': case['layout'], 'image_sha256': case['image_sha256'],
                  'uses_ground_truth_geometry': True}
        started = time.perf_counter()
        try:
            if sha256_file(out/case['image']) != case['image_sha256']:
                raise ValueError('Input image hash mismatch')
            with Image.open(out/case['image']) as image:
                rect = normalize_known_rows(np.asarray(image.convert('RGB')), case['metadata'], model.cv2)
            Image.fromarray(rect).save(out/f'{name}-oracle-normalized.png')
            image2 = torch.from_numpy(np.ascontiguousarray(rect[:1696, :4352].transpose(2, 0, 1)))[None]
            with torch.inference_mode():
                rows, observed, count = model._waveforms(image2, fs, None)
            result = DigitizationResult(split_rows(rows, fs))
            record.update(status='returned', checkpoint_count=count,
                observed_fraction=observed.mean(axis=1).tolist(),
                evaluation=evaluate_overlap(values, fs, case['metadata'], result.leads),
                official_evaluation=metric.image(truth, result.leads))
            waveform_svg(out/f'{name}-oracle-waves.svg', values, fs, case['metadata'], result)
            _write_prediction(out/f'{name}-oracle-prediction.npz',
                              {lead: signal.values for lead, signal in result.leads.items()})
        except (ValueError, RuntimeError, model.cv2.error) as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        record['wall_seconds'] = time.perf_counter()-started
        report['cases'].append(record)
        _write_json(out/'oracle-report.json', report)
        print(json.dumps(record), flush=True)
    report['status'] = 'completed'
    _write_json(out/'oracle-report.json', report)
    write_page(out, json.loads((out/'report.json').read_text()))
    return report


def run(out, assets, precision):
    from ecg_gpu.digitization.reference import HengTangDigitizer
    out = Path(out)
    if (out/'report.json').exists():
        raise FileExistsError('A probe report already exists, choose another directory')
    manifest = json.loads((out/'manifest.json').read_text())
    if [case['name'] for case in manifest['cases']] != [case[0] for case in CASES]:
        raise ValueError('The preparation phase did not finish all declared cases')
    if sha256_file(out/'source.npz') != manifest['source_sha256']:
        raise ValueError('Source signal hash mismatch')
    with np.load(out/'source.npz') as data:
        values, fs = data['values_mv'], int(data['fs'])
    model = HengTangDigitizer(assets, precision=precision)
    report = {'schema': manifest['schema'], 'status': 'running', 'cases_total': len(manifest['cases']),
        'manifest_sha256': sha256_file(out/'manifest.json'), 'precision': precision, 'device': 'cpu',
        'torch': str(torch.__version__), 'python': platform.python_version(),
        'cpu_threads': torch.get_num_threads(), 'model_artifacts': model.artifacts,
        'limits': ['One artificial signal, one fixed seed per configuration',
                   'No parameter tuning or model training on these examples',
                   'Fixed Kaggle layout remains an assumption even if prediction returns arrays',
                   'Overlap SNR is raw, not the official aligned competition metric'], 'cases': []}
    last_stage = [None]
    hooks = [getattr(model, f'net{i}').register_forward_pre_hook(
        lambda module, args, i=i: last_stage.__setitem__(0, i)) for i in range(3)]
    try:
        for case in manifest['cases']:
            record = {key: case[key] for key in ('name', 'layout', 'scenario', 'image_sha256')}
            name = case['name']
            print(json.dumps({'starting': name}), flush=True)
            if sha256_file(out/case['image']) != case['image_sha256']:
                raise ValueError('Input image hash mismatch')
            started = time.perf_counter()
            last_stage[0] = None
            result = None
            try:
                with Image.open(out/case['image']) as image:
                    rgb = np.asarray(image.convert('RGB'))
                result, debug = model.predict(rgb, fs=fs)
                record.update(status='returned', diagnostics=result.diagnostics, warnings=result.warnings,
                              evaluation=evaluate_overlap(values, fs, case['metadata'], result.leads))
                Image.fromarray(debug['rectified']).save(out/f'{name}-rectified.png')
                payload = {lead: signal.values for lead, signal in result.leads.items()}
                payload.update(start_s=np.array([result.leads[lead].start_s for lead in LEADS]),
                               fs=np.asarray(fs), lead_names=np.array(LEADS))
                _write_prediction(out/f'{name}-prediction.npz', payload)
            except (ValueError, RuntimeError, model.cv2.error) as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            record.update(wall_seconds=time.perf_counter()-started, last_network_stage=last_stage[0])
            waveform_svg(out/f'{name}-waves.svg', values, fs, case['metadata'], result)
            report['cases'].append(record)
            _write_json(out/'report.json', report)
            write_page(out, report)
            print(json.dumps(record), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    report['status'] = 'completed'
    _write_json(out/'report.json', report)
    write_page(out, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=Path('build/reference-assets'))
    parser.add_argument('--phase', choices=('prepare', 'run', 'oracle', 'all'), default='all')
    parser.add_argument('--precision', choices=('float32', 'bfloat16'), default='float32')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.phase in ('prepare', 'all'):
        prepare(args.out)
    if args.phase in ('run', 'all'):
        run(args.out, args.assets, args.precision)
    if args.phase == 'oracle':
        run_oracle_control(args.out, args.assets, args.precision)
