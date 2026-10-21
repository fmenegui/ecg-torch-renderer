"""Summarize a completed pilot with paired patient bootstrap and actual crops."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def summarize(report_path, out, *, captures=None, seed=71, repeats=2000):
    report_path, out = Path(report_path), Path(out)
    report = json.loads(report_path.read_text())
    if report.get('status') != 'completed':
        raise ValueError('Use a completed report')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty summary directory')
    baselines = [m for m in report['models'] if m['name'].startswith('baseline-')]
    groups = sorted({r['group_id'] for r in baselines[0]['per_image']})
    if len(baselines) != 3 or not groups:
        raise ValueError('Three baseline models and nonempty test groups required')
    rng = np.random.default_rng(seed)
    resampled = rng.integers(len(groups), size=(repeats, len(groups)))
    output = {'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'bootstrap': {'unit': 'patient group, keeping all views together', 'repeats': repeats, 'seed': seed,
                      'test_groups': len(groups), 'interval': 'percentile 95%',
                      'limitation': 'Exploratory sampling uncertainty only, not training-seed or domain uncertainty'},
        'models': {}, 'paired_differences': {}}
    draws = {}
    for model in baselines:
        name = model['name']
        output['models'][name] = {}
        for backend, metrics in model['by_backend'].items():
            rows = [r for r in model['per_image'] if r['backend'] == backend]
            if any(r['truncated'] for r in rows) or {r['group_id'] for r in rows} != set(groups):
                raise ValueError('Truncated or mismatched test partitions cannot be compared')
            counts = np.array([[sum(r[k] for r in rows if r['group_id'] == group)
                                for k in ('tp', 'fp', 'fn')] for group in groups])
            sampled = counts[resampled].sum(1)
            denominator = 2*sampled[:, 0]+sampled[:, 1:].sum(-1)
            scores = np.divide(2*sampled[:, 0], denominator, out=np.zeros(repeats), where=denominator > 0)
            draws[name, backend] = scores
            output['models'][name][backend] = {**metrics, 'f1_patient_bootstrap_95': np.quantile(scores, [.025, .975]).tolist()}
    for backend in ('online', 'offline'):
        delta = draws['baseline-mixed', backend]-draws['baseline-online', backend]
        output['paired_differences'][backend] = {'comparison': 'mixed minus online F1',
            'bootstrap_95': np.quantile(delta, [.025, .975]).tolist()}
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    font = ImageFont.truetype('DejaVuSans.ttf', 24)
    small = ImageFont.truetype('DejaVuSans.ttf', 17)
    chart = Image.new('RGB', (1080, 540), '#f2f5f7')
    draw = ImageDraw.Draw(chart)
    draw.text((26, 20), 'Localizacao da grade: transferencia entre dominios', font=font, fill='#172937')
    draw.text((26, 57), 'F1 de pontos a 1,5 px | barras = estimativa | linhas = bootstrap por paciente', font=small, fill='#43566a')
    for i, model in enumerate(baselines):
        name = model['name']
        draw.text((26, 115+i*112), name.replace('baseline-', 'Treino '), font=font, fill='#172937')
        for j, backend in enumerate(('online', 'offline')):
            result = output['models'][name][backend]
            y = 113+i*112+j*37
            x, width = 350, 570
            value = result['f1'] or 0.
            color = '#007e9a' if j == 0 else '#be4a36'
            draw.line((x, y+12, x+width, y+12), fill='#dbe3e8', width=16)
            draw.line((x, y+12, x+width*value, y+12), fill=color, width=16)
            lower, upper = result['f1_patient_bootstrap_95']
            draw.line((x+width*lower, y+12, x+width*upper, y+12), fill='#142d3a', width=2)
            for bound in (lower, upper):
                draw.line((x+width*bound, y+5, x+width*bound, y+19), fill='#142d3a', width=2)
            draw.text((950, y), f'{value:.3f}', font=small, fill='#172937')
    draw.text((28, 467), 'Azul: teste torch    Vermelho: teste ray tracing    Mesma rede e orcamento de treino.', font=small, fill='#43566a')
    draw.text((28, 500), f'{len(groups)} pacientes de teste. Uma semente de treino. Piloto exploratorio, sem ranking definitivo.', font=small, fill='#43566a')
    chart.save(out/'comparacao-regimes.png')
    if captures:
        panel = Image.new('RGB', (1200, 720), '#f2f5f7')
        draw = ImageDraw.Draw(panel)
        draw.text((18, 12), 'Recortes reais das imagens sinteticas avaliadas', font=font, fill='#172937')
        draw.text((18, 46), 'Ciano: referencia geometrica | vermelho: previsao da rede | mesmo recorte por dominio', font=small, fill='#43566a')
        for column, model in enumerate(baselines):
            draw.text((column*400+18, 82), model['name'].replace('baseline-', 'Treino '), font=small, fill='#172937')
            for row, backend in enumerate(('online', 'offline')):
                record = next(r for r in model['per_image'] if r['backend'] == backend and '-0-' in r['key'])
                sample = torch.load(Path(captures)/(record['key']+'.pt'), weights_only=True)
                predicted = np.load(report_path.parent/(model['name']+'-'+record['key']+'-prediction.npz'))['xy']
                image = Image.fromarray((sample['image'][0].permute(1, 2, 0).numpy().clip(0, 1)*255).round().astype('uint8'))
                x0, y0 = image.width//3, image.height//3
                crop = image.crop((x0, y0, x0+140, y0+100)).resize((392, 280), Image.Resampling.LANCZOS)
                overlay = ImageDraw.Draw(crop)
                for reference, color in ((sample['xy'][0].numpy(), '#00d4dc'), (predicted, '#ff3356')):
                    for x, y in reference:
                        if x0 <= x < x0+140 and y0 <= y < y0+100:
                            cx, cy = (x-x0+.5)*2.8-.5, (y-y0+.5)*2.8-.5
                            if color == '#00d4dc':
                                overlay.ellipse((cx-2.4, cy-2.4, cx+2.4, cy+2.4), outline=color, width=1)
                            else:
                                overlay.line((cx-3, cy, cx+3, cy), fill=color, width=1)
                                overlay.line((cx, cy-3, cx, cy+3), fill=color, width=1)
                panel.paste(crop, (column*400+4, 120+row*292))
        panel.save(out/'pontos-previstos-recortes.png')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--captures', type=Path)
    args = parser.parse_args()
    summarize(args.report, args.out, captures=args.captures)
