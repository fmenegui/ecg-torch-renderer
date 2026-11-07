"""Exploratory geometry ablation with an EXISTING checkpoint, no new training.

All test cases remain in the denominator. Synthetic truth is used exclusively
after prediction. Output geometry residuals are conditional on supported nodes.
"""
import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.field_rectifier import GridFieldNet
from ecg_gpu.digitization.hybrid_rectifier import forward_supported, refine_lattice, similarity_residual
from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.runner import _write_json
from ecg_gpu.grid_evaluation import match_grid_points, summarize_grid_matches
from ecg_gpu.grid_point_learning import decode_grid_points
from examples.digitization_grid_rectification import grid_remap


def embedded(rgb):
    stream = io.BytesIO()
    display = Image.fromarray(rgb)
    display.thumbnail((1200, 900))
    display.save(stream, format='JPEG', quality=90)
    return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()


def detect(model, rgb, threshold, device):
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().to(device)/255
    with torch.inference_mode():
        output = model(tensor)
        prediction = decode_grid_points(output['heat'].sigmoid(), output['offsets'], threshold=threshold,
                                        max_points=rgb.shape[0]*rgb.shape[1])
        keep = prediction['valid'][0]
        return prediction['xy'][0, keep].cpu().numpy(), prediction['scores'][0, keep].cpu().numpy()


def main(root, checkpoint, out, device='cpu', limit=0):
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    model = GridFieldNet(contrast=state['variant'] != 'raw-point').to(device).eval()
    model.load_state_dict(state['state_dict'])
    manifest = [r for r in json.loads((root/'capture-manifest.json').read_text()) if r['split'] == 'test']
    if limit:
        manifest = manifest[:limit]
    report = {'status': 'exploratory-old-checkpoint-no-training', 'checkpoint_sha256': sha256_file(checkpoint),
              'device': device, 'cases': [], 'source_study': 'consolidated-gpu',
              'limits': ['Not the new larger study', 'Previously evaluated synthetic test cohort',
                         'Smoothing fixed a priori at 0.05, not selected or claimed superior',
                         'Residuals conditional on supported true grid nodes',
                         'No Kaggle or npj comparison in this preview']}
    articles = []
    for row in manifest:
        path = root/row['file']
        if sha256_file(path) != row['sha256']:
            raise ValueError('Capture checksum differs')
        with np.load(path) as data:
            rgb, truth, truth_ids = data['image'].copy(), data['xy'].copy(), data['ids'].copy()
        xy, scores = detect(model, rgb, float(state['threshold']), device)
        match = match_grid_points(xy, truth, radius=1.5)
        topology = organize_lattice(xy, scores)
        case = {'key': row['key'], 'group_id': row['group_id'], 'backend': row['backend'], 'layout': row['layout'],
                'points': summarize_grid_matches([match]), 'arms': {}}
        panels = []
        overlay = Image.fromarray(rgb.copy())
        draw = ImageDraw.Draw(overlay)
        matched = set(match['prediction_indices'].tolist())
        missed = set(range(len(truth)))-set(match['reference_indices'].tolist())
        for i, (x, y) in enumerate(xy):
            color = '#16a34a' if i in matched else '#ef4444'
            draw.ellipse((x-1.5, y-1.5, x+1.5, y+1.5), outline=color, width=1)
        for i in missed:
            x, y = truth[i]
            draw.rectangle((x-2, y-2, x+2, y+2), outline='#f59e0b', width=1)
        panels.extend([('Entrada', rgb), ('Pontos: verde correto, vermelho extra, laranja ausente', np.asarray(overlay))])
        for arm, strength in (('point-only', 0.), ('hybrid-smooth', .05)):
            if topology['status'] != 'accepted' or len(topology['cells']) < 16:
                case['arms'][arm] = {'status': 'rejected', 'point_support_fraction': 0.}
                continue
            fitted, checked = refine_lattice(xy, scores, topology, smoothness=strength)
            ids = np.asarray(checked['ids'])
            canonical_mm = ids[:, ::-1]*5.
            predicted = forward_supported(fitted, canonical_mm, ids, truth, checked['cells'])
            metrics = similarity_residual(predicted, truth_ids[:, ::-1]*5.)
            metrics.update(status=checked['status'], cells=len(checked['cells']),
                           smoothing_displacement_p95_px=checked['smoothing_displacement_p95_px'])
            case['arms'][arm] = metrics
            canonical = canonical_mm*4
            size = (canonical.max(0)+1).astype(int)[::-1]
            if max(size) > 4096:
                continue
            warped, support = grid_remap(rgb, fitted, canonical, ids, tuple(size), allowed_cells=checked['cells'])
            Image.fromarray(warped).save(out/(row['key']+'-'+arm+'.png'))
            np.savez_compressed(out/(row['key']+'-'+arm+'-geometry.npz'), observed=fitted,
                                canonical_mm=canonical_mm, ids=ids, cells=np.asarray(checked['cells']), support=support)
            panels.append((arm, warped))
        np.savez_compressed(out/(row['key']+'-points.npz'), xy=xy, scores=scores)
        report['cases'].append(case)
        _write_json(out/'preview.json', report)
        # Every evaluated case is included. No silent gallery selection.
        article = '<article><h2>'+html.escape(row['key']+' · '+row['layout'])+'</h2><div class="panels">'
        for title, image in panels:
            article += '<figure><figcaption>'+html.escape(title)+'</figcaption><img src="'+embedded(image)+'"></figure>'
        article += '</div><details><summary>Métricas geométricas</summary><pre>'+html.escape(json.dumps(case, indent=2))+'</pre></details></article>'
        articles.append(article)
        print(json.dumps({'completed': len(report['cases']), 'total': len(manifest), 'key': row['key']}), flush=True)
    page = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ECG · prévia geométrica</title><style>body{font:16px/1.5 system-ui;background:#101827;color:#e2e8f0;margin:24px auto;padding:20px;max-width:1500px}
h1{line-height:1.15}article{background:#1d293b;padding:18px;margin:24px 0;border-radius:12px}.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
figure{margin:0}img{width:100%;cursor:zoom-in}img:active{position:fixed;inset:2%;width:96%;height:96%;object-fit:contain;background:#fff;z-index:3}figcaption{padding:8px 0}pre{white-space:pre-wrap}
@media(max-width:700px){.panels{grid-template-columns:1fr}}</style>
<h1>Retificação de ECG: pontos e geometria</h1>
<p><strong>Prévia com o modelo anterior. Nenhum novo treinamento. Execução em CPU.</strong></p>
<p>Todos os casos do teste sintético anterior aparecem abaixo. Verde indica detecção a até 1,5 pixel de uma referência, vermelho indica ponto extra, laranja indica referência não detectada. Pressione uma imagem para ampliar.</p>
<p>O ajuste híbrido usa somente os pontos detectados. A verdade geométrica entra depois, exclusivamente na avaliação. A área branca representa ausência de suporte. Esta prévia não estabelece superioridade sobre Kaggle ou npj.</p>'''
    page += '<p>Imagens de exibição comprimidas em JPEG. As métricas usam os dados originais, sem essa compressão.</p>'
    page += ''.join(articles)+'</html>'
    (out/'galeria-previa.html').write_text(page)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'checkpoint', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--limit', type=int, default=0)
    torch.set_num_threads(4)
    main(**vars(parser.parse_args()))
