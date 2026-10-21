"""Build a source-image-free summary of completed digitization evaluations."""

import argparse
import csv
import html
import json
from pathlib import Path
import statistics


def summarize(reports, out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty summary directory')
    documents = [json.loads(Path(p).read_text()) for p in reports]
    if any(d.get('status') not in ('completed', 'completed_with_failures') for d in documents):
        raise ValueError('Refusing to summarize an interrupted or incomplete evaluation')
    rows, groups = [], []
    for doc in documents:
        profile = next((c['provenance']['profile'] for c in doc['cases'] if c['status'] == 'success'), 'failed')
        geometry = next((c['provenance']['geometry'] for c in doc['cases'] if c['status'] == 'success'), 'unknown')
        if geometry != 'reference':
            profile += ' / '+geometry
        score = doc.get('official_snr_db')
        individual = [c['evaluation']['official_snr_db'] for c in doc['cases'] if 'evaluation' in c]
        groups.append({'profile': profile, 'geometry': geometry, 'images': doc['cases_total'], 'patients': doc['patients'],
            'successes': doc['successes'], 'failures': doc['failures'], 'official_snr_db': score,
            'minimum_image_snr_db': min(individual) if individual else None,
            'median_wall_seconds': statistics.median(c['wall_seconds'] for c in doc['cases']),
            'by_image_type': doc['by_image_type'], 'metric': doc['metric'], 'environment': doc['environment'],
            'limits': doc['limits']})
        for case in doc['cases']:
            evaluation = case.get('evaluation', {})
            lead_errors = [v['raw_rmse_mv'] for v in evaluation.get('per_lead', {}).values()]
            rows.append({'profile': profile, 'image_id': case['image_id'],
                'patient_id': case['patient_id'], 'image_type': case['image_type'],
                'status': case['status'], 'snr_db': evaluation.get('official_snr_db'),
                'mean_lead_raw_rmse_mv': statistics.mean(lead_errors) if lead_errors else None,
                'wall_seconds': case['wall_seconds'],
                'precision': case.get('provenance', {}).get('precision'),
                'image_sha256': case['image_sha256'], 'prediction_sha256': case.get('prediction_sha256')})
    out.mkdir(parents=True, exist_ok=True)
    with (out/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = {'schema_version': 1, 'runs': groups, 'cases': rows,
        'interpretation': ['Development smoke cohort, not the private Kaggle test',
            'Training overlap of public checkpoints is unknown',
            'A completed inference can still have a low-quality waveform',
            'Comparison of one full-ensemble image does not establish a cohort-wide improvement',
            'Single-profile CPU timings include concurrent work, not isolated throughput',
            'Official score aligns time and DC, also inspect raw errors and low-scoring cases']}
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    def fmt(value, digits=2):
        return 'n/d' if value is None else f'{value:.{digits}f}'
    cards = ''
    modes = ''
    for group in groups:
        cards += ('<article><span>'+html.escape(group['profile'])+'</span><strong>'
            +fmt(group['official_snr_db'])+' <small>dB</small></strong><p>'
            +f"{group['images']} imagens · {group['patients']} pacientes · {group['failures']} falhas"
            +'</p><p>Menor SNR: '+fmt(group['minimum_image_snr_db'])+' dB</p></article>')
        for kind, values in group['by_image_type'].items():
            modes += '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
                (group['profile'], kind, values['images'], fmt(values['official_snr_db']), values['failures']))+'</tr>'
    case_rows = ''
    for row in rows:
        case_rows += '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
            (row['profile'], row['image_id'], row['image_type'], fmt(row['snr_db']),
             fmt(row['mean_lead_raw_rmse_mv'], 4), fmt(row['wall_seconds'], 1), row['status']))+'</tr>'
    page = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Digitalização de ECG · referência</title>
<style>body{margin:0;background:#101722;color:#e5ebf5;font:16px/1.6 system-ui}main{max-width:1100px;margin:auto;padding:48px 24px}h1{font-size:clamp(28px,5vw,48px);line-height:1.15;margin:16px 0}h2{font-size:23px;margin-top:42px}p{color:#aab9cc}.eyebrow{color:#79daca;text-transform:uppercase;letter-spacing:.12em;font-size:12px}.cards{display:flex;flex-wrap:wrap;gap:16px;margin:28px 0}article{background:#182432;border:1px solid #2b3c4e;border-radius:14px;padding:22px;flex:1;min-width:240px}article span{color:#aab9cc}strong{display:block;font-size:42px;font-weight:600}small{font-size:18px}article p{margin:5px 0}.note{border-left:3px solid #e6bb70;padding:8px 18px;background:#202631}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:11px 12px;border-bottom:1px solid #2b3c4e}th{color:#aab9cc;font-weight:500}.scroll{overflow:auto}a{color:#79daca}footer{margin-top:38px;color:#8fa0b5;font-size:13px}</style>
<main><div class="eyebrow">ECG torch renderer · digitalização</div><h1>Da imagem ao sinal.</h1>
<p>Referência de retificação Heng e extração de ondas Tang. Inferência medida em CPU, com precisão bfloat16.</p>
<div class="cards">'''+cards+'''</div>
<div class="note">Esta é uma avaliação de desenvolvimento. A sobreposição com o treinamento dos checkpoints públicos é desconhecida. Os números não representam o teste privado do Kaggle nem validação clínica.</div>
<p>O ensemble completo e a geometria composta foram avaliados em uma imagem cada. Compare-os com o perfil simples na mesma imagem. Os tempos do perfil simples incluem trabalho concorrente.</p>
<h2>Qualidade por tipo de imagem</h2><p>A agregação usa as razões lineares de SNR antes de converter para dB, conforme a métrica oficial.</p>
<div class="scroll"><table><thead><tr><th>Perfil</th><th>Imagem</th><th>Quantidade</th><th>SNR (dB)</th><th>Falhas</th></tr></thead><tbody>'''+modes+'''</tbody></table></div>
<h2>Todos os exemplos</h2><p>Uma inferência concluída pode recuperar uma onda inadequada. Examine também os menores escores. O RMSE abaixo é a média entre derivações, sem alinhamento temporal ou correção de offset.</p>
<div class="scroll"><table><thead><tr><th>Perfil</th><th>Exame</th><th>Imagem</th><th>SNR (dB)</th><th>RMSE (mV)</th><th>CPU (s)</th><th>Status</th></tr></thead><tbody>'''+case_rows+'''</tbody></table></div>
<footer>Pesos, imagens originais e escaneamentos retificados não são redistribuídos nesta página. <a href="summary.json">Resumo JSON</a> · <a href="metrics.csv">Métricas CSV</a></footer></main></html>'''
    (out/'index.html').write_text(page)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    result = summarize(**vars(parser.parse_args()))
    print(json.dumps({'runs': len(result['runs']), 'case_rows': len(result['cases'])}))
