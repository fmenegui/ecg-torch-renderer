"""Self-contained research report from recovered experiment results."""
import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def embed(path):
    image = Image.open(path).convert('RGB')
    image.thumbnail((1000, 750))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def number(value, decimals=3):
    return 'indefinido' if value is None else f'{value:.{decimals}f}'


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    report = json.loads((root/'consolidated-summary.json').read_text())
    trials = json.loads((root/'trials.json').read_text())
    selected = json.loads((root/'selected-model.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels = [t['name'].replace('-20261021',' / B').replace('-1973',' / A') for t in trials]
    for axis, backend in zip(axes, ('online','offline')):
        rows = [next(x for x in report['points'] if x['name']==t['name'] and x['backend']==backend) for t in trials]
        values = np.array([x['detection']['f1'] for x in rows])
        bounds = np.array([x['f1_patient_bootstrap_95'] for x in rows])
        # Percentile intervals need not enclose the original estimate. Draw endpoints directly.
        axis.barh(np.arange(len(rows)), values, color=['#466a91','#466a91','#267970','#267970','#b57939','#b57939'], height=.65)
        axis.hlines(np.arange(len(rows)), bounds[:,0], bounds[:,1], color='#172b3a', linewidth=2)
        axis.set(yticks=np.arange(len(rows)), yticklabels=labels, xlim=(0,1), xlabel='F1 · raio de 1,5 px', title=backend)
        axis.invert_yaxis()
    figure.savefig(root/'detection-comparison.png',dpi=180)
    plt.close(figure)
    page='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>ECG · experimento consolidado</title>
<style>body{font:16px/1.6 system-ui;color:#193044;background:#f4f6f8;max-width:1180px;margin:auto;padding:32px}h1{font-size:38px;line-height:1.15}h2{margin-top:44px}table{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}td,th{padding:10px;text-align:left;border-bottom:1px solid #d8e0e7}th{font-size:13px}img{width:100%;height:auto}article{background:white;padding:20px;margin:18px 0;border-radius:12px}.notice{border-left:4px solid #bf8428;padding:14px;background:#fff5e3}.small{font-size:13px;color:#536678}summary{cursor:pointer;color:#246a86}code{overflow-wrap:anywhere}</style>
<p class="small">PESQUISA · 21 OUT 2026 · GPU RTX 3090</p><h1>Retificação de ECG: o que os experimentos demonstram</h1>
<p>Três variantes, duas sementes e o mesmo orçamento. Os sinais são originais de pacientes novos do PTB-XL. As imagens de teste são capturas sintéticas e não constituem validação clínica.</p>'''
    page+=f'<p>Modelo escolhido somente na validação: <strong>{html.escape(selected["name"])}</strong>. Treino com 32 pacientes, validação com oito e teste com oito. Cada paciente tem quatro capturas, totalizando 192 imagens.</p>'
    page+='<p class="notice">Contar pontos corretamente não garante ordenar a grade, preservar todas as derivações ou reconstruir o sinal. A geometria conhecida é um controle com acesso à verdade, não uma solução automática.</p>'
    page+='<h2>Detecção e organização da grade</h2><img alt="F1 por variante e semente, com intervalo bootstrap por paciente" src="'+embed(root/'detection-comparison.png')+'"><p class="small">Intervalos percentis de 95%, 2.000 reamostragens dos oito pacientes. Capturas de um mesmo paciente permanecem no mesmo grupo.</p><table><tr><th>Modelo</th><th>Captura</th><th>F1</th><th>Erro p95 (px)</th><th>Grades aceitas</th><th>Identidade relativa ≥99%</th></tr>'
    for row in report['points']:
        page+=f'<tr><td>{html.escape(row["name"])}</td><td>{row["backend"]}</td><td>{number(row["detection"]["f1"])}</td><td>{number(row["detection"]["p95_error_px"])}</td><td>{row["topology_accepted"]}/{row["images"]}</td><td>{row["accepted_with_relative_identity_at_least_99pct"]}/{row["images"]}</td></tr>'
    page+='</table><p class="small">Identidade relativa é medida somente entre nós selecionados e associados à referência. Não certifica a deformação inteira nem a ausência de sobreposições globais.</p><h2>Reconstrução do sinal</h2><table><tr><th>Entrada do npj congelado</th><th>Casos</th><th>Saída com layout correto</th><th>Cobertura média, incluindo falhas</th><th>SNR condicional (dB)</th></tr>'
    for arm,row in report['signals']['totals'].items():
        page+=f'<tr><td>{arm}</td><td>{row["attempted"]}</td><td>{row["correct_layout"]}</td><td>{row["all_case_mean_coverage"]:.1%}</td><td>{number(row["mean_snr_correct_layout_returned_only_db"],2)} ({row["snr_cases"]} casos)</td></tr>'
    page+='</table><p class="small">Ganho do papel fornecido. SNR nos trechos observados, com alinhamento limitado e remoção de DC. Não é a métrica da competição Kaggle. Falhas e layouts incorretos têm cobertura zero na população completa.</p>'
    if (root/'reference-results.json').exists():
        page+='<h2>Referências congeladas nas mesmas imagens</h2><table><tr><th>Referência</th><th>Saídas retornadas</th><th>Casos</th><th>Observação</th></tr>'
        references=json.loads((root/'reference-results.json').read_text())
        for arm in ('npj_dewarping_enabled','kaggle'):
            cases=[c for p in references for c in p[arm]]
            note='Perfis configurados, retificação não linear ativada' if arm.startswith('npj') else f'Perfil single, layout assumido 3x4+r1, compatível em {sum(c.get("layout_assumption_matches_input",False) for c in cases)} imagens'
            page+=f'<tr><td>{arm}</td><td>{sum(c["status"]=="returned" for c in cases)}</td><td>{len(cases)}</td><td>{note}</td></tr>'
        page+='</table>'
    page+='<h2>Inspeção de todas as imagens usadas na reconstrução</h2>'
    for patient in json.loads((root/'signal-inputs.json').read_text()):
        original=root/patient['original']
        manifest=json.loads((original/'manifest.json').read_text())
        for entry in manifest['cases']:
            page+=f'<article><details><summary>{html.escape(patient["source_id"])} · {entry["name"]} · {entry["layout"]}</summary>'
            for arm in ('original','oracle','learned'):
                path=original.parent/arm/entry['image']
                page+=f'<h3>{arm}</h3>'
                page+=('<img alt="Captura sintética de ECG" src="'+embed(path)+'">') if path.exists() else '<p>Geometria rejeitada. Nenhuma saída foi inventada.</p>'
            page+='</details></article>'
    page+='<h2>Limites da evidência</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in report['limits'])+'</ul>'
    page+='<p>As fotografias físicas exploratórias pertencem a um único ECG impresso anteriormente examinado. Uma região possui anotações automáticas aproximadas e duas imagens não possuem anotações. Não são um teste independente de acurácia clínica.</p>'
    page+='<details><summary>Protocolo e reprodução</summary><pre>'+html.escape(json.dumps(json.loads((root/'locked-protocol.json').read_text()),indent=2))+'</pre></details></html>'
    (root/'consolidated-report.html').write_text(page)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
