"""Self-contained review of frozen geometry fixes and every reserved replay."""
import argparse
import base64
from html import escape
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.cell_mesh import remap_cells
from examples.legacy_rectification_maps import legacy_grid_remap


def picture(rgb, label):
    image = Image.fromarray(rgb)
    image.thumbnail((720, 520), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, 'JPEG', quality=86, subsampling=0)
    encoded = base64.b64encode(stream.getvalue()).decode()
    return '<figure><figcaption>'+escape(label)+'</figcaption><img loading="lazy" alt="'+escape(label)+'" src="data:image/jpeg;base64,'+encoded+'"></figure>'


def preview(rgb, path, legacy=False):
    with np.load(path) as data:
        xy, selected, ids, cells = (data[k] for k in ('xy', 'selected', 'ids', 'cells'))
    if not len(cells):
        return np.full((180, 320, 3), 227, np.uint8)
    canonical = ids[:, ::-1]*10.
    size = tuple((canonical.max(0)+1).astype(int)[::-1])
    if max(size) > 4096:
        raise ValueError('Unexpected preview extent')
    sampler = legacy_grid_remap if legacy else remap_cells
    warped, support = sampler(rgb, xy[selected], canonical, ids, size, allowed_cells=cells)
    warped[~support] = (225, 229, 231)
    return warped


def percent(value):
    return f'{100*value:.1f}%'


def metrics_table(summary, selected):
    rows = []
    names = {'backend-balanced': 'Equilíbrio entre os dois motores',
             'torch-cached': 'Captura torch', 'offline': 'Captura 3D por raios'}
    for split, title in (('validation', 'Validação'), ('test', 'Teste reservado')):
        for backend, label in names.items():
            by = summary['splits'][split]['by_method'][backend]
            old, mesh, new = (by[k] for k in ('legacy-network', 'cells-network', selected))
            rows.append('<tr>'+''.join('<td>'+escape(x)+'</td>' for x in (
                title, label, percent(old['good_coverage']), percent(mesh['good_coverage']),
                percent(new['good_coverage']), percent(old['wrong_covered_fraction']),
                percent(new['wrong_covered_fraction'])))+'</tr>')
    return '<div class="scroll"><table><thead><tr><th>Conjunto</th><th>Captura</th><th>Correta antes</th><th>Só malha</th><th>Correta final</th><th>Com erro antes</th><th>Com erro final</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'


def create(bank, evaluation, out, research):
    summary = json.loads((evaluation/'summary.json').read_text())
    selection = json.loads((evaluation/'selection.json').read_text())
    report = json.loads((evaluation/'test.json').read_text())
    manifest = {r['key']: r for r in json.loads((bank/'manifest.json').read_text())}
    selected = summary['selected']
    variant = selection['candidate']
    blocks = []
    for case in sorted(report['cases'], key=lambda r: r['key']):
        key = case['key']
        with np.load(bank/manifest[key]['file']) as data:
            rgb = data['image']
        baseline = preview(rgb, evaluation/'topologies'/(key+'-legacy.npz'), legacy=True)
        corrected = preview(rgb, evaluation/'topologies'/(key+'-'+variant+'.npz'))
        old, new = (case['results'][m] for m in ('legacy-network', selected))
        delta = new['coverage_within_0_5mm']-old['coverage_within_0_5mm']
        tag = 'gain' if delta > .01 else 'regression' if delta < -.01 else 'stable'
        metadata = manifest[key]
        title = key+' · '+case['backend']+' · '+case['layout']+' · '+str(metadata.get('noise'))
        caption = 'Área correta '+percent(old['coverage_within_0_5mm'])+' → '+percent(new['coverage_within_0_5mm'])
        caption += '. Área com erro '+percent(old['covered_query_fraction']-old['coverage_within_0_5mm'])
        caption += ' → '+percent(new['covered_query_fraction']-new['coverage_within_0_5mm'])+'.'
        blocks.append('<article data-backend="'+case['backend']+'" data-change="'+tag+'" data-key="'+key+'"><h3>'+escape(title)+'</h3><p>'+caption+'</p><div class="images">'
            +picture(rgb, 'Entrada observada')+picture(baseline, 'Antes: Delaunay separado')
            +picture(corrected, 'Final: malha compartilhada + faixas condicionais')+'</div></article>')
        print(json.dumps({'preview': key}), flush=True)
    test = summary['splits']['test']
    paired = test['paired']['backend-balanced']['legacy-network']['good_coverage']
    ci = paired['paired_patient_bootstrap_ci95']
    arms = test['by_method']['backend-balanced']
    diagnostic_rows = ''.join('<tr><td>'+escape(label)+'</td><td>'+percent(arms[method]['good_coverage'])+'</td></tr>'
        for method, label in (('cells-oracle', 'Coordenadas e IDs exatos'),
            ('cells-anonymous', 'Pontos exatos sem IDs'), (selected, 'Detecções da CNN congelada')))
    selection_rows = ''.join('<tr><td>'+escape(name)+'</td><td>'+percent(row['backend_macro_good_coverage'])
        +'</td><td>'+percent(row['backend_macro_wrong_coverage'])+'</td><td>'+('Sim' if row['eligible'] else 'Não')+'</td></tr>'
        for name, row in selection['validation'].items())
    data_uri = 'data:application/json;base64,'+base64.b64encode(json.dumps(summary, indent=2).encode()).decode()
    checks = json.loads((research/'checks.json').read_text()) if (research/'checks.json').exists() else {}
    checks_html = '<pre>'+escape(json.dumps(checks, ensure_ascii=False, indent=2))+'</pre>'
    html = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ECG · correções da retificação</title><style>
:root{color-scheme:light;font-family:Inter,system-ui,sans-serif;color:#1d333d;background:#f3f5f4}body{margin:0}main{max-width:1400px;margin:auto;padding:42px 30px}h1{font-size:clamp(30px,4vw,52px);letter-spacing:-.035em;line-height:1.1;max-width:960px}h2{font-size:27px;margin-top:48px}h3{font-size:16px;margin:0 0 8px}p,li{line-height:1.65;max-width:1050px}.eyebrow{color:#52736f;font-size:13px;font-weight:700;letter-spacing:.12em}.lead{font-size:19px}.box{padding:22px 26px;background:#e3efea;border-left:4px solid #287961;margin:28px 0}.small{font-size:13px;color:#526870}table{border-collapse:collapse;min-width:730px;width:100%;background:white;font-size:14px}th,td{padding:13px 16px;border-bottom:1px solid #e4e9e7;text-align:left}th{background:#e4ece9}.scroll{overflow:auto}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0;display:flex;flex-direction:column}figcaption{font-size:12px;color:#526870;margin-bottom:8px}img{width:100%;height:300px;object-fit:contain;background:#e1e5e7}article{background:white;border:1px solid #dfe6e2;border-radius:10px;margin:18px 0;padding:20px}article p{font-size:14px;margin-top:0}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}select,input{font:inherit;padding:10px;border:1px solid #bdccc4;border-radius:5px;background:white}a{color:#126a7b}pre{white-space:pre-wrap;background:white;padding:20px;font-size:12px}footer{border-top:1px solid #cad7d0;margin-top:48px;padding-top:20px;color:#526870;font-size:13px}@media(max-width:800px){main{padding:24px 16px}.images{grid-template-columns:1fr}img{height:auto;max-height:520px}}
</style><main><div class="eyebrow">ECG TORCH RENDERER · 22 OUT 2026</div><h1>Malha corrigida, avaliação separada e limite da CNN exposto</h1>
<p class="lead">A retificação agora usa as mesmas células no mapeamento direto e na amostragem inversa. O perfil selecionado remove faixas periódicas quando há evidência no fundo e mantém o limiar original da CNN.</p>
<div class="box"><b>O RGB final é amostrado da entrada original.</b> A filtragem serve apenas à detecção de pontos. Não há substituição do traçado pela cabeça restauradora, novos pesos ou novo treinamento.</div>
<h2>O que foi corrigido</h2><p>O código anterior triangulava o espaço observado e o espaço canônico separadamente. Sob cisalhamento, as diagonais deixam de coincidir. No controle analítico com uma malha perfeita e 5.000 consultas interiores, o suporte antigo era de 8,18%. A malha explícita cobre 100% e preserva as coordenadas com erro máximo da ordem de 10<sup>−13</sup> pixel.</p>
<p>Cada quadrilátero completo usa uma diagonal fixa para gerar dois triângulos compartilhados entre as duas direções. Células com cantos ausentes ou não convexas ficam sem suporte. Sobreposições interiores ambíguas são rejeitadas, e a saída verifica a volta ao mesmo pixel canônico. Isso não certifica injetividade global de uma folha dobrada.</p>
<h2>Resultado medido</h2><p>“Correta” é a fração do mesmo domínio de consultas com suporte e erro físico de até 0,5 mm. “Com erro” é a fração com suporte e erro maior que 0,5 mm. O denominador não diminui quando o método rejeita uma região. Esse limiar é uma escolha de engenharia para o experimento.</p>'''
    html += metrics_table(summary, selected)
    html += '<p>Diferença no teste reservado, ponderando igualmente os motores: <b>'+f'{100*paired["delta"]:+.2f}'+' pontos percentuais</b> de área correta. Intervalo por bootstrap pareado de pacientes: ['+f'{100*ci[0]:+.2f}, {100*ci[1]:+.2f}'+'] pontos. São seis pacientes e 48 capturas. O intervalo é condicionado a esta pequena amostra de replays.</p>'
    html += '<h2>Onde permanece a falha</h2><div class="scroll"><table><thead><tr><th>Braço diagnóstico no teste</th><th>Área correta</th></tr></thead><tbody>'+diagnostic_rows+'</tbody></table></div>'
    html += '<p>Os braços usam as mesmas entradas e consultas. Os pontos verdadeiros são usados apenas nos controles. A CNN e os organizadores recebem RGB e pontos detectados. O alinhamento de avaliação admite apenas permutações e sinais dos eixos e origem inteira da grade. Não ajusta escala, cisalhamento ou uma transformação afim para esconder erros.</p>'
    html += '<h2>Escolha congelada antes do teste</h2><p>Foram comparadas cinco configurações nos seis primeiros pacientes da validação. A regra maximiza a área correta com limite de aumento da área com erro: até 0,5 ponto percentual no agregado e 1 ponto em cada motor, comparados à malha com o detector original. O perfil escolhido foi <b>'+escape(selection['cli_profile'])+'</b>. A opção de maior cobertura bruta não passou nessa regra.</p><div class="scroll"><table><thead><tr><th>Candidato</th><th>Área correta</th><th>Área com erro</th><th>Elegível</th></tr></thead><tbody>'+selection_rows+'</tbody></table></div>'
    html += '<p class="small">Congelamento: '+escape(selection['frozen_at_utc'])+'. Hashes do checkpoint, do código e da validação estão registrados. Os pacientes reservados são os seis últimos do teste, índices globais 1274 a 1279, sem sobreposição com a galeria previamente examinada.</p>'
    html += '<h2>Todas as 48 capturas reservadas</h2><p>As prévias JPEG são reduzidas para leitura. As métricas usam os arrays completos e até 4.096 consultas por imagem. Cinza nas saídas indica falta de suporte. O tamanho do canvas pode mudar com os pontos inferidos e não é usado como medida de cobertura.</p><div class="controls"><select id="backend"><option value="all">Todas as capturas</option><option value="torch-cached">Torch</option><option value="offline">3D por raios</option></select><select id="change"><option value="all">Todas as variações</option><option value="gain">Ganho acima de 1 ponto</option><option value="regression">Queda acima de 1 ponto</option><option value="stable">Variação até 1 ponto</option></select><input id="search" placeholder="Buscar ID da captura" aria-label="Buscar captura"><span id="count"></span></div>'+''.join(blocks)
    html += '<h2>Verificação e alcance</h2>'+checks_html
    html += '<p>Os sinais públicos originais foram conferidos por hash. Estas imagens são replays novos em CPU e LLVM das receitas e sementes congeladas. Não são os bytes do banco CUDA original e não constituem validação com fotografias reais. Não foi executado novo treinamento nem provisionada GPU.</p>'
    html += '<p><a download="ecg-rectification-summary.json" href="'+data_uri+'">Baixar todas as métricas resumidas</a> · <a href="https://github.com/fmenegui/ecg-torch-renderer/pull/19">Código e revisão no PR #19</a> · <a href="https://github.com/fmenegui/ecg-torch-renderer/tree/fix/cpu-rectification-support/research/rectification-attack-2026-10-22">Protocolo, seleção e resultados por captura</a> · <a href="https://physionet.org/content/ptb-xl/1.0.3/">Sinais PTB-XL</a></p><footer>Relatório de engenharia. A detecção e a organização da grade continuam limitando capturas difíceis.</footer></main><script>function filter(){const b=document.getElementById("backend"),c=document.getElementById("change"),s=document.getElementById("search");let n=0;document.querySelectorAll("article").forEach(a=>{const keep=(b.value==="all"||a.dataset.backend===b.value)&&(c.value==="all"||a.dataset.change===c.value)&&a.dataset.key.includes(s.value.trim());a.hidden=!keep;n+=keep});document.getElementById("count").textContent=n+" capturas"}for(const id of ["backend","change","search"])document.getElementById(id).addEventListener("input",filter);filter();</script></html>'
    out.write_text(html)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bank', 'evaluation', 'out', 'research'):
        parser.add_argument('--'+key, type=Path, required=True)
    create(**vars(parser.parse_args()))
