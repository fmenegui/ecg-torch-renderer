"""Build a self-contained visual audit from native and cached reference replays."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.cell_mesh import remap_cells
from ecg_gpu.digitization.complete_rectifier import fit_complete_map, render_complete


def encoded(path):
    image = Image.open(path).convert('RGB')
    image.thumbnail((1000, 750), Image.Resampling.LANCZOS)
    stream = BytesIO()
    image.save(stream, format='JPEG', quality=79)
    return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()


def analytic(root):
    folder = root/'content-preserving-previews/analytic'
    folder.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', (640, 400), 'white')
    draw = ImageDraw.Draw(image)
    for x in range(60, 600, 20):
        draw.line((x, 65, x, 365), fill=(235, 170, 175), width=1)
    for y in range(65, 366, 20):
        draw.line((60, y, 580, y), fill=(235, 170, 175), width=1)
    draw.rectangle((240, 165, 400, 265), fill='white')
    draw.text((20, 20), 'HEADER OUTSIDE DETECTED GRID / ORIGINAL RGB', fill=(120, 10, 35))
    points = [(x, int(220+15*np.sin(x/35)-55*np.exp(-((x-315)/7)**2))) for x in range(20, 621)]
    draw.line(points, fill=(15, 20, 25), width=3)
    row, col = np.mgrid[:16, :27]
    ids = np.column_stack((row.ravel(), col.ravel()))
    observed = ids[:, ::-1]*20.+[60, 65]
    keep = ~((observed[:, 0] >= 240) & (observed[:, 0] <= 400)
             & (observed[:, 1] >= 165) & (observed[:, 1] <= 265))
    observed, ids = observed[keep], ids[keep]
    canonical = ids[:, ::-1]*5.
    old, _ = remap_cells(np.asarray(image), observed, canonical*4, ids, (301, 521))
    mapping, _ = fit_complete_map(observed, canonical, ids, (400, 640))
    new, _, _, _ = render_complete(np.asarray(image), mapping, pixels_per_mm=4)
    image.save(folder/'input.png')
    Image.fromarray(old).save(folder/'before.png')
    Image.fromarray(new).save(folder/'after.png')
    return {key: encoded(folder/file) for key, file in
            (('input', 'input.png'), ('before', 'before.png'), ('output', 'after.png'))}


def build(root, out):
    evaluation = root/'complete-preserved-evaluation'
    summary = json.loads((evaluation/'summary.json').read_text())
    rows = json.loads((root/'content-preserving-previews/manifest.json').read_text())
    comparison = root/'comparison'
    native_manifest = {c['key']: c for c in json.loads((root/'diagnostic/manifest.json').read_text())}
    reference_rows = json.loads((comparison/'references/references.json').read_text())
    references = {(r['method'], r['id']): r for r in reference_rows}
    records = []
    for row in rows:
        folder = Path(row['folder'])
        record = {k: v for k, v in row.items() if k != 'folder'}
        record['input'] = encoded(folder/'original-rgb.png')
        record['output'] = encoded(folder/'output-original-rgb.png')
        record['evidence'] = encoded(folder/'geometry-evidence.png')
        record['eligibility'] = encoded(folder/'geometry-eligibility.png')
        metadata = json.loads((folder/'report.json').read_text())['complete_geometry']
        record['reason'] = metadata.get('reason') or 'Malha completa proposta. A precisão não está certificada.'
        if row['kind'] == 'cached-reference':
            old = comparison/'ours'/row['key']/'rectified-original-rgb.png'
            record['before'] = encoded(old) if old.exists() else None
            record['reference'] = encoded(comparison/'references/npj'/row['key']/'rectified.png')
            reference = references[('npj', row['key'])]
            mode = 'identidade, fallback' if reference.get('identity_fallback_reported') else 'campo não linear estimado'
            record['reference_status'] = mode
        else:
            record['backend'] = native_manifest[row['key']]['backend']
            record['scenario'] = native_manifest[row['key']].get('scenario', 'captura 3D')
        records.append(record)
    control = analytic(root)
    table = []
    for split, label in (('validation', 'Validação'), ('test', 'Teste antigo, retrospectivo')):
        data = summary['splits'][split]
        for backend, title in (('backend_balanced', 'equilíbrio entre backends'),
                               ('torch-cached', 'torch'), ('offline', '3D')):
            b, n = data['baseline'][backend], data['eligible_coordinates'][backend]
            values = ''.join(f'<td>{v:.2%}</td>' for v in
                (b['good_coverage'], n['good_coverage'], b['wrong_covered_fraction'], n['wrong_covered_fraction']))
            table.append(f'<tr><td>{label}, {title}</td>{values}</tr>')
    data_json = json.dumps(records, ensure_ascii=False).replace('</', '<\\/')
    control_json = json.dumps(control).replace('</', '<\\/')
    document = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ECG: retificação sem apagar conteúdo</title>
<style>
[hidden]{display:none!important}
:root{font-family:system-ui,sans-serif;color:#18342f;background:#f2f5f2;font-size:16px}*{box-sizing:border-box}body{margin:0}main{max-width:1440px;margin:auto;padding:40px 26px 80px}header{max-width:980px}h1{font-size:clamp(2rem,4vw,3.4rem);line-height:1.1;letter-spacing:-.045em;margin:16px 0}h2{font-size:1.5rem;margin:34px 0 12px}p{line-height:1.6;max-width:1000px}.eyebrow{font-size:.78rem;letter-spacing:.13em;text-transform:uppercase;color:#53736a}.note{background:#fff2dd;border-left:4px solid #b97820;padding:14px 18px;line-height:1.5;margin:18px 0}a{color:#236e5b}.cards,.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.cards{grid-template-columns:repeat(3,minmax(0,1fr))}.stat{border:1px solid #d6e2da;border-radius:12px;padding:18px;background:#fff}.stat strong{font-size:1.9rem;display:block;letter-spacing:-.04em}.stat small{color:#557168}table{border-collapse:collapse;width:100%;background:#fff;font-size:.9rem}td,th{text-align:left;padding:12px;border-bottom:1px solid #e0e8e2}th{background:#e2ece5;font-weight:600}.scroll{overflow:auto}.toolbar{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0;align-items:center}select,button{font:inherit;padding:10px 12px;border:1px solid #bbcec1;background:#fff;border-radius:7px;color:#18342f}button{cursor:pointer}button:hover{background:#e2ece5}select{max-width:100%}figure{margin:0;background:white;border:1px solid #d7e2da;border-radius:10px;overflow:hidden}figcaption{padding:10px 14px;font-size:.83rem;background:#e8efe9}figure img{display:block;width:100%;height:360px;object-fit:contain;background:#f9faf8;cursor:zoom-in}.grid.small img{height:220px}#detail{font-size:.92rem;line-height:1.6;color:#425f56;padding:12px 0}.controls{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.controls img{height:240px}.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:.85rem;padding:14px 0}.swatch{width:12px;height:12px;display:inline-block;margin-right:5px;border-radius:2px}.sources{font-size:.93rem;line-height:1.6}.sources li{margin:9px 0}footer{border-top:1px solid #c8d8cc;margin-top:32px;padding-top:16px;color:#5b7366;font-size:.85rem}dialog{border:0;padding:12px;max-width:96vw;background:#fff;border-radius:10px}dialog::backdrop{background:#10251fce}dialog img{display:block;max-width:92vw;max-height:87vh;object-fit:contain}dialog button{float:right;margin-bottom:8px}@media(max-width:760px){main{padding:25px 14px}.cards,.grid,.controls{grid-template-columns:1fr}.stat strong{font-size:1.6rem}figure img{height:auto;min-height:180px}.grid.small img{height:200px}td,th{padding:9px;font-size:.82rem}}@media print{.toolbar,dialog{display:none}body{background:white}main{padding:0}.grid{break-inside:avoid}}
</style><main><header><div class="eyebrow">Revisão técnica e comparação visual · 22 outubro 2026</div>
<h1>O grid pode faltar.<br>O traçado deve continuar visível.</h1>
<p>A saída antiga apagava regiões sem uma célula completa detectada e cortava a imagem à extensão do grid. A nova saída preserva o quadro inteiro, mantém a transformação das células observadas e estima somente a geometria que falta. Quando essa extensão falha, conserva o RGB original.</p>
<p><a href="https://github.com/fmenegui/ecg-torch-renderer/pull/19">Código e validação no PR #19</a> · <a href="https://github.com/fmenegui/ecg-torch-renderer/tree/fix/cpu-rectification-support/research/content-preserving-rectification-2026-10-22">Protocolo, fontes e todos os resultados</a></p></header>
<div class="cards"><div class="stat"><small>Cobertura correta, teste antigo</small><strong>65,33% → 68,86%</strong><small>Erro físico até 0,5 mm, coordenadas elegíveis</small></div><div class="stat"><small>Cobertura com erro maior</small><strong>2,43% → 2,94%</strong><small>A interpolação também aumenta a área errada</small></div><div class="stat"><small>48 imagens retrospectivas</small><strong>32 propostas · 16 originais</strong><small>Malhas completas antes do limite de tamanho da saída</small></div></div>
<div class="note"><b>A precisão da geometria inferida ainda é limitada.</b> A cobertura errada do campo completo, sem o filtro de elegibilidade, chega a 15,62% no total e 26,30% nas fotos 3D. Uma imagem inteira e visualmente plausível não é uma calibração física validada. O original é salvo separadamente e nenhuma máscara de incerteza apaga o RGB.</div>
<h2>Controle: traçado em uma falha do grid</h2><p>Deformação afim conhecida, com cabeçalho fora do grid detectado e um trecho de onda atravessando a região sem pontos. O recorte antigo perde ambos. O novo mapa conserva o quadro. Este controle verifica o defeito específico, sem provar generalização a fotografias.</p>
<div id="control" class="controls"></div>
<h2>Inspecione as imagens e as falhas</h2><div class="toolbar"><select id="dataset" aria-label="Conjunto"><option value="native-retrospective">48 capturas nativas, teste retrospectivo</option><option value="cached-reference">8 JPEGs com referência npj em cache</option></select><select id="filter" aria-label="Filtro"><option value="all">Todas as saídas</option><option value="rectification-proposal">Propostas de retificação</option><option value="original-preserved">Original conservado</option></select><button id="previous">← Anterior</button><select id="case" aria-label="Imagem"></select><button id="next">Próxima →</button></div>
<div id="detail"></div><div class="grid"><figure><figcaption>Entrada RGB original</figcaption><img id="input" alt="ECG original"></figure><figure><figcaption id="output-label">Saída completa</figcaption><img id="output" alt="Saída completa com RGB original"></figure></div>
<div class="grid" id="reference-grid" hidden style="margin-top:14px"><figure><figcaption>Nossa saída anterior, somente células</figcaption><img id="before" alt="Antigo recorte parcial"></figure><figure><figcaption id="reference-label">Referência npj, adaptador em cache</figcaption><img id="reference" alt="RGB amostrado pelo adaptador npj"></figure></div>
<div class="legend"><span><i class="swatch" style="background:rgb(49,145,109)"></i>Célula observada</span><span><i class="swatch" style="background:rgb(104,160,207)"></i>Interpolação dentro do casco</span><span><i class="swatch" style="background:rgb(227,150,85)"></i>Extrapolação</span><span><i class="swatch" style="background:rgb(225,229,231)"></i>Fora do quadro ou sem mapa completo</span></div>
<div class="grid small"><figure><figcaption>Evidência geométrica, separada do RGB</figcaption><img id="evidence" alt="Mapa de evidência geométrica"></figure><figure><figcaption>Coordenadas elegíveis em branco, sem certificado de precisão</figcaption><img id="eligibility" alt="Máscara separada de elegibilidade geométrica"></figure></div>
<p>Amplie qualquer imagem clicando. Os previews são reduzidos e amostrados a 2 px/mm. O limite de tamanho em 8 px/mm pode conservar o original em casos adicionais. Os números usam coordenadas nativas e não estes JPEGs de prévia.</p>
<h2>Resultado físico no mesmo conjunto de consultas</h2><div class="scroll"><table><thead><tr><th>Conjunto e captura</th><th>Correta antes</th><th>Correta agora</th><th>Errada antes</th><th>Errada agora</th></tr></thead><tbody>__TABLE__</tbody></table></div>
<p>Correta significa erro até 0,5 mm sobre todas as consultas físicas fixadas. Errada significa coordenada finita com erro maior. Consultas sem coordenadas continuam no denominador. Pacientes têm peso igual e os dois backends têm peso igual. Há seis pacientes por conjunto. Em rejeições, as coordenadas parciais anteriores continuam disponíveis separadamente, mesmo que o RGB original permaneça sem retificação global.</p>
<p>O raio de elegibilidade 0,75 foi escolhido somente na validação e congelado antes desta comparação. O ganho retrospectivo de cobertura correta é de 3,53 pontos percentuais, intervalo bootstrap pareado por paciente de aproximadamente 1,10 a 6,98 pontos. A cobertura errada subiu 0,51 ponto, ligeiramente acima do orçamento de desenvolvimento de 0,50. Nenhuma das 48 imagens perdeu mais de um ponto de cobertura correta, e 21 ganharam mais de um ponto. O conjunto de teste já havia sido inspecionado e não é um novo teste independente.</p>
<h2>O que as referências acrescentam</h2><ul class="sources"><li><a href="https://www.nature.com/articles/s41746-025-02327-1">npj Digital Medicine</a>: o artigo usa correção global por Hough e recorte informado pelo sinal. O dewarper não linear experimental do repositório, baseado em TPS, é um componente diferente.</li><li><a href="https://patents.google.com/patent/US20240144417A1/en">Patente de geometria de Dotter/Gridder</a>: inclui inferência de vértices ausentes e amostragem por célula. <a href="https://patents.google.com/patent/US20260065633A1/en">A outra patente revisada</a> trata principalmente da extração e associação de trajetórias de onda.</li><li><a href="https://github.com/tanghaozhe/physionet-ecg-image-digitization-3rd-place">Kaggle, terceiro lugar</a> e <a href="https://github.com/dangnh0611/kaggle_ecg_digitization">quinto lugar</a>: interpolam geometria ausente, com hipóteses de grid e layout que exigem checagem. <a href="https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/2nd-place-solution">O segundo lugar</a> mantém os estágios iniciais de Heng e concentra melhorias na extração do sinal.</li></ul>
<p>O nome “Lauder” não foi identificado com segurança. A revisão técnica cobre as duas famílias de patentes ligadas ao trabalho anterior. Não atribuí esse nome a uma delas.</p>
<h2>O que ainda falta para superar os concorrentes</h2><p>Os oito JPEGs não têm geometria de referência comum e pertencem a um paciente. As saídas npj são resultados anteriores do nosso adaptador, que amostra RGB pelo campo estimado upstream. Duas usam fallback de identidade. Não são imagens oficiais do artigo. O adaptador Kaggle falhou no estágio inicial nos oito casos e isso continua registrado. Nenhuma rede concorrente foi executada novamente nesta revisão. Esses exemplos não sustentam um ranking geral.</p>
<p>A maior limitação permanece a detecção em papel pequeno, oblíquo e degradado. O próximo estudo precisa de detecção em resolução nativa ou por blocos, evidência de linhas, validação em novos pacientes e comparação pareada em fotografias reais. Também precisa medir a distorção do sinal extraído. Não há evidência suficiente para afirmar que este sistema já supera todos.</p>
<footer>Revisão de fontes primárias, correção da saída e avaliação geométrica em CPU. Sem novo treino, sem execução CUDA e sem validação clínica. Original e propostas são mantidos separadamente. Relatório autossuficiente, sem requisições externas.</footer>
<dialog id="zoom"><button id="close">Fechar</button><img alt="Imagem ampliada"></dialog>
<script>const records=__DATA__;const control=__CONTROL__;
const $=id=>document.getElementById(id);const percent=x=>(100*x).toFixed(2).replace('.',',')+'%';
for(const [key,label] of [['input','Entrada completa'],['before','Antes: regiões apagadas'],['output','Agora: quadro completo']]){const f=document.createElement('figure');const c=document.createElement('figcaption');c.textContent=label;const i=document.createElement('img');i.src=control[key];i.alt=label;f.append(c,i);$('control').append(f)}
let filtered=[];
function load(){filtered=records.filter(r=>r.kind===$('dataset').value&&($('filter').value==='all'||r.status===$('filter').value));$('case').replaceChildren(...filtered.map((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.key+' · '+(r.status==='rectification-proposal'?'proposta':'original conservado');return o}));show()}
function show(){const r=filtered[Number($('case').value)];if(!r){$('detail').textContent='Nenhuma imagem neste filtro.';return}for(const id of ['input','output','evidence','eligibility'])$(id).src=r[id];$('output-label').textContent=r.status==='rectification-proposal'?'Proposta completa, geometria ainda não certificada':'Original conservado, sem retificação global';$('reference-grid').hidden=r.kind!=='cached-reference';if(r.kind==='cached-reference'){if(r.before)$('before').src=r.before;$('reference').src=r.reference;$('reference-label').textContent='npj, adaptador em cache · '+r.reference_status;$('detail').textContent=r.key+' · '+r.reason+' Referência Kaggle: falha no estágio inicial.'}else{$('detail').textContent=r.key+' · '+r.backend+' · '+r.scenario+'. Cobertura correta: '+percent(r.baseline.good_coverage)+' → '+percent(r.metrics.good_coverage)+'. Cobertura errada: '+percent(r.baseline.wrong_covered_fraction)+' → '+percent(r.metrics.wrong_covered_fraction)+'. '+r.reason}}
$('dataset').addEventListener('change',load);$('filter').addEventListener('change',load);$('case').addEventListener('change',show);for(const [id,step] of [['previous',-1],['next',1]])$(id).addEventListener('click',()=>{if(!filtered.length)return;$('case').value=(Number($('case').value)+step+filtered.length)%filtered.length;show()});document.addEventListener('click',e=>{if(e.target.tagName==='IMG'&&!e.target.closest('dialog')){$('zoom').querySelector('img').src=e.target.src;$('zoom').showModal()}});$('close').addEventListener('click',()=>$('zoom').close());load();
</script></main></html>'''
    document = document.replace('__TABLE__', ''.join(table)).replace('__DATA__', data_json).replace('__CONTROL__', control_json)
    out.write_text(document)
    print(json.dumps({'out': str(out), 'images': len(records), 'bytes': out.stat().st_size}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    build(**vars(parser.parse_args()))
