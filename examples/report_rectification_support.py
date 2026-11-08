"""Build a self-contained qualitative report from completed frozen CPU outputs."""
import argparse
import base64
import html
import json
from pathlib import Path


def embedded(path):
    mime = 'image/jpeg' if path.suffix.lower() in ('.jpg', '.jpeg') else 'image/png'
    return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()


def build(comparison, evidence, out, pr_url):
    rows = json.loads((comparison/'inputs/images.json').read_text())
    ours = {r['id']: r for r in json.loads((comparison/'ours.json').read_text())}
    refs = json.loads((comparison/'references/references.json').read_text())
    lookup = {(r['method'], r['id']): r for r in refs}
    bootstrap = json.loads((evidence/'patient-bootstrap.json').read_text())
    validation = json.loads((evidence/'validation.json').read_text())
    payload = {'inputs': rows, 'ours': list(ours.values()), 'references': refs,
               'patient_bootstrap': bootstrap, 'validation': validation}
    page = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retificação de ECG: retomada e comparação</title>
<style>
*{box-sizing:border-box}body{font:16px/1.6 system-ui,sans-serif;background:#f4f6f7;color:#172931;margin:0}
main{max-width:1380px;margin:auto;padding:30px 22px}h1{font-size:clamp(27px,4vw,44px);line-height:1.15;letter-spacing:-.035em}
h2{font-size:25px;margin-top:36px}h3{font-size:20px}p{max-width:1000px}a{color:#006878}
.note{border-left:4px solid #cb9135;padding:12px 18px;background:#fff7e7;border-radius:5px}
.card{background:white;padding:20px;border:1px solid #dde4e8;border-radius:14px;margin:24px 0}
.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0;min-width:0}
img{width:100%;height:300px;object-fit:contain;background:#f1f3f4;border:1px solid #e1e7eb;cursor:zoom-in}
figcaption{font-size:14px;font-weight:600;margin:7px 0}.small{color:#53666f;font-size:14px}
table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #dde4e8}
.scroll{overflow:auto}th{color:#46616b}.pill{display:inline-block;padding:2px 9px;background:#e7f1f3;border-radius:8px;font-size:14px}
details{margin:14px 0}dialog{width:95vw;max-width:none;height:95vh;border:0;border-radius:12px;padding:12px;background:#f4f6f7}
dialog img{height:88vh;object-fit:contain;border:0;cursor:zoom-out}dialog::backdrop{background:#172931df}
button{background:#006878;color:white;padding:10px 14px;border:0;border-radius:7px;cursor:pointer}
@media(max-width:800px){.images{grid-template-columns:1fr}img{height:auto;max-height:650px}main{padding:20px 12px}.card{padding:14px}}
</style><main><div class="small">22 de outubro de 2026 · inspeção em CPU</div>
<h1>Os pontos melhoraram.<br>A retificação ainda perde regiões.</h1>
<p>O estudo anterior terminou com 10.240 capturas sintéticas de 1.280 pacientes públicos do PTB-XL e quatro modelos de 10.000 passos.
Esta retomada recuperou os resultados, concluiu o bootstrap por paciente e repetiu a comparação visual nas oito primeiras capturas da galeria recuperada.</p>
<div class="note">As oito imagens desta comparação vêm de <b>um único paciente</b>.
São JPEGs da galeria, apresentados com os mesmos bytes aos três métodos. O teste maior tem 1.024 imagens de 128 pacientes.
Contar imagens devolvidas não mede precisão geométrica, cobertura do ECG ou preservação clínica do sinal.</div>
<h2>O que foi corrigido</h2>
<p>O comparador Kaggle agora verifica apenas os sete arquivos necessários à geometria quando a extração do sinal está desativada.
Os hashes continuam obrigatórios. O adaptador npj também passou a preservar a saída de contingência quando a lista de pontos fica vazia,
registrando explicitamente quando o ajuste não linear falhou. O código dos autores e seus pesos foram mantidos.</p>
<div class="scroll"><table><tr><th>Método</th><th>Imagens devolvidas</th><th>Interpretação</th></tr>'''
    returned = {name: sum(r['method'] == name and r['status'] == 'returned' for r in refs) for name in ('kaggle', 'npj')}
    fallback = sum(r.get('identity_fallback_reported', False) for r in refs)
    page += f'<tr><td>Nosso modelo joint-1973</td><td>{len(ours)}/{len(rows)} parciais</td><td>Há buracos e fragmentação em vários casos.</td></tr>'
    page += f'<tr><td>Kaggle, etapas 0 e 1</td><td>{returned["kaggle"]}/{len(rows)}</td><td>Referências do documento insuficientes na etapa 0.</td></tr>'
    page += f'<tr><td>npj, geometria aplicada ao RGB</td><td>{returned["npj"]}/{len(rows)}</td><td>{fallback} saídas usam a contingência após falha do ajuste TPS.</td></tr></table></div>'
    page += '<p class="small">A saída RGB do npj é uma adaptação para inspeção. A implementação publicada aplica o dewarp ao mapa de probabilidade do sinal. Nenhuma extração de traçado foi executada nesta comparação.</p>'
    page += '<h2>Saídas lado a lado</h2><p>Toque nas imagens para ampliar. Na nossa saída, áreas brancas sem suporte permanecem ausentes. A cobertura abaixo usa o retângulo dos índices relativos observados, sem medir a página inteira.</p>'
    for row in rows:
        key = row['id']
        rec = ours[key]
        support = rec['cell_support']
        ref = lookup['npj', key]
        tag = 'Contingência após falha TPS' if ref.get('identity_fallback_reported') else 'Saída do adaptador npj'
        page += f'<article class="card"><h3>{html.escape(key)}</h3>'
        page += f'<span class="pill">Células com suporte: {support["cell_coverage_fraction"]:.1%}</span> '
        page += f'<span class="pill">Regiões conectadas: {support["connected_components"]}</span> '
        page += f'<span class="pill">Maior região / extensão observada: {support["largest_component_extent_fraction"]:.1%}</span><div class="images">'
        for label, path in [('Entrada comum', comparison/'inputs'/row['file']),
                            ('Nosso: retificação parcial do RGB original', comparison/'ours'/key/'rectified-original-rgb.png'),
                            (tag, comparison/'references'/ref['image'])]:
            page += f'<figure><figcaption>{html.escape(label)}</figcaption><img loading="lazy" src="{embedded(path)}" alt="{html.escape(key+": "+label)}"></figure>'
        page += '</div><p class="small">Kaggle: '+html.escape(lookup['kaggle', key].get('error', 'Imagem devolvida'))+'</p>'
        source = embedded(comparison/'inputs'/row['file'])
        page += f'<a class="small" download="{html.escape(row["file"])}" href="{source}">Baixar entrada exata</a></article>'
    page += '<h2>O que o teste maior permite concluir</h2><p>Abaixo, diferença de F1 entre treino conjunto e treino apenas com pontos, para cada semente. O bootstrap pareado reamostrou 128 pacientes em 2.000 repetições. Os intervalos são condicionais aos modelos treinados, sem estimar a incerteza de novas inicializações.</p>'
    page += '<div class="scroll"><table><tr><th>Semente</th><th>Captura</th><th>Diferença F1</th><th>IC 95%</th></tr>'
    for r in bootstrap['paired_seed_comparisons']:
        lo, hi = r['ci95']
        page += f'<tr><td>{r["seed"]}</td><td>{"3D" if r["backend"] == "offline" else "torch"}</td><td>{r["delta_f1_right_minus_left"]:+.4f}</td><td>[{lo:+.4f}, {hi:+.4f}]</td></tr>'
    page += '</table></div><p>Nas capturas 3D, as duas sementes apresentam ganho de F1. Nas capturas torch, o ganho não se repete na segunda semente. Isso sustenta uma melhora na detecção em parte do domínio sintético. A montagem de uma grade contínua e a recuperação do traçado ainda precisam de avaliação própria.</p>'
    page += '<h2>Próximo avanço</h2><ol><li>Avaliar localização do papel e inferência em maior resolução nas capturas em que a folha ocupa poucos pixels.</li><li>Corrigir a atribuição dos índices da grade e medir conectividade, orientação, cobertura e erro geométrico juntos. Nenhuma região ausente deve ser preenchida sem evidência.</li><li>Reexecutar os três métodos com as coordenadas de referência do banco e comparar erro no mesmo suporte, mantendo as falhas no denominador. Depois medir o sinal recuperado e validar fotografias reais.</li></ol>'
    page += '<h2>Verificação e fontes</h2><p>Passaram 23 testes focados. A integração executou 378 testes: 351 passaram, 25 foram pulados e dois ficaram bloqueados pela ausência dos arquivos originais de regressão. Esses arquivos não foram regenerados. Uma galeria adicional de 16 cenários foi exportada e inspecionada em CPU.</p>'
    page += '<p>O Pod A40 foi consultado e estava parado. O volume com o banco e os checkpoints continua preservado. Nenhum novo treino foi iniciado nesta retomada.</p>'
    page += '<p><a href="'+html.escape(pr_url)+'">Alterações e evidências no GitHub</a> · <a href="https://github.com/Ahus-AIM/Open-ECG-Digitizer">Código dos autores do npj</a> · <a href="https://www.nature.com/articles/s41746-025-02327-1">Artigo npj</a> · <a href="https://physionet.org/content/ptb-xl/1.0.3/">PTB-XL 1.0.3</a></p>'
    page += '<button id="download">Baixar métricas e proveniência</button>'
    page += '<script type="application/json" id="evidence">'+json.dumps(payload).replace('<', '\\u003c')+'</script>'
    page += '''</main><dialog id="zoom"><img alt="Imagem ampliada"></dialog><script>
const zoom=document.getElementById('zoom');
document.querySelectorAll('main img').forEach(im=>im.onclick=()=>{zoom.querySelector('img').src=im.src;zoom.showModal()});
zoom.onclick=()=>zoom.close();
document.getElementById('download').onclick=()=>{const url=URL.createObjectURL(new Blob([document.getElementById('evidence').textContent],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='ecg-retificacao-evidencias.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></html>'''
    out.write_text(page)
    return {'out': str(out), 'bytes': out.stat().st_size}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('comparison', 'evidence', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--pr-url', required=True)
    print(json.dumps(build(**vars(parser.parse_args()))))
