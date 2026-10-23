"""Build the self-contained M2 engineering report from completed measurements.

Offline reporting only. Requires matplotlib in addition to the study tools.
Run after summarize_nextgen_study.py --include-test, never during selection.
"""
# ruff: noqa: E501
import argparse
import base64
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SELECTED = 'soft-1.0'
INK, TEAL, RED, GRAY = '#172d43', '#087f7b', '#ba4a39', '#dce4e8'


def read(path):
    return json.loads((ROOT / path).read_text())


def fmt(value, digits=2):
    return f'{value:.{digits}f}'.replace('.', ',')


def pct(value):
    return fmt(100 * value) + '%'


def interval(row, key):
    delta = row['delta_vs_B0'][key] * 100
    ci = row['delta_95_descriptive_interval'][key]
    return f'{delta:+.4f} [{ci["lower"]*100:+.4f}, {ci["upper"]*100:+.4f}]'.replace('.', ',')


def embedded(path, alt):
    path = Path(path)
    mime = 'image/jpeg' if path.suffix.lower() in ('.jpg', '.jpeg') else 'image/png'
    content = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'<img src="data:{mime};base64,{content}" alt="{html.escape(alt)}">'


def figure_data(fig):
    stream = io.BytesIO()
    fig.savefig(stream, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return stream.getvalue()


def plots(summary, analytic, captures, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': INK, 'text.color': INK})
    fig, ax = plt.subplots(figsize=(9.5, 3.3))
    labels, metrics = [], []
    for split, label in [('validation', 'Desenvolvimento'), ('test', 'Pacientes novos')]:
        for method, name in [('B0', 'B0'), (SELECTED, 'M2')]:
            labels.append(label + '\n' + name)
            metrics.append(summary['splits'][split]['methods'][method]['domains']['grid']['overall']['metrics'])
    x = np.arange(4)
    g = np.array([v['g'] for v in metrics])*100
    w = np.array([v['w'] for v in metrics])*100
    ax.barh(x, g, color=TEAL, label='Correto, erro até 0,5 mm')
    ax.barh(x, w, left=g, color=RED, label='Retornado com erro > 0,5 mm')
    ax.barh(x, 100-g-w, left=g+w, color=GRAY, label='Sem coordenada elegível')
    for i in x:
        ax.text(g[i]/2, i, fmt(g[i])+'%', ha='center', va='center', color='white', weight='bold')
    ax.set(yticks=x, yticklabels=labels, xlim=(0, 100), xlabel='Consultas fixas no domínio da grade (%)')
    ax.invert_yaxis()
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.2), ncol=1, frameon=False)
    fig.tight_layout()
    (out/'coverage.png').write_bytes(figure_data(fig))

    cases = captures['cases']
    groups = sorted({c['group_id'] for c in cases})
    deltas = {metric: [] for metric in ('g', 'w')}
    for group in groups:
        for metric in deltas:
            values = []
            for backend in ('offline', 'torch-cached'):
                rows = [c for c in cases if c['group_id'] == group and c['backend'] == backend]
                values.append(np.mean([c['methods'][SELECTED]['domains']['grid'][metric]
                                       - c['methods']['B0']['domains']['grid'][metric] for c in rows]))
            deltas[metric].append(100*np.mean(values))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.1), sharey=True)
    for ax, metric, title in zip(axes, ('g', 'w'), ('Δ cobertura correta', 'Δ cobertura errada')):
        ax.barh(np.arange(8), deltas[metric], color=TEAL if metric == 'g' else RED)
        ax.axvline(0, color=INK, linewidth=.7)
        ax.set(yticks=np.arange(8), yticklabels=[g.split('-')[-1].removesuffix('.0') for g in groups],
               xlabel='M2 menos B0, pontos percentuais', title=title)
    axes[0].set_ylabel('Paciente PTB-XL')
    fig.tight_layout()
    (out/'patients.png').write_bytes(figure_data(fig))

    by_key = {c['case']['key']: c for c in analytic['cases']}
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1), sharey=True)
    specs = [('projective-noisy-7919', 'Projetiva com ruído'),
             ('crease-clean', 'Dobra limpa'), ('crease-noisy', 'Dobra com ruído')]
    for ax, (key, title) in zip(axes, specs):
        methods = by_key[key]['methods']
        values = [methods[m]['all_queries']['physical_rmse_mm']
                  for m in ('B0', 'soft-0.15-guard-false', 'soft-0.15-guard-true')]
        ax.bar(np.arange(3), values, color=[INK, RED, TEAL], width=.65)
        for i, value in enumerate(values):
            ax.text(i, value+.012, fmt(value, 3), ha='center', fontsize=9)
        ax.set(xticks=[0, 1, 2], xticklabels=['B0', 'Sem\nproteção', 'Com\nproteção'], title=title, ylim=(0, .65))
    axes[0].set_ylabel('RMSE físico, mm')
    fig.tight_layout()
    (out/'controls.png').write_bytes(figure_data(fig))


def report(out, status_path=None):
    summary = read('build/nextgen-summary/summary.json')
    analytic = read('build/refinement-controls/results.json')
    native = read('build/native-comparison/results.json')
    test = read('build/nextgen-evaluation/test.json')
    assert len(test['cases']) == 64 and 'summary' in test
    assert summary['selected_method'] == SELECTED
    out.parent.mkdir(parents=True, exist_ok=True)
    assets = out.parent/'m2-report-assets'
    assets.mkdir(exist_ok=True)
    plots(summary, analytic, test, assets)
    rows, cis = [], []
    for split, label in [('validation', 'Desenvolvimento · 6 pacientes'), ('test', 'Teste · 8 pacientes novos')]:
        methods = summary['splits'][split]['methods']
        for domain, dlabel in [('grid', 'Grade'), ('paper', 'Papel visível')]:
            for method, name in [('B0', 'B0'), (SELECTED, 'M2')]:
                m = methods[method]['domains'][domain]['overall']['metrics']
                rows.append(f'<tr><td>{label}</td><td>{dlabel}</td><td>{name}</td>'
                            + ''.join(f'<td>{pct(m[k])}</td>' for k in ('g', 'w', 'c', 'r'))+'</tr>')
            r = methods[SELECTED]['domains'][domain]['overall']
            cis.append(f'<tr><td>{label}</td><td>{dlabel}</td>'
                       + ''.join(f'<td>{interval(r,k)}</td>' for k in ('g', 'w', 'r'))+'</tr>')
    t = summary['splits']['test']['methods']
    counts = t[SELECTED]['counts']
    common = t[SELECTED]['domains']['grid']['common_support_error']['overall']
    ns = native['summary']
    status = json.loads(Path(status_path).read_text()) if status_path else {}
    validation_text = html.escape(status.get('validation_text', 'Verificação de integração registrada no PR #21.'))
    visuals = []
    captions = {'test-208114-6-comparison': 'Teste 208114-6 · maior ganho de G na grade', 'test-208114-1-comparison': 'Teste 208114-1 · regressão local', 'validation-01025-6-comparison': 'Desenvolvimento 01025-6 · aumento de W', 'validation-01026-6-comparison': 'Desenvolvimento 01026-6 · maior ganho de G'}
    for p in sorted((ROOT/'build/nextgen-visuals').glob('*comparison*.jpg')):
        visuals.append('<figure>'+embedded(p, p.stem)+'<figcaption>'+html.escape(captions.get(p.stem, p.stem))+'</figcaption></figure>')
    visual_section = ('<h2>Exemplos RGB e casos difíceis</h2><p>Exemplos escolhidos após pontuar, apenas para inspeção. '
                      'Incluem melhora e regressão. Uma saída original por fallback não conta como imagem retificada. '
                      'As oito integrações a 4 px/mm reproduzem exatamente as coordenadas e máscaras da avaliação nos dois domínios. '
                      'Seis saídas são reamostradas e duas preservam o RGB original após rejeição da malha.</p>'
                      + ''.join(visuals)) if visuals else ''
    body = f'''<header><p class="eyebrow">ECG · ENGENHARIA EXPERIMENTAL · 22 OUT 2026</p>
    <h1>Retificação M2</h1><p class="lead">Menos coordenadas erradas, com ganho moderado de cobertura correta em pacientes novos.</p>
    <div class="cards"><div><strong>79,35 → 79,76%</strong><span>cobertura correta no teste</span></div>
    <div><strong>2,86 → 2,43%</strong><span>cobertura errada no teste</span></div>
    <div><strong>64 capturas · 8 pacientes</strong><span>sem sobreposição com a coorte anterior</span></div></div></header>
    <main><h2>O que mudou</h2><p>O novo refinador reduz ruído de localização com um prior projetivo robusto e diferenças locais dos resíduos.
    Mantém os IDs da grade, limita deslocamentos e evita inverter as células observadas. Restrições só conectam IDs adjacentes observados.
    Um teste conservador de curvatura coerente faz o método se abster diante de possíveis dobras. Ele não certifica a forma física do papel.</p>
    <p>A integração usa <code>--refinement-strength</code>. O padrão continua <code>0</code>, a geometria B0.
    O braço experimental usa <code>1.0</code> com proteção de dobra. Detecções originais, candidato e coordenadas efetivas são arquivados separadamente.
    Falha de construção devolve a geometria B0 e registra a decisão.</p>
    <p>A confiança da grade determina elegibilidade geométrica. Ela não apaga o RGB. O sampling usa a imagem original completa,
    e a saída original permanece disponível quando não há suporte suficiente para uma retificação integral.</p>
    <h2>Teste congelado em pacientes novos</h2><p>Foram renderizadas 112 capturas, sem falhas: 48 de desenvolvimento de seis pacientes previamente inspecionados
    e 64 de teste de oito pacientes ausentes dos 1.280 grupos anteriores e das 77 exclusões. A seleção dos pacientes foi congelada antes de observar as imagens.
    A força foi escolhida apenas no desenvolvimento, entre 0, 0,05, 0,15, 0,5 e 1,0, com limites pré-definidos de aumento de erro.
    O teste foi processado com código e configuração congelados. Não houve novo treino.</p>
    <p>Cada paciente contribui com seis capturas torch e duas 3D LLVM. As médias dão o mesmo peso a pacientes e a cada um dos dois backends.
    Portanto, a média abaixo não é uma média simples das 64 imagens. Os geradores e o detector são os mesmos da metodologia anterior.</p>
    <figure>{embedded(assets/'coverage.png', 'Cobertura correta, errada e sem coordenadas no desenvolvimento e teste')}<figcaption>
    Consultas físicas fixas. Correto significa erro geométrico de até 0,5 mm.</figcaption></figure>
    <div class="table"><table><thead><tr><th>Coorte</th><th>Domínio</th><th>Método</th><th>G correto</th><th>W errado</th><th>C retornado</th><th>R = W/C</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
    <p>G e W usam todas as consultas do domínio como denominador. C = G + W. R é a fração errada entre as consultas retornadas.
    Grade usa a extensão visível da grade principal. Papel inclui margens e cabeçalho fisicamente visíveis, sem contar o fundo.
    Para pontuar, os dois braços recebem o mesmo alinhamento de sinais, permutação dos eixos e origem inteira obtido na B0,
    sem ajuste de escala ou cisalhamento. A orientação absoluta automática não é avaliada.</p>
    <div class="table"><table><thead><tr><th>Coorte</th><th>Domínio</th><th>ΔG [95%]</th><th>ΔW [95%]</th><th>ΔR [95%]</th></tr></thead><tbody>{''.join(cis)}</tbody></table></div>
    <p class="small">Diferenças em pontos percentuais. Intervalos descritivos de 20.000 bootstraps pareados, unidade paciente, semente 1973,
    preservando os dois backends e recalculando R como razão agregada. O teste tem oito unidades independentes.
    Intervalos condicionais, sem ajuste de multiplicidade.</p>
    <figure>{embedded(assets/'patients.png', 'Diferenças por paciente em cobertura correta e errada')}<figcaption>Todos os oito pacientes novos, com o mesmo peso por backend.</figcaption></figure>
    <h2>Onde o ganho aparece</h2><p>Nas capturas 3D, G passa de 64,40% para 65,21% e W de 4,99% para 4,14%.
    Nas capturas torch, G passa de 94,3049% para 94,3115%. O intervalo da diferença torch inclui zero.
    O ganho está concentrado em poucos casos 3D. Quatro capturas ganham mais de um ponto percentual de G.</p>
    <p>No teste, {counts['refinement_applied']} de 64 imagens recebem refinamento efetivo. A proteção atua em {counts['crease_guard_triggered']}.
    Há 42 propostas de malha integral em ambos os braços, 21 casos com coordenadas parciais e um sem coordenadas elegíveis.
    Nenhuma tentativa efetiva de refinamento exige fallback de construção no teste.
    A mediana da etapa geométrica passa de 0,187 para 0,338 segundo, sem rede nem sampling, em CPU compartilhada.</p>
    <p class="callout">A cauda de erro grave permanece. O RMSE no suporte comum muda de {fmt(common['B0_rmse_mm_on_same_queries'],5)} para
    {fmt(common['method_rmse_mm'],5)} mm. Ajustar localizações não resolve IDs incorretos, ambiguidade de escala ou toda falha de topologia.
    O caso <code>206852-2</code> continua com RMSE de 14,89 mm, p95 de 35,20 mm e C de 3,49%.
    O caso <code>215864-6</code> permanece com RMSE de 9,82 mm e p95 de 20,07 mm, com C de 64,94%.</p>
    <h2>Controles que podem reprovar a ideia</h2><p>Dez geometrias analíticas independentes produziram 90 avaliações, com B0 e quatro forças,
    com e sem proteção. Nas três sementes de projetiva ruidosa, força 0,15 reduz RMSE de 0,5277 para 0,2757 mm,
    de 0,5966 para 0,3351 mm e de 0,3412 para 0,1343 mm.</p>
    <figure>{embedded(assets/'controls.png', 'Controles de ruído, dobra limpa e dobra ruidosa')}<figcaption>
    Força 0,15 nestes controles. Todos os braços negativos e demais forças permanecem nos dados.</figcaption></figure>
    <p>Na dobra limpa, suavizar sem proteção cria 10% de coordenadas erradas na força 0,15 e 22,67% na força 1, a partir de B0 exata. A proteção mantém B0 nesse controle,
    nas dobras fortes rotacionadas e na dobra oblíqua. Existe custo: na dobra ruidosa, a proteção se abstém e perde uma redução possível
    de RMSE de 0,2798 para 0,1787 mm. Uma dobra fraca escapa e sofre viés de aproximadamente 0,003 mm.</p>
    <h2>Regressões e cobertura</h2><p>No desenvolvimento, o caso <code>01025-6</code> aumenta W em 2,7344 pontos percentuais no domínio da grade
    e 2,3926 pontos no papel. Esse resultado continua registrado mesmo com melhora da média global.</p>
    <p>No teste, não houve caso acima dos limiares diagnósticos graves previamente definidos.
    Isso não significa ausência de regressões: <code>208114-1</code> perde 0,0488 ponto de G na grade e aumenta W no papel em 0,1953 ponto.
    <code>220288-1</code> perde 0,3906 ponto de G no papel. O CSV e os arquivos de consultas mantêm todos os casos, as cinco forças de desenvolvimento e os dois braços congelados do teste.</p>
    {visual_section}
    <h2>Experimento de resolução nativa</h2><p>A ramificação de tiles nativos foi implementada e testada, mas não promovida.
    Em oito capturas de um único paciente de desenvolvimento, F1 micro passa de {fmt(ns['global']['micro']['f1']*100,4)}% para
    {fmt(ns['roi384']['micro']['f1']*100,4)}%. São 28 verdadeiros positivos adicionais e 26 falsos positivos adicionais.
    A mediana CPU passa de {fmt(ns['global']['latency_seconds']['median'])} para {fmt(ns['roi384']['latency_seconds']['median'])} segundos.
    Quatro imagens usam tiles, quatro retornam ao global pelo orçamento. Esse ensaio pequeno não demonstra melhora geométrica final.</p>
    <h2>Verificação e reprodução</h2><p>{validation_text}</p>
    <p>As 16 capturas do catálogo de cenários foram geradas e inspecionadas. Os controles incluem a paridade da via strength 0,
    as transformações de pixel, os limites de orçamento, a preservação dos IDs e consultas físicas que não são nós usados no ajuste.</p>
    <p>Arquivos publicados no <a href="https://github.com/fmenegui/ecg-torch-renderer/pull/21">PR #21</a>, empilhado sobre os PRs #19 e #20.
    O diretório <code>research/rectification-m2-2026-10-22</code> guarda protocolos, seleção, resumos e hashes.
    Os pacotes dev e novos recompõem o banco. O pacote de medições contém detecções, consultas, controles e resultados, inclusive falhas e regressões.</p>
    <p>As fontes de sinal vêm de <a href="https://physionet.org/content/ptb-xl/1.0.3/">PTB-XL 1.0.3</a>, com proveniência e checksums preservados.
    As coordenadas de referência são usadas para avaliação, sem fornecer geometria à inferência.</p>
    <h2>Decisão de engenharia</h2><p>M2 permanece opcional. O teste novo sustenta uma redução moderada de erro geométrico neste banco sintético.
    Ainda faltam avaliação com outros geradores, fotografias externas e impacto na recuperação do sinal.
    Nenhum concorrente foi executado neste estudo. Estes resultados não demonstram superioridade universal ou estado da arte.</p></main>
    <footer>Experimento M2 · 22/10/2026 · Detector congelado · 14 pacientes no total, 8 novos no teste</footer>'''
    css = '''*{box-sizing:border-box}body{margin:0;background:#f1f4f5;color:#172d43;font:16px/1.65 system-ui,-apple-system,sans-serif}header,main,footer{max-width:1080px;margin:auto;padding:36px 44px;background:white}header{padding-top:54px;border-top:8px solid #087f7b}.eyebrow{font-size:12px;letter-spacing:.14em;color:#087f7b;font-weight:700}h1{font-size:52px;line-height:1.1;margin:18px 0}h2{font-size:25px;line-height:1.3;margin:42px 0 14px}.lead{font-size:23px;max-width:830px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin-top:30px}.cards div{padding:18px;background:#edf7f6;border-radius:8px}.cards strong,.cards span{display:block}.cards strong{font-size:23px}.cards span{font-size:13px;margin-top:6px}p{max-width:950px}figure{margin:28px 0}img{display:block;width:100%;height:auto}figcaption,.small{font-size:13px;color:#526575}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{text-align:right;padding:10px;border-bottom:1px solid #dde5e9}th{background:#edf4f5}td:first-child,th:first-child{text-align:left}.table{overflow:auto;margin:24px 0}code{background:#eef2f4;padding:2px 5px;border-radius:3px;font-size:13px}.callout{padding:20px;border-left:4px solid #ba4a39;background:#fff4ef}a{color:#087f7b}footer{font-size:12px;color:#526575;padding-bottom:45px;border-top:1px solid #dde5e9}@media(max-width:700px){header,main,footer{padding:22px}h1{font-size:40px}.cards{grid-template-columns:1fr}.lead{font-size:20px}}@media print{body{background:white}header,main,footer{max-width:none}.table{overflow:visible}figure,table,.cards{break-inside:avoid}}'''
    out.write_text('<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECG · Retificação M2 · Resultados</title><style>'+css+'</style></head><body>'+body+'</body></html>')
    return {'report': str(out), 'bytes': out.stat().st_size, 'figures': 3+len(visuals)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--status', type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.out, args.status)))
