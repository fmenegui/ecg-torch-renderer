"""Build the evidence report from complete frozen M3 results, without inference."""
import argparse
import base64
import hashlib
from html import escape
import json
from pathlib import Path

import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


METHODS = ['our-B0', 'our-M2', 'npj-generic', 'npj-george-moody-2024', 'kaggle-third']
LABELS = {'our-B0': 'Nossa B0', 'our-M2': 'Nossa M2', 'npj-generic': 'npj Generic',
          'npj-george-moody-2024': 'npj George Moody 2024', 'kaggle-third': 'Kaggle, terceiro lugar'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, digits=2):
    return 'indef.' if value is None else f'{value:.{digits}f}'.replace('.', ',')


def pct(value):
    return 'indef.' if value is None else number(value*100)+'%'


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def figure(path, caption):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return '<figure><img alt="'+escape(caption, quote=True)+'" src="data:image/png;base64,'+data+'"><figcaption>'+escape(caption)+'</figcaption></figure>'


def coverage_plot(part, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.1), layout='constrained', sharex=True, sharey=True)
    for ax, track, title in zip(axes, ('primary', 'operational_RGB'),
                               ('Coordenadas nativas elegíveis', 'RGB efetivamente retificado')):
        values = [part['tracks'][track]['paper']['methods'][m]['aggregation']['overall'] for m in METHODS]
        good = np.array([v['g'] for v in values])*100
        wrong = np.array([v['w'] for v in values])*100
        y = np.arange(5)
        ax.barh(y, good, color='#087f8c', label='G correta')
        ax.barh(y, wrong, left=good, color='#cf533c', label='W errada')
        ax.barh(y, np.maximum(0, 100-good-wrong), left=good+wrong, color='#e6e9ee', label='Sem medição')
        ax.set_yticks(y, [LABELS[m] for m in METHODS], fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('Percentual do domínio fixo de papel')
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    fig.savefig(out, dpi=180, facecolor='white')
    plt.close(fig)


def gallery(execution, bank, schedule, out):
    # Selection uses schedule order/layout, never prediction quality.
    selected = [r for r in schedule if r['group_id'] == schedule[0]['group_id']]
    selected += [r for r in schedule if r['layout'] == '3x4+r1' and r not in selected]
    width, height, header = 440, 310, 60
    canvas = Image.new('RGB', (width*3, (height+header)*2*len(selected)), '#edf0f3')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('DejaVuSans.ttf', 16)
    small = ImageFont.truetype('DejaVuSans.ttf', 13)
    for row_index, row in enumerate(selected):
        with np.load(bank/row['file'], allow_pickle=False) as data:
            source = Image.fromarray(data['image'])
        entries = [('Entrada '+row['key'], row['layout']+' | '+row['backend'], source)]
        for method in METHODS:
            folder = execution/'cases'/row['key']/method
            result = read(folder/'result.json')
            if method.startswith('our-') and 'native_output' in result['metadata']:
                image_path = folder/result['metadata']['native_output']['output_image']
            else:
                image_path = folder/'rectified-native.png'
            native = Image.open(image_path).convert('RGB') if image_path.exists() else None
            status = ('RGB retificado' if result['rgb_resampled'] else
                      'Original preservado' if result['status'] == 'original_preserved_partial_coordinates' else
                      'Falha nativa: '+result['status'])
            entries.append((LABELS[method], status, native))
        for index, (title, subtitle, native) in enumerate(entries):
            x, y = (index % 3)*width, (row_index*2+index//3)*(height+header)
            draw.text((x+14, y+9), title, fill='#18283d', font=font)
            draw.text((x+14, y+33), subtitle, fill='#5b687c', font=small)
            if native is not None:
                native.thumbnail((width-20, height-20), Image.Resampling.LANCZOS)
                canvas.paste(native, (x+(width-native.width)//2, y+header+(height-native.height)//2))
            else:
                draw.text((x+25, y+header+110), 'Nenhuma imagem retificada entregue', fill='#9e4538', font=small)
    canvas.save(out)


def build(args):
    summary, audit = read(args.summary/'summary.json'), read(args.qa)
    receipt = read(args.summary/'postprocessing-receipt.json')
    if audit['status'] != 'pass' or audit['actual_case_method_results'] != 160:
        raise ValueError('A complete independently audited diagnostic panel is required')
    if receipt['actual_case_method_results'] != 160 or summary['scored_cases'] != 32:
        raise ValueError('Incomplete or expanded primary panel')
    if summary['input_sha256']['lock'] != audit['lock_sha256']:
        raise ValueError('QA and summary describe different executions')
    for name, expected in receipt['output_sha256'].items():
        if sha(args.summary/name) != expected:
            raise ValueError('Summary changed after postprocessing: '+name)
    part = summary['splits']['test']
    output = args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    plot = output.parent/'m3-cobertura-papel.png'
    coverage_plot(part, plot)
    schedule = [r for r in read(args.execution/'schedule.json')['cases'] if r['split'] == 'test']
    gallery_path = output.parent/'m3-galeria-predefinida.png'
    gallery(args.execution, args.bank, schedule, gallery_path)
    body = '<p class="eyebrow">ENGENHARIA DE RETIFICAÇÃO · M3 · 23 OUT 2026</p><h1>Uma decisão metodológica melhor exige expor as falhas</h1>'
    body += '<p class="lead">Auditoria por seis agentes, implementações autênticas, controles geométricos independentes e 160 execuções diagnósticas. Não há evidência suficiente para afirmar superioridade geral sobre npj, Kaggle ou Powerful Medical.</p>'
    body += '<div class="note"><strong>O que este relatório decide.</strong> Escala, associação e orientação precisam ser resolvidas antes de aumentar a flexibilidade do mapa. Ausência de grade não autoriza apagar o original. Homografia global, correção local e abstenção têm condições distintas de uso. O protótipo novo permanece experimental.</div>'
    body += markdown.markdown(args.interpretation.read_text(), extensions=['tables', 'fenced_code'])
    body += '<h2>Comparação experimental no mesmo domínio</h2><p>32 capturas sintéticas já inspecionadas do estudo M2, oito pacientes, quatro vistas por paciente e dois backends balanceados. Cada método recebeu o mesmo RGB original. A avaliação usou 4.096 queries de papel e 4.096 de grade por captura, mais 64 anchors disjuntos para remover uma única translação global. Escala, rotação e reflexão nativas permanecem no resultado principal. O limiar de 0,5 mm é um endpoint de engenharia.</p>'
    body += '<p><strong>G</strong> é a fração correta do domínio fixo, <strong>W</strong> a fração errada, <strong>C=G+W</strong> a cobertura e <strong>R=W/C</strong> o risco entre pontos aceitos. R fica indefinido quando C=0. Pacientes e backends têm pesos iguais. Falhas e abstenções permanecem no denominador. O registro por translação é apenas do avaliador, não um estimador autônomo de origem física.</p>'
    captions = {'primary': 'Principal: escala e orientação nativas',
                'secondary_D4': 'Diagnóstico secundário: D4 e translação',
                'unrestricted_finite_field': 'Campo físico finito sem gate de confiança',
                'operational_RGB': 'Operacional: RGB realmente reamostrado'}
    body += figure(plot, 'Papel como domínio principal. A barra cinza inclui ausência de medição, falha ou abstenção. O campo RGB é avaliado apenas na pegada inversa válida e não certifica preservação do traçado.')
    for track in captions:
        block = part['tracks'][track]['paper']
        body += '<h3>'+captions[track]+'</h3>'
        rows = []
        for method in METHODS:
            info = block['methods'][method]
            m = info['aggregation']['overall']
            rows.append([LABELS[method], pct(m['g']), pct(m['w']), pct(m['c']), pct(m['r']),
                         info['registration_unavailable_cases'], info['undefined_R_cases']])
        body += table(['Método', 'G ↑', 'W ↓', 'C', 'R ↓', 'Sem registro /32', 'R indef. /32'], rows)
        if track == 'primary':
            body += '<p>Uma geometria nativa correta apenas após D4 ainda não resolveu orientação automática. Os números M2 antigos usavam outro gauge e outro painel e não são comparáveis diretamente a esta tabela.</p>'
        elif track == 'secondary_D4':
            body += '<p>O avaliador escolhe D4 usando apenas os anchors, fora das queries de score. Essa ajuda de referência serve para localizar erros de orientação. Não é um resultado operacional nem um critério para escolher o vencedor principal.</p>'
        elif track == 'unrestricted_finite_field':
            body += '<p>Remove a confiança geométrica dentro da pegada nativa explícita. Na falta do registro principal pode usar outra translação, declarada nos arquivos por caso. Não inventa geometria fora do crop. No Kaggle, o inversor exclui células não convexas e posições ambíguas que o RGB original ainda pode amostrar.</p>'
        else:
            body += '<p>Quando o sistema só preserva o original ou devolve coordenadas parciais, G=W=C=0 nesta tabela. Quando entrega RGB, usa o campo físico finito na pegada válida. Isso mede geometria associada à saída, sem certificar texto, spikes, amplitude ou fidelidade de waveform.</p>'
    body += '<h3>Incerteza descritiva da comparação principal</h3>'
    rows = []
    for method in METHODS:
        b = part['tracks']['primary']['paper']['bootstrap']['methods'][method]
        interval = b['delta_95_descriptive_interval']['g']
        rows.append([LABELS[method], number(b['delta_vs_baseline']['g']*100)+' pp',
                     '['+number(interval['lower']*100)+'; '+number(interval['upper']*100)+'] pp'])
    body += table(['Método versus B0', 'ΔG', 'Intervalo descritivo de 95%'], rows)
    body += '<p>20.000 reamostragens pareadas por paciente, seed 1973, com todos os backends preservados. Oito pacientes e uma família de renderização dão intervalos condicionais ao painel. Não são teste confirmatório de superioridade, não têm ajuste de multiplicidade e não autorizam uma conclusão sobre a população clínica.</p>'
    body += '<h3>Formato previamente declarado como compatível com Kaggle</h3>'
    rows = []
    for method in METHODS:
        subset = part['tracks']['primary']['paper']['methods'][method]['predeclared_Kaggle_supported_layout']
        m = subset['metrics']
        rows.append([LABELS[method], subset['cases'], subset['patients'],
                     pct(m['g']) if m else 'N/A', pct(m['w']) if m else 'N/A', pct(m['c']) if m else 'N/A'])
    body += table(['Método', 'Capturas 3×4+r1', 'Pacientes', 'G', 'W', 'C'], rows)
    body += '<p><strong>O estrato compatível contém apenas uma captura de um paciente.</strong> Não permite avaliar desempenho dentro da distribuição de origem do Kaggle. Foi declarado antes da inferência e conserva todas as falhas. É descritivo, com backends disponíveis por paciente, sem intervalo bootstrap próprio. Compartilhar formato não equivale a compartilhar distribuição de fonte, impressão, cor, captura ou treinamento.</p>'
    body += '<h3>Domínio de grade, secundário</h3>'
    rows = []
    for method in METHODS:
        m = part['tracks']['primary']['grid']['methods'][method]['aggregation']['overall']
        rows.append([LABELS[method], pct(m['g']), pct(m['w']), pct(m['c']), pct(m['r'])])
    body += table(['Método', 'G', 'W', 'C', 'R'], rows)
    body += '<h3>Entrega de saída e custo em CPU</h3>'
    rows = []
    for method in METHODS:
        d = part['method_diagnostics'][method]
        t, wall = d['native_inference_and_RGB_seconds'], d['study_wall_seconds_including_query_evaluation']
        rows.append([LABELS[method], str(d['RGB_actually_resampled'])+'/32',
                     number(t['median'])+' / '+number(t['p95'])+' ('+str(t['count'])+')',
                     number(wall['median'])+' / '+number(wall['p95']),
                     ', '.join(k+': '+str(v) for k, v in d['status_counts'].items())])
    body += table(['Método', 'RGB entregue', 'Pipeline s: mediana/p95 (n)', 'Wall s: mediana/p95', 'Estados'], rows)
    body += '<p>Processador AMD EPYC 9V74, contêiner com cota de oito CPUs e 8 GiB, máquina compartilhada. Dois threads Torch/OpenMP, um OpenCV e um OpenBLAS, inferência serial e modelos já carregados. B0/M2 compartilham detecção, mas o custo completo do detector é acrescentado a cada estimativa de pipeline. O wall inclui consultas do avaliador e reflete esse compartilhamento. Não são latências de cold start, benchmark de GPU ou medidas de pico de memória. I/O nativo difere e houve trabalho analítico concorrente. Prazos cooperativos podem exceder 180 s dentro de chamadas nativas.</p>'
    body += '<h3>Caudas que a média não resolve</h3>'
    rows = []
    for method in METHODS:
        worst = part['tracks']['primary']['paper']['methods'][method]['worst_finite_tail'][:3]
        for r in worst:
            rows.append([LABELS[method], r['key'], r['layout'], number(r['rmse_mm']),
                         number(r['p95_mm']), pct(r['c']), r['status']])
        if not worst:
            rows.append([LABELS[method], 'Sem campo finito registrado', '', 'indef.', 'indef.', '', ''])
    body += table(['Método', 'Caso', 'Formato', 'RMSE mm', 'p95 mm', 'C', 'Estado'], rows)
    body += '<p>Erros são condicionais ao subconjunto finito retornado. Casos sem saída não recebem RMSE zero. Todas as 1.280 linhas método/caso/trilha/domínio, regressões, suporte comum, contagens e intervalos estão nos arquivos de medições.</p>'
    body += '<h3>Galeria com seleção anterior aos scores</h3>'
    body += figure(gallery_path, 'Quatro vistas do primeiro paciente do schedule e a única captura do formato 3×4+r1, incluindo todos os métodos. Seleção por ordem/formato, sem escolher sucesso ou falha. Miniaturas preservam o aspecto dos rasters nativos. São saídas reais deste experimento, não desenhos gerados nem demonstração de generalização.')
    body += '<h2>O que significa “apagar regiões sem grade”</h2><p>A revisão do código e os controles distinguem perda de evidência, falta de coordenadas e perda efetiva de RGB. Selecionar só o maior componente pode descartar pontos válidos de outra região. Um mapa parcial pode omitir margem e cabeçalho. Já um RGB retificado pode preencher uma área sem suporte, extrapolar errado ou recortar conteúdo. Manter uma cópia do arquivo original protege a origem, mas não prova fidelidade da imagem reamostrada.</p><p>A decisão é preservar o RGB original, exportar separadamente evidência observada, interpolação, extrapolação e elegibilidade, e impedir que ausência de grade vire uma coordenada sentinela válida. A avaliação principal inclui regiões sem grade. Retenção de traços, texto e spikes continua sendo um endpoint próprio ainda não certificado aqui.</p>'
    body += '<h3>Perda real reproduzida no amostrador atual</h3><p><strong>A preocupação com conteúdo perdido tem fundamento.</strong> A máscara de confiança da grade não é usada pelo CLI para apagar RGB. Mesmo assim, o amostrador bilinear real, sem pré-filtro, eliminou completamente uma linha preta de 0,1 mm em duas de cinco posições testadas a 4 pixels/mm. A geometria era afim exata e a linha estava fora da região confiável. Em 8 pixels/mm, todas as cinco posições conservaram algum contraste. Linhas de controle de 1 mm mantiveram contraste total em ambas as densidades.</p><p>São 44 verificações de um controle de componente. Não medem frequência nas fotos nem demonstram que 8 pixels/mm bastam em qualquer caso. O mecanismo é subamostragem dependente da fase, distinto da máscara ou da inferência geométrica. A correção deve combinar integração de área, densidade e um gate de retenção de traços. A comparação M3 congelada mantém o amostrador original para não trocar o método depois de ver resultados.</p>'
    body += figure(args.qa.parent/'sampling-stroke-contact-sheet.png', 'A mesma janela física em todas as colunas, sem ajuste de contraste: original, saída nativa a 4 pixels/mm e controle a 8 pixels/mm. A segunda linha demonstra desaparecimento completo por subamostragem. Imagens do controle real, visualizadas por vizinho mais próximo.')
    sampling = read(args.qa.parent.parent/'sampling/results/results.json')
    body += '<h3>Candidato separado: integrar a área na mesma resolução</h3><p>Um protocolo novo, congelado antes dos scores, comparou sete braços em 280 cenas: dois mapas afins exatos, duas espessuras, sete orientações e dez fases. O baseline chama o amostrador real a 4 pixels/mm. Os candidatos integram 2×2, 4×4 ou 8×8 subpontos físicos diretamente no RGB original, com reconstrução bilinear ou constante por célula. A referência é a área analítica de uma faixa ideal em cada pixel, com formação e quantização explícitas da imagem-fonte.</p>'
    sample_rows = []
    for r in sampling['summaries']:
        if r['width_mm'] == .1:
            sample_rows.append([r['arm'], str(r['disappearances'])+'/140', number(r['mean_pixel_MAE'], 6),
                                pct(r['mean_abs_relative_energy_error']), pct(r['worst_abs_relative_energy_error'])])
    body += table(['Braço, todos a 4 pixels/mm', 'Linhas que somem', 'MAE pixel [0,1]', 'Erro médio absoluto de energia', 'Pior erro de energia'], sample_rows)
    body += '<p>Na reconstrução bilinear, as integrações eliminaram os quatro desaparecimentos observados. A integração 2×2 reduziu o erro médio absoluto de energia de 16,93% para 1,18%, com MAE por pixel menor que nas integrações 4×4/8×8. A 8×8 chegou a 0,159% de erro de energia, mas teve MAE maior que 2×2. O controle boxhold 2×2 ainda perdeu duas linhas. Nenhum braço vence monotonamente em todos os endpoints.</p><p>A revisão independente passou 32.558 verificações, incluindo integração por recorte de polígonos, formação-fonte e todos os 1.960 scores. Existem 78 empates numéricos de quantização com diferença de um nível de uint8 entre as referências independentes, declarados nos recibos. A média é dos códigos RGB sintéticos: não houve conversão radiométrica sRGB. Não se infere preservação de radiância, fidelidade clínica ou recuperação de informação perdida na entrada.</p><p>O protótipo pré-calcula mapas e precisa ser executado em blocos para produção. Um cache completo 8×8 em 16 milhões de pixels usa aproximadamente 8,2 GB só para coordenadas. Preparar mapas e aplicá-los são custos distintos, sem alegação de speedup contra o baseline completo. Mais integração também acrescenta blur de reconstrução. O candidato não alterou o M3 nem o padrão de produção.</p>'
    body += figure(args.qa.parent.parent/'sampling/results/contact-sheet.png', 'Cenas de amostragem selecionadas no protocolo: referência e todos os sete braços. Mesmo tamanho físico e pixels reais ampliados por vizinho mais próximo. As imagens não são resultados dos modelos de retificação no banco.')
    body += '<h2>Comparação metodológica antes dos resultados</h2>'
    method = args.methodology.read_text()
    method = method[method.index('\n')+1:]
    # Keep the signed pre-result methodological review distinct from new scores.
    for level in range(5, 0, -1):
        method = method.replace('\n'+'#'*level+' ', '\n'+'#'*(level+1)+' ')
    body += markdown.markdown(method, extensions=['tables', 'fenced_code'])
    body += '<h2>Todos os controles do candidato, incluindo regressões</h2>'
    body += figure(args.controls_figure, '12 cenas × 5 braços, 4.096 queries por cena. São controles 2D analíticos de componentes com observações iguais, sem detector, RGB ou execução de produto comercial. Valores ≈0 representam precisão numérica, sem significado clínico. Mais flexibilidade não corrige IDs errados e pode ajustar ruído.')
    body += '<h2>Auditoria e reprodução</h2><p>A revisão independente confirmou '+str(audit['checks'])+' verificações dos 160 recibos diagnósticos, linhagem de entrada/pesos/queries/predições, replay de todos os scores, registro e métricas principais recomputados separadamente, orçamento de pixels e igualdade RGB dos fallbacks. Os controles adicionais incluem 1.939 verificações de restauração/oclusão, 763 dos operadores de células e 831 da composição. Não foram executados novos treinamentos.</p>'
    body += '<p>Uma exportação de fallback estava truncada e foi detectada na decodificação completa. Seus bytes defeituosos foram preservados para auditoria. A entrega foi recuperada atomicamente pela outra cópia nativa, previamente verificada como idêntica à entrada. O recibo de recuperação confirma que lock, predições, resultados, tempos e estados não mudaram. O PASS final é posterior à recuperação e não apaga a falha original de exportação.</p>'
    body += '<p>O conjunto de consultas usa suporte material exportado pelo renderer, com remoção de oclusões opacas registradas. Não é metrologia independente de uma câmera real. Nosso detector foi treinado com a mesma família de renderer usada neste diagnóstico. O resultado pode favorecer essa distribuição. A sanidade positiva do Kaggle usou uma imagem dos autores, já normalizada, e retornou todos os 2.508 nós e uma saída 4400×1700. Isso valida execução positiva, sem substituir o teste de novas fotografias.</p>'
    body += '<p>NPJ foi executado com pesos oficiais e dois presets separados. O método principal publicado é projetivo. O braço Kaggle executa o terceiro lugar, com adaptações declaradas para CPU float32 e assets externos. Dotter/Gridder comercial não foi executado: acesso autorizado e exportação geométrica não estavam disponíveis. Sua ausência é N/A e não um resultado zero. SNR/PCC publicados medem cadeias completas e não são comparados diretamente com erro geométrico em mm.</p>'
    body += '<p>Estado aleatório compartilhado de OpenCV/NumPy não foi uniformemente reinicializado por método no runner histórico. A ordem está registrada. Não se afirma replay bit a bit sob retomada ou reordenação. Sucesso de carga de modelo não tem duração registrada e não foi medida RAM por método.</p>'
    body += '<p><a href="'+escape(args.pr_url, quote=True)+'">Código, protocolo e dossiê no PR</a>. O pacote de medições contém os recibos por caso, predições, geometrias e saídas nativas, consultas, controles, QA, resumo e CSV. Pesos e código externo não foram redistribuídos.</p>'
    body += '<p class="hash">Lock SHA256: '+audit['lock_sha256']+'<br>Scorer SHA256: '+summary['source_sha256']['metric_core']+'<br>Postprocessor SHA256: '+summary['source_sha256']['summarizer']+'</p>'
    css = '''body{margin:0;background:#edf0f4;color:#202b3b;font:17px/1.65 system-ui,sans-serif}main{max-width:1180px;margin:30px auto;background:white;padding:48px 56px;box-shadow:0 8px 35px #17304812}h1{font-size:43px;line-height:1.13;max-width:900px;margin:10px 0 24px;color:#112641;letter-spacing:-1.3px}h2{font-size:28px;line-height:1.25;margin:58px 0 18px;border-top:2px solid #dce5ed;padding-top:22px;color:#173859}h3{font-size:21px;margin-top:35px;color:#174963}.eyebrow{font-size:12px;font-weight:750;letter-spacing:2px;color:#087f8c}.lead{font-size:22px;color:#3c5269}.note{background:#edf7f8;border-left:4px solid #087f8c;padding:20px 24px;margin:25px 0}.table-wrap{overflow-x:auto;margin:18px 0 24px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.5}th{text-align:left;background:#143650;color:white;padding:12px 11px;vertical-align:top}td{padding:11px;border-bottom:1px solid #dfe6ec;vertical-align:top}tr:nth-child(even){background:#f6f8fb}td:first-child{font-weight:650}figure{margin:28px 0}figure img{width:100%;height:auto}figcaption{font-size:13px;line-height:1.5;color:#53677d;margin-top:12px}a{color:#096b87}code{font-size:.85em;background:#f1f4f7;padding:2px 4px;word-break:break-word}pre{overflow:auto}.hash{font:11px/1.8 monospace;overflow-wrap:anywhere;color:#64798a}li{margin-bottom:10px}@media(max-width:750px){main{padding:24px 18px;margin:0}h1{font-size:32px}.lead{font-size:19px}table{font-size:12px}}@media print{body{background:white}main{max-width:none;margin:0;padding:0;box-shadow:none}h2,h3{break-after:avoid}tr,figure{break-inside:avoid}a{color:inherit}h1{font-size:32px}}'''
    output.write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Retificação ECG: comparação metodológica e experimental M3</title><style>'+css+'</style><main>'+body+'</main></html>')
    print(json.dumps({'report': str(output), 'sha256': sha(output), 'qa_checks': audit['checks']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('execution', 'bank', 'summary', 'qa', 'methodology', 'interpretation', 'controls-figure', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--pr-url', default='https://github.com/fmenegui/ecg-torch-renderer/pull/22')
    build(p.parse_args())
