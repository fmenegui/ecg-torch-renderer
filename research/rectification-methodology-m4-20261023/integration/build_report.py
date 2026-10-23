"""Build a self-contained Portuguese report from retained M4 evidence."""
from __future__ import annotations
import argparse
import base64
import html
import json
from pathlib import Path
import markdown

ROOT=Path(__file__).resolve().parents[2]
M4=ROOT/'m4'

def load(path):
    return json.loads((ROOT/path).read_text())

def esc(value):
    return html.escape(str(value))

def num(value, digits=4):
    if value is None:
        return 'rejeitado / N/A'
    if value and abs(value)<1e-6:
        return f'{value:.2e}'
    return f'{value:.{digits}f}'.replace('.',',')

def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join(
        f'<th>{esc(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def image(path, caption):
    data=base64.b64encode((ROOT/path).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'

def md(text):
    return markdown.markdown(text,extensions=['tables','fenced_code'])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=M4/'deliverables/ecg-retificacao-m4-2026-10-23.html')
    parser.add_argument('--pr-url',default='')
    args=parser.parse_args()
    census=load('m4/kaggle/census.json')
    geo=load('m4/geometry/results-guard-v2/results.json')
    parity_path='m4/integration/final-sampling-replay/parity.json'
    parity=load(parity_path if (ROOT/parity_path).exists() else 'm4/sampling/results/parity.json')
    cost=load('m4/sampling/results/final-cost/benchmark.json')
    area=load('m4/sampling/results/axis-area.json')
    ident=load('m4/identifiability/results.json')
    matrix=load('m4/qa/coverage-matrix.json')
    sections=[]
    def section(anchor,title,content):
        sections.append(f'<section id="{anchor}"><h2>{esc(title)}</h2>{content}</section>')
    section('decision','Decisão metodológica',md('''
O candidato mais forte para avançar é **um sistema que mantém hipóteses físicas e escolhe a geometria pela evidência**, com RGB original preservado e reamostragem adequada ao footprint. Não há evidência de que um único interpolador, uma rede maior ou um residual sempre ativo vença todos os casos.

Seis agentes trabalharam em censo Kaggle, fontes e patentes, identificabilidade, geometria, sampling/export e crítica adversarial independente. A rodada produziu código executável e novos resultados condicionais. **Não implementou um detector completo novo nem demonstrou superioridade universal, clínica ou comercial.**

A prioridade é resolver saltos inteiros, pitch, orientação/reflexão e fase entre componentes. Um warp pode ser contínuo, invertível e ter resíduo quase zero com escala física errada. Sem informação suficiente, a saída correta mantém hipóteses ou coordenadas parciais, junto do original.
''')+f'<div class="facts"><div><strong>{len(census["entries"])}</strong>writeups públicos lidos</div><div><strong>{len(matrix["cases"])}</strong>famílias na matriz adversarial</div><div><strong>19 × 4</strong>saídas geométricas principais</div><div><strong>{parity["outputs_compared"]:,}</strong>saídas de sampling com paridade</div></div>')
    section('competitors','Comparação com npj, Powerful Medical e Kaggle',md('''
**npj Digital Medicine:** a implementação principal usa segmentação, linhas Hough, perspectiva global e periodicidade para escala. O TPS de `dewarper.py` é experimental, não formalmente avaliado e está desligado nos presets oficiais. O commit oficial consultado permanece `97a1508`, o mesmo congelado em M3. H global é um baseline essencial sob seu modelo de projeção, com correspondências corretas. Não é um modelo geral de papel curvo.

**Powerful Medical:** Dotter detecta interseções, Gridder associa a rede e infere ausências, Undistortion reamostra células e Leader extrai o sinal. A família de patentes descreve direções, probabilidades, ciclos, seleção de rede e regressão. A descrição pública não identifica inequivocamente homografia versus bilinear, nem o interpolador e a versão comercial atual. **O produto não foi executado.** Nossos controles geométricos não recebem seu nome como se fossem reprodução.

**Kaggle:** não é uma abordagem única. O censo percorreu seis páginas oficiais, com 117 tópicos únicos e 24 entradas `isWriteUp`, e leu as 24. A cobertura é completa para a listagem retornada em 23/10/2026. Não cobre soluções privadas, material apagado ou todo notebook sem writeup. Os mecanismos abaixo vêm das fontes originais, com inspeção de código selecionada. Somente o terceiro lugar teve inferência nativa executada na etapa M3.
''')+table(['Writeup / posição declarada','Família e mecanismo','Sampling / suporte','Reprodução nesta linha de trabalho'],[
        [f'<a href="{esc(e["url"])}">{esc(e["title"])}</a>'+(' <span class="warn">posição no metadata diverge</span>' if e.get('rank_metadata_discrepancy') else ''),
         esc(e['family'])+'<br>'+esc(e['geometry']),
         esc(e.get('rgb_sampling','não descrito'))+'<br>'+esc(e.get('missing_support','não descrito')),
         esc(e['weights_status'])] for e in census['entries']])+md('''
O 5º publica SNR downstream de 21,546 dB para H global, 22,930 para H local K4, 22,846 para spline e 23,087 para H local K16 no fold do autor. Isso justifica reproduzir K16, sem convertê-lo em medida geométrica nossa. O código seleciona uma H por célula e não aplica blending por padrão. Cantos compartilhados não garantem continuidade ao longo da aresta. O contraexemplo da classe gerou diferença de 0,04545 célula, sem ruído de detector. Não é uma medição do checkpoint nativo.

O 15º já documenta correções de canvas, borda e convenção `align_corners` no Heng. Não reivindicamos novidade nessas correções. O M3 preservou a convenção nativa e mediu seu efeito, sem silenciosamente corrigir o competidor.
'''))
    decisions=(M4/'integration/methodological-decisions-pt.md').read_text()
    section('architecture','Arquitetura e decisões entre alternativas',md(decisions.split('## Arquitetura proposta',1)[1]))
    section('geometry','Experimento novo de seleção geométrica',md('''
Dezenove cenas analíticas de desenvolvimento, 18 candidatos por cena e quatro comparadores. A regra usa quatro folds espaciais de observações, perda robusta e preferência predefinida por menor complexidade. Pontos retidos não entram no ajuste. As decisões foram seladas antes das consultas físicas. A seleção recebe pontos, IDs, células e pitch assumido. Não recebe gold.

O modelo compara H global e uma deformação compartilhada regularizada após normalização por H. H local K16, TPS e modelos de crease não entraram neste experimento. Logo não existe uma ordenação experimental nova dessas famílias contra nosso candidato. A família analítica já foi vista no M3, com novos seeds, e não constitui teste externo.

G é a fração do domínio fixo retornada com erro ≤ 0,5 mm, W é retornada com erro maior, C=G+W e R=W/C, indefinido se C=0. O limiar é de engenharia. A tabela não usa ajuste por translação. O controle de origem mede um alvo absoluto declarado e seria removido por um gauge global livre. Ele não prova defeito de forma nem equivale ao registro por translação do M3. O alias de pitch persiste após translação.
''')+table(['Cena','H global RMSE mm','M2 direto RMSE mm','Composição sem regularização RMSE mm','Selecionado RMSE mm','G / W / C selecionado (%)','Candidato'],[
        [esc(c['scene']['name'])]+[num(c['methods'][k]['rmse_mm']) for k in
        ('fixed-H-LS','current-direct-M2-cells','old-unregularized-composition-LS','selected-input-only')]+
        [' / '.join(num(100*c['methods']['selected-input-only'][k],2) for k in ('G','W','C')),
         esc(c['selection']['selected_candidate'])] for c in geo['cases']])+
        image('m4/integration/visuals/geometry-all-cases.png','Todos os casos, com RMSE condicional e cobertura fixa. Rejeição tem C=0 e RMSE indefinido. Origem absoluta é um controle de gauge, sem registro por translação.')+md('''
**Ganhos:** no plano com ruído alto, RMSE caiu de 0,26844 para 0,05053 mm frente às células M2 diretas. Em curva suave com outliers, de 0,55187 para 0,16500 mm, com uma pequena perda de cobertura que permanece na tabela.

**Negativos retidos:** a dobra oblíqua limpa piorou de 0,09217 para 0,10748 mm e W aumentou. Com ruído, o RMSE da dobra oblíqua melhora, mas W também aumenta. Outliers planares provocam rejeição conservadora de todo o complexo, com C=0. Corrupções coerentes de linhas e meia página continuam ruins. O alias de pitch passa pela validação e tem W=100%. A falha coerente de meia página retorna 52/256 queries do exterior físico. Estar dentro do suporte inferido dos nós não prova pertencer à folha física. **Validação interna não verifica identidade ou escala.**

**Correção descoberta por QA:** a primeira guarda aceitou uma faixa fechada em anel com contato de vértices distintos na costura. A versão corrigida rejeita vértices coincidentes e contatos/cruzamentos de arestas não incidentes da fronteira. O original e a falha foram preservados. Reexecutar as 19 cenas manteve exatamente as seleções, máscaras, arrays e 76 scores. Não houve ajuste dos candidatos por gold. Os testes numéricos não são uma prova universal de injetividade de superfícies fotografadas.
'''))
    sparse=load('m4/geometry/sparse-results/results.json')
    section('sampling','Preservação de traços, memória e exportação',md(f'''
A suspeita de apagamento tem um mecanismo reproduzido: amostragem pontual pode perder tinta fina mesmo com geometria exata. Isso não exige uma máscara de confiança apagando o RGB. Também precisamos distinguir crop, remap fora da captura e ausência de geometria defensável.

O novo componente por blocos compõe o mapa inverso diretamente para o RGB original. Centros de pixels, espaçamento por eixo, bordas, suporte e alpha são explícitos. Confiança geométrica não modifica imagem nem alpha. RGBA usa cores premultiplicadas. O PNG é codificado temporariamente, verificado por chunks/CRC e pixels totalmente decodificados, e só então publicado atomicamente.

O replay comparou **{parity['outputs_compared']:,} saídas e {parity['channel_values_compared']:,} valores RGB**, nas seis variantes de área, 280 cenas e dois orçamentos de bloco, com **zero diferenças** contra as predições M3 arquivadas. O domínio é o ROI de 41×41 pixels pontuado no M3. Não é uma afirmação de paridade de todo frame ou de recuperação universal.
''')+image('m4/integration/visuals/thin-trace.png','Exemplo M3 arquivado com geometria exata. A mudança em aparência mostra menor pico, não criação de novo traço.')+
    table(['Método','Subgrid','Mediana total s','Pico RSS MB decimais','Escopo'],[
        [esc(x['method']),str(x['subgrid']),num(x['median_total_seconds'],3),num(x['peak_rss_bytes']/1e6,1),'Fonte 1400×1100 RGB, saída 1024×768'] for x in cost['measurements']])+
    md(f'''
CPU {cost['hardware']['cpu_model']}, OpenCV/BLAS com uma thread, host compartilhado. O tempo inclui preparação dos mapas, conversão, amostragem e coleta, com três repetições. A redução de memória não é um ganho de velocidade: no ensaio 8×8, o componente por blocos levou {num(cost['measurements'][-1]['median_total_seconds'],3)} s. O limite de coordenadas exclui alocações internas de callbacks, fonte e saída retida.

**Contraexemplo novo:** na redução 32:1 de um traço fonte de um pixel, as quadraturas 2×2, 4×4 e 8×8 retornam branco 255, embora a integral exata seja 247,03125. Concordância entre subgrids não certifica antialiasing. Quantização, blur e tinta já perdida na captura continuam limites.

**Controle positivo separado:** integração exata de células fonte sob transformação afim com eixos alinhados retorna 247 nesse caso. Em 60 controles motivados pela falha, os desaparecimentos foram 0 na área exata e 29/17/9 nas grades 2/4/8. Esse braço usa reconstrução constante por célula, diferente da bilinear. Rejeita rotação, shear e mapas não lineares. Não é um antialiaser geral nem um teste independente pós-desenvolvimento.

O solver residual esparso mantém a mesma equação, família e regra de seleção. Passou a comparação declarada nos 19 casos. A medição de um campo a 1.855 nós passou de aproximadamente 0,461 s e 285 MiB para 0,0381 s e 88 MiB. A 7.245 nós, o campo esparso levou 0,136 s. **Isso mede uma solução de campo, não o seletor completo nem o retificador.** Guards e consultas ainda precisam escalar. A extensão esparsa tem verificação de seu implementador e não foi integralmente auditada pela QA independente.
'''))
    section('identifiability','Casos identificáveis, ambíguos e impossíveis sob o modelo',md(f'''
Os {ident['controls_registered']} controles de identificabilidade passaram {ident['assertions_passed']} verificações. PASS confirma a testemunha ou o contrato, e não que o sistema atual corrigiu o problema. Há rasters construídos byte a byte iguais para verdades diferentes, perdas de informação do frontend e controles de restrições explicitamente distintos.

Quatro fiduciais físicos ordenados eliminam 23 das 24 hipóteses de passo/pose no controle positivo. Sem essa evidência, grid quadrado mantém D4, omissões coerentes podem manter ciclos perfeitos e uma ponte escondida pode alterar a fase de outra ilha. Uma dobra branca pode ficar ambígua sob iluminação declarada. Conteúdo oculto não vira observação ao ser preenchido.

A auditoria analítica adicional reconstruiu equações e rasters sem importar o gerador original, com 36 verificações. Foi escrita pelo mesmo agente. Não é revisão externa nem simulador fotográfico independente. QA da rodada tem testes próprios de geometria e sampling, descritos separadamente.
''')+table(['Controle','Nome','Status','Métrica / condição'],[
        [esc(c['id']),esc(c['name']),esc(c['status']),'<details><summary>Ver condições</summary><pre>'+esc(json.dumps(c['metrics'],ensure_ascii=False,indent=2))+'</pre></details>'] for c in ident['results']]))
    section('corners','Matriz adversarial e lacunas',md('''
A matriz contém 48 famílias finitas: 21 com testes finitos de componentes, 13 parciais, seis representadas por testemunhas construtivas revisadas e oito não executadas em M4. Um ataque executado pode ter reproduzido uma falha ainda não corrigida. O catálogo de identificabilidade acrescenta 20 ameaças explicitamente não experimentadas, e as propostas externas têm 24 famílias de falsificação. Esses catálogos se sobrepõem e não devem ser somados como número de testes independentes. Não há alegação de atacar todos os inputs possíveis.
''')+table(['ID','Família / gravidade','Identificabilidade','Gate','Evidência atual'],[
        [esc(c['id']),esc(c['family'])+'<br>'+esc(c['severity']),esc(c['identifiability']),esc(c['acceptance_gate']),esc(c.get('M4_review',{}).get('execution_state',c['current_evidence']))+'<br>'+esc(c.get('M4_review',{}).get('scope_and_remaining_gap',''))] for c in matrix['cases']]))
    prior=(M4/'prior-art/notes.md').read_text()
    external=prior.split('## Mecanismos úteis de dewarping de documentos',1)[1].split('## Casos que devem poder refutar',1)[0]
    section('new-research','Fontes recentes e mecanismos a testar',md(external)+md('''
Nenhum desses novos modelos de documentos foi executado nesta rodada. Código, links de pesos, licença e anotações anunciadas foram separados de disponibilidade verificada. Métricas de OCR ou distorção de documentos não equivalem a mm, tempo, mV ou preservação clínica de ECG. A revisão não constitui busca sistemática de anterioridade.
'''))
    section('benchmark','O que falta para afirmar superioridade',md('''
O M3 continua sendo um painel sintético retrospectivo de 32 capturas de oito pacientes. Nosso M2 obteve G=55,272% no domínio primário e entregou RGB reamostrado em 18/32. O detector conhece a mesma família de renderizador. O estrato compatível com Kaggle tem apenas uma captura. Esses limites não desapareceram com a rodada M4.

O próximo benchmark deve separar: estimadores com observações idênticas, pipelines nativos no mesmo RGB e preservação de imagem/sinal. Deve incluir vários pacientes e fontes em 3×4+r1, layouts amplos e fotografias físicas com metrologia independente. Reservar pacientes, documentos, fabricantes e sessões, não só augmentations. A base Ahus/Emory/PTB-XL já usada pelo npj não vira holdout novo por receber outra publicação.

Manter todos os casos no denominador. Medir G/W/C/R, caudas e cobertura física, queries em área material e pixel visível separadamente, objetos pequenos, waveform com extrator comum e integridade de arquivo. Fixar orçamento, configurações e regra antes da inferência. C=0 não é risco zero. Calibrar incerteza fora do teste. Um produto sem campo geométrico requer contrato explícito de landmarks do raster.

Powerful Medical continua N/A por falta de execução autorizada/acesso adequado nesta tarefa. Nenhuma imagem foi enviada ao fornecedor. Não há vitória experimental nova sobre o produto.
'''))
    validation_path=M4/'integration/validation.json'
    validation=json.loads(validation_path.read_text()) if validation_path.exists() else {'status':'Validação final ainda em fechamento.'}
    section('validation','Verificações, versão e entregas',md('''
Os componentes M4 são offline, fora do hot path de treinamento Torch. As fontes e resultados M2/M3 foram preservados. Uma galeria de regressão do renderer com 16 cenários e múltiplos layouts foi exportada, totalmente decodificada e inspecionada. Ela não representa saídas de um detector M4 nem equivalência com fotografias reais.
''')+'<pre>'+esc(json.dumps(validation,ensure_ascii=False,indent=2))+'</pre>'+
        (f'<p><a href="{esc(args.pr_url)}">Código e testes em PR de pesquisa</a></p>' if args.pr_url else ''))
    sources=load('m4/prior-art/sources.json')['sources']
    section('sources','Fontes e rastreabilidade',table(['ID','Fonte primária','Limite'],[
        [esc(s['id']),f'<a href="{esc(s["url"])}">{esc(s["title"])}</a>',esc(s.get('limit',''))] for s in sources])+
        '<p>As 24 referências Kaggle estão na tabela do censo. O pacote de medições contém protocolos, arrays, scores, hashes e recibos, incluindo falhas retidas. Código, pesos e textos integrais de terceiros não estão nesse pacote.</p>')
    links=[('decision','Decisão'),('competitors','Comparadores'),('architecture','Arquitetura'),('geometry','Geometria'),
           ('sampling','Sampling'),('identifiability','Ambiguidade'),('corners','Corner cases'),
           ('new-research','Pesquisa recente'),('benchmark','Benchmark externo'),('validation','Validação'),('sources','Fontes')]
    css='''*{box-sizing:border-box}body{margin:0;color:#1b2931;background:#f3f5f5;font:16px/1.65 system-ui,Arial,sans-serif}header,main{max-width:1250px;margin:auto}header{padding:60px 34px 24px}.eyebrow{color:#167b86;text-transform:uppercase;letter-spacing:.16em;font-size:12px;font-weight:700}h1{font-size:clamp(30px,4vw,50px);line-height:1.13;max-width:950px;letter-spacing:-.025em}h2{font-size:28px;line-height:1.25;margin:0 0 22px}h3{font-size:21px;margin-top:34px}p{max-width:1050px}a{color:#126774;text-underline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:8px;margin-top:26px}nav a{padding:5px 12px;border:1px solid #cddadd;border-radius:16px;text-decoration:none;font-size:13px}main{padding:0 22px 70px}section{padding:34px;background:white;border:1px solid #dfe7e8;margin:20px 0;border-radius:10px;scroll-margin-top:20px}.facts{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-top:28px}.facts div{padding:18px;background:#f1f7f7;border-left:3px solid #167b86;font-size:13px}.facts strong{display:block;font-size:32px;color:#126774;line-height:1.25}.table-wrap{overflow:auto;margin:24px 0}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.5}th{text-align:left;background:#eaf1f2;font-size:12px}th,td{padding:12px 13px;border-bottom:1px solid #dce5e6;vertical-align:top;min-width:105px}tr:nth-child(even){background:#f8fafa}figure{margin:30px 0}figure img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:#52636b;margin-top:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:18px;background:#f3f6f6;font-size:12px;line-height:1.55;border-radius:6px}code{font-size:.88em}details summary{cursor:pointer;color:#126774}.warn{color:#885323;font-size:11px}footer{max-width:1250px;margin:auto;padding:0 34px 35px;color:#5d6b71;font-size:13px}@media(max-width:720px){header{padding:30px 22px 12px}section{padding:22px 18px}.facts{grid-template-columns:repeat(2,1fr)}main{padding:0 10px 40px}h2{font-size:24px}}@media print{body{background:white;font-size:11pt}nav{display:none}main,header{max-width:none;padding:10px}section{break-inside:auto;border:0;padding:12px 0}.table-wrap{overflow:visible}table{font-size:9pt}a{color:inherit}h2{break-after:avoid}.facts strong{font-size:23px}}'''
    document='<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Retificação ECG M4: decisões e testes adversariais</title><style>'+css+'</style></head><body><header><div class="eyebrow">Pesquisa ECG · M4 · 23 outubro 2026</div><h1>Retificação com evidência física, preservação de conteúdo e falhas explícitas</h1><p>Confronto metodológico de Kaggle, npj Digital Medicine e Dotter/Gridder, com protótipos executados e revisão adversarial.</p><nav>'+''.join(f'<a href="#{a}">{t}</a>' for a,t in links)+'</nav></header><main>'+''.join(sections)+'</main><footer>Estudo de engenharia. Fontes primárias e resultados condicionais são separados de hipóteses e de validação externa ainda não executada.</footer></body></html>'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(document)
    print(json.dumps({'file':str(args.out),'bytes':args.out.stat().st_size,'sections':len(sections)},ensure_ascii=False))

if __name__=='__main__':
    main()
