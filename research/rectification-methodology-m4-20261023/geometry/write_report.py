"""Render the fixed-result report, without changing methods or selecting models."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
r=json.loads((ROOT/'results-guard-v2/results.json').read_text())
s=json.loads((ROOT/'sparse-results/results.json').read_text())
check=json.loads((ROOT/'guard-v2-retest.json').read_text())
methods=['fixed-H-LS','current-direct-M2-cells','old-unregularized-composition-LS','selected-input-only']
rows=[]
for c in r['cases']:
 def f(x): return 'rejeitado' if x is None else f'{x:.6g}'
 m=c['methods']; selected=m['selected-input-only']
 rows.append('| '+c['scene']['name']+' | '+' | '.join(f(m[k]['rmse_mm']) for k in methods)+' | '+f(selected['W']*100)+' | '+f(selected['C']*100)+' |')
coverage=[]
for c in r['cases']:
 m=c['methods']['selected-input-only']
 coverage.append('| '+c['scene']['name']+' | '+str(m['returned_outside_reference_support'])+' | '+str(m['exterior_returned'])+' |')
costrows=[]
for c in s['costs']:
 if 'skipped' in c:
  costrows.append(f"| {c['nodes']} | denso | não executado, limite | não executado |")
 else:
  costrows.append(f"| {c['nodes']} | {c['implementation']} | {c['median_four_irls_solve_seconds']:.6f} | {c['peak_process_rss_mib']:.2f} |")
report = r'''# Retificação M4: seleção geométrica a partir das observações

O protótipo executa uma decisão entre H global e uma correção contínua regularizada. Ele reduz erros em planos ruidosos e em algumas deformações suaves. **Não domina todos os casos.** Uma quina oblíqua limpa piorou, uma cena planar com outliers foi rejeitada integralmente e erros coerentes de escala e origem passaram com W=100%. Nada aqui estabelece superioridade sobre npj, Kaggle, Dotter ou Gridder.

Foram usados 19 controles analíticos com novas sementes de famílias que já vimos no M3. Não são novos exames reais nem um holdout clínico. Não houve nova inferência neural, saída RGB, alteração do renderer ou mudança de produção.

## Método executável e seleção

A interface `fit_select(observed, ids, allowed_cells, pitch_mm=5)` recebe somente pontos observados, IDs inteiros fornecidos, células permitidas e pitch assumido. Não existe argumento de gold. O pitch, a origem e a orientação dos IDs ficam explicitamente não verificados.

Para cada H ajustado por LS ou RANSAC, normalizamos os pontos observados como z=H(x). No reticulado de IDs, p=5*(coluna,linha), ajustamos d≈z−p com uma penalidade de diferenças de segunda ordem horizontal, vertical e mista. São quatro atualizações IRLS, com peso min(1,1mm/erro). A família fixa contém H global e λ em `[30, 3, 0.3, 0.03]`, com shrinkage α em `[0.5, 1]`, para cada estimador de H. São 18 candidatos.

O mapa S leva p a p+αd(p), por triângulos que compartilham seus vértices. A correção T é a inversa desse mesmo complexo. A retificação é **T∘H**, e a inversa usa H⁻¹∘S. Isso preserva continuidade C0 e evita o viés de reinterpolar uma homografia como se fosse afim em cada célula. Não oferece continuidade C1 na quina.

A seleção usa quatro folds em blocos de 3×4 IDs. Em cada fold, os pontos fonte retidos não participam do ajuste de H nem do campo residual. A base do campo é definida pelos IDs, e não pelas coordenadas retidas. O erro validado é a previsão de z no ID retido, com loss média min(erro,2mm)². Aplicamos a regra de um erro padrão na ordem de complexidade congelada. O residual tem vértices latentes em lacunas, mas **o suporte de saída não atravessa lacunas**. É a interseção entre as células observadas completas e o complexo ajustado.

O protocolo foi congelado antes da primeira execução, SHA `a3ae872d70b34206a8ff809bca10046aa1d3adc81b6c9d9765f50506ac544fe4`. Todas as decisões das 19 cenas foram salvas em `decisions-before-physical-scoring.json` antes de gerar as queries de referência. Todos os 18 candidatos continuam nos resultados, incluindo os rejeitados. A referência analítica serve para medir o resultado final, nunca para escolher a ramificação por imagem.

## Resultado completo

RMSE em mm, condicionado aos pontos retornados. W e C são percentuais sobre as 4.096 queries físicas, sem renormalizar pela cobertura. G+W=C. R=W/C está no CSV e no JSON. Queries de alias de escala e origem passam pela transformação física explicitamente declarada nesses controles, enquanto o seletor continua assumindo os IDs de 5mm. Estes scores analíticos usam coordenadas absolutas, sem registro por translação. O controle de origem não é comparável diretamente ao score M3 que permite escolher a translação global.

| Cena | H fixo LS | M2 direto atual | composição antiga LS | selecionado | W selecionado % | C selecionado % |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows) + r'''

O arquivo `results-guard-v2/measurements.csv` contém G, W, C, R, RMSE, p95, p99, máximo e comparação no suporte comum para os 76 braços. Os arrays guardam todas as previsões, máscaras, status, pontos físicos e os 18 candidatos por cena.

Ganhos observados: o plano com ruído alto passou de 0,26844mm no M2 para 0,05053mm. A deformação suave com ruído passou de 0,20269 para 0,15546mm. Com outliers sobre a deformação suave, 0,55187 para 0,16500mm, com cobertura de 98,32% no M2 e 97,41% no selecionado. No suporte comum, o M2 tem 0,48811mm e o selecionado 0,16500mm. Este ganho não autoriza ignorar a perda de cobertura.

Negativos que afetam a decisão metodológica:

- Na quina oblíqua limpa, M2=0,09217mm, composição antiga=0,08020mm e selecionado=0,10747mm. W do selecionado=1,2207%, contra 0,2686% no M2 e 0,3662% na composição antiga. Regularizar arredonda uma quina real e a loss de validação não garantiu a melhor geometria física.
- Na quina alinhada limpa, a composição antiga alcança 0,00112mm, contra 0,02951mm do selecionado. O selecionado ainda melhora o M2, mas perde para a composição sem regularização neste caso.
- A cena planar com outliers foi rejeitada inteira: C=0 e R indefinido. A malha observada contém sobreposição. O guard conservador não isola apenas as células problemáticas, e não escolhemos outro ramo depois de olhar o gold.
- O erro coerente por linha tem W=72,75% e RMSE=1,2304mm. O erro coerente na metade tem W=79,54% e RMSE=4,4453mm. Regularizar não valida os IDs.
- O alias de pitch 10mm observado e 5mm assumido tem RMSE=53,94mm e W=100%. A origem deslocada em (10,5)mm tem RMSE=11,18mm e W=100% na coordenada absoluta declarada pelo controle. A loss de CV fica próxima de 10⁻¹⁴mm² em ambos. A origem global é um grau de liberdade quando não existe referência absoluta: permitir registro por translação elimina esse erro de origem. Portanto esse controle demonstra uma ambiguidade de coordenadas, e não um defeito de forma. O erro de pitch permanece mesmo quando se permite escolher a origem. Uma excelente previsão de IDs coerentemente errados não certifica a escala física.
- Na superfície suave limpa, a composição antiga tem 0,05509mm e o selecionado 0,05892mm. O benefício do denoising não é gratuito.

## Guardas e a falha encontrada pela QA

A primeira versão verificava horizonte, degeneração, orientação dos triângulos e sobreposição de área positiva. A QA independente construiu uma faixa retangular que fecha exatamente uma volta em anel. Dois IDs de fronteira distintos coincidiam no espaço fonte, sem sobreposição de área positiva. A versão inicial aceitou um ponto ambíguo e falhou. O recibo está em `m4/qa/receipts/non-simple-boundary-before-fix.json`, e a fonte original foi preservada.

A correção guard-v2 rejeita também vértices distintos coincidentes e contato ou cruzamento de arestas de fronteira não incidentes. O guard de sobreposição continua presente. A revisão foi registrada antes do reteste em `guard-revision-v2.json`. O anel passa a ser rejeitado. **Essas verificações são exclusões computacionais conservadoras em um complexo limitado, não um certificado físico de uma folha fotografada.**

O mesmo experimento de 19 cenas e 76 braços foi repetido. Nenhum nome selecionado, máscara, coordenada salva ou score mudou. Os 342 candidatos conservaram o estado aceito/rejeitado, embora alguns motivos de rejeição apareçam antes na nova guarda. `guard-v2-retest.json` registra esses motivos. As fontes e resultados da primeira execução permanecem em `geometry_selector_v1.py`, `run_experiment_v1.py` e `results/`.

A QA independente auditou os 76 scores e 342 candidatos com 6.279 verificações. Também rejeitou anéis de uma e duas voltas e preservou os controles de horizonte, lacuna, exterior e 173 limites de aresta. O recibo está em `m4/qa/receipts/geometry-result-audit.json`. A aprovação é condicional ao escopo testado da formulação densa e guard-v2. A extensão esparsa não recebeu auditoria integral independente, e os seus resultados de paridade e performance são verificação separada desta frente.

Nos casos aceitos, a maior descontinuidade numérica de fronteira foi 2,93×10⁻¹⁴mm e o maior erro de ida e volta foi 4,55×10⁻¹³px. Os planos projetivos sem ruído ficaram abaixo de 1,03×10⁻⁸mm. Os controles verificam colinearidade, horizonte, área sobreposta, ambiguidade interior, contato de fronteira, lacuna central e exterior estrito do suporte observado.

## Suporte físico e status

As coordenadas, o status e a incerteza são saídas separadas. A dispersão entre candidatos e o erro preditivo dos folds são diagnósticos **não calibrados**, sem interpretação de probabilidade ou limite de erro físico. Um ponto retornado não recebe o rótulo de identidade ou escala confirmada. Um ponto fora da malha não autoriza apagar pixels de uma imagem. Não há renderer RGB neste estudo.

O exterior estrito da própria malha observada teve zero retornos. O exterior físico de referência pode ter retornos porque os pontos observados ruidosos ou corrompidos podem deslocar a fronteira da malha para fora do papel físico. Isso também ocorre perto das bordas de lacunas. Os retornos físicos exteriores são preservados abaixo, e não removidos por uma máscara de gold.

| Cena | Retornos fora do suporte físico observado de referência | Retornos nas 256 queries do exterior físico |
|---|---:|---:|
'''+ '\n'.join(coverage) + r'''

## Custo e implementação esparsa idêntica

A primeira implementação usa matrizes densas, memória quadrática e fatorizações cúbicas. Seu limite duro de 2.000 nós retangulares é um limite de implementação, não uma propriedade matemática do método. Em 192 nós, a seleção completa com todos os candidatos levou mediana de 0,606s em CPU nesta máquina. Esse número não inclui detector, RGB ou fluxo de produção.

Congelamos um protocolo separado, depois do estudo denso e antes da execução esparsa, SHA `348d586f1f583f331ec7891faeb36bad08ff69f18912133c2a434f4c56b56d1b`. A versão esparsa monta exatamente os mesmos stencils e resolve a mesma equação IRLS. Não troca regularização, folds, candidatos, limiar ou regra de decisão. A paridade foi repetida com guard-v2 nas 19 cenas e nos 18 candidatos. Todos os nomes selecionados, estados aceito/rejeitado e máscaras coincidiram. A maior diferença de loss de CV foi 2,41×10⁻¹³, de campo 1,50×10⁻¹²mm, de coordenada 1,81×10⁻¹²mm e de inversa 8,59×10⁻¹²px.

Tempos abaixo são a mediana de três ajustes de **um campo**, cada ajuste com quatro soluções IRLS. Incluem só esse componente. RSS é o pico do processo, incluindo imports. São medições locais de CPU, sem isolamento de toda carga concorrente da máquina.

| Nós | Implementação | Um ajuste IRLS4, s | Pico RSS, MiB |
|---:|---|---:|---:|
'''+ '\n'.join(costrows) + r'''

Em 1.855 nós, a montagem da penalidade densa ocupa 27,53MB, contra 0,286MB esparsos. Em 7.245 nós, a penalidade esparsa usa 1,14MB. A matriz normal esparsa tem poucos vizinhos por nó, mas sua fatorização tem fill-in. O custo de toda a seleção inclui 40 ajustes de campo, ajustes de H, 18 refits com guardas e queries. Não multiplicamos a medição de um componente e a apresentamos como runtime fim a fim.

O inventário somente de metadados encontrou 33 arquivos de observações congeladas no diretório existente, incluindo o arquivo `01024-0`, além do censo de 32 casos do M3. É um censo de arquivos, não 33 exames reais novos. As extensões retangulares vão de 15 a 7.350 nós. São 24 abaixo do limite denso e 33 abaixo do limite esparso de 20.000. Nenhuma seleção completa foi executada nesses arquivos e nenhum benchmark físico real foi refeito. O guard de sobreposição ainda tem pior caso quadrático no número de triângulos, e a adaptação esparsa ainda não é thread safe. O limite de 20.000 é uma proteção, não uma promessa de latência ou robustez.

## Decisões para a próxima metodologia

O que este experimento sustenta é incorporar **seleção condicional à confiança dos IDs** e preservar um H global como candidato explícito para denoising de planos. Ele também sustenta compor uma correção contínua com H e usar a mesma malha nas duas direções. Esses mecanismos não devem virar promoção automática para produção.

Antes de avançar, faltam hipóteses de quina que preservem descontinuidade de derivada, quarentena local de observações e células em vez de rejeição global, avaliação de incerteza calibrada, teste de blocos e lacunas maiores e desempenho integral em malhas de tamanho real. A consistência de IDs, pitch, orientação e origem exige evidência adicional ou abstinência, porque não é identificável nestes controles apenas com a grade observada. Usar informação independente de escala e identidade deve ser uma camada explícita, com resultados condicionais quando ela faltar.

A comparação desta geometria com os componentes de npj, Kaggle e produtos comerciais exige correspondências de entrada idênticas ou decomposição explícita do efeito do detector. Este estudo não executa esses métodos e não estabelece ranking entre pipelines.

## Reprodução e integridade

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 swarm-venv/bin/python m4/geometry/run_experiment.py --out m4/geometry/results-guard-v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 swarm-venv/bin/python m4/geometry/check_results.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 swarm-venv/bin/python m4/geometry/sparse_parity_cost.py
```

As fontes, os protocolos, a implementação M2 importada e a composição M3 importada são identificados por SHA256 nos JSONs. Nenhuma fonte congelada do M2/M3 foi modificada. O script verificador compara as duas execuções, os arrays de todos os candidatos, as métricas e os guardas. O script esparso verifica paridade e custo sem parâmetros derivados do gold.
'''
(ROOT/'analysis.md').write_text(report)
print(ROOT/'analysis.md')
