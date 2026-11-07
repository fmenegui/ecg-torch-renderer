# Digitalizador seletivo, primeira implementação

O objetivo é melhorar a reconstrução de ECGs reais sem perder as amostras que
o extrator direto já recuperou. Esta alteração implementa e testa uma seleção
entre resultados completos. Ainda não estabelece superioridade sobre os
trabalhos publicados.

## O que foi implementado

- Contrato único de sinal com 12 derivações, mV, frequência e origem explícitas,
  mantendo NaNs onde não há reconstrução.
- Extrator npj original com commit e dois checkpoints verificados, catálogo
  completo de layouts e resolução nativa de 3000 pixels na execução em GPU.
- Candidato geométrico com nosso detector selecionado, malha completa,
  verificação de dobras/sobreposições e amostragem do RGB original. A resolução
  vem do Jacobiano estimado e de um limite explícito de memória.
- Candidato especializado com a submissão original de seis modelos do segundo
  colocado no Kaggle, condicionado ao layout inferido 3x4+1R.
- Regras de rejeição sem sinal de referência. A decisão exige ganho de suporte
  e nenhuma perda de amostra direta, além de concordância por derivação.
- CLI experimental, testes físicos independentes, replay auditável e execução
  limitada no Runpod, com limpeza apenas dos recursos criados pela execução.

Não misturamos amostras de extratores diferentes nem corrigimos amplitude ou
tempo do resultado retornado para fazê-lo coincidir com uma referência. Uma
maior quantidade de valores finitos não certifica que esses valores estejam
corretos. As tolerâncias foram desenvolvidas com casos já conhecidos.

## Evidência disponível

O replay dos 24 resultados M7 usa as mesmas imagens e saídas retidas, portanto
é evidência de desenvolvimento e não uma nova validação externa. Com a política
v2, os 24 candidatos geométricos antigos foram rejeitados. O resultado seletivo
fica idêntico ao npj direto.

| Medida no replay M7 | npj direto | M6 seguido de npj | Seletivo |
| --- | ---: | ---: | ---: |
| Cobertura impressa, média por paciente | 56,57% | 46,55% | 56,57% |
| Derivações inteiramente ausentes | 81/288 | 125/288 | 81/288 |
| Layouts compatíveis | 13/24 | 7/24 | 13/24 |
| NRMSE bruto em todo o suporte impresso | 1,2153 | 1,1032 | 1,2153 |

A seleção evita a regressão de cobertura, mas não melhora o extrator direto.
Também conserva seu erro bruto maior neste piloto. Esses números não podem
ser resumidos como uma vitória geral do híbrido.

No replay do único D0 já inspecionado, a política v2 aceita a saída nativa do
Kaggle: 18.750 amostras finitas contra 18.638, sem perder amostras diretas.
Esse caso ajudou a calibrar as tolerâncias de tempo e nível DC. Ele serve como
regressão de engenharia e não conta como confirmação independente de ganho.

O estudo pareado separado reexecutou npj direto, amostragem adaptativa da mesma
malha M7 e seleção. A execução local usou explicitamente 1500 pixels.

| Medida, mesmas 24 imagens de desenvolvimento | M6 anterior | Amostragem adaptativa | npj direto / seletivo |
| --- | ---: | ---: | ---: |
| Cobertura impressa por paciente | 46,55% | 51,31% | 56,57% |
| Derivações inteiramente ausentes | 125/288 | 110/288 | 81/288 |
| Layouts compatíveis | 7/24 | 10/24 | 13/24 |
| NRMSE bruto, suporte impresso completo | 1,1032 | 1,1395 | 1,2153 |

A adaptação recuperou 4,76 pontos percentuais de cobertura em relação ao M6,
mas piorou seu NRMSE bruto e ainda perdeu cobertura em relação ao direto.
Nenhum candidato foi aceito: nove mudaram o layout, dez excederam o limite de
extrapolação e dois perderam amostras diretas. Três malhas anteriores não
existiam porque a correção fora rejeitada. Nesses três casos, este controle de
amostragem reutiliza o resultado direto e registra `direct_alias`. O braço
`grid_adaptive` da próxima execução em GPU contará falhas como NaNs explícitos.
O estudo não refez nem corrigiu pontos detectados ou IDs da grade com dados de
referência. Os resultados e intervalos por paciente estão em
[`sampling-summary.json`](sampling-summary.json).

A tentativa
de 3000 pixels em CPU foi encerrada por falta de memória antes da primeira
predição, com o erro preservado. A execução de 3000 pixels é reservada à GPU.

## Próxima medição congelada

[`protocol.json`](../../reproductions/hybrid/protocol.json) e
[`frozen-cohort.json`](../../reproductions/hybrid/frozen-cohort.json) definem uma
coorte nova de oito pacientes e 24 fotografias, fora dos pacientes usados no
nosso treinamento e desenvolvimento anterior. O cruzamento com o treinamento
dos modelos originais continua desconhecido. Nenhum desfecho dessa nova coorte
foi usado para escolher regras ou parâmetros.

A comparação usa quatro braços, conserva falhas no denominador e só abre as
referências para pontuar depois de salvar todas as predições e decisões. O
pipeline Runpod tem fila própria para executar esse ensaio limitado em paralelo
ao treinamento npj já iniciado. Cópias antigas do ensaio são canceladas apenas
enquanto ainda não têm nenhum job. Um ensaio híbrido já iniciado bloqueia a nova
execução. O treinamento original não é cancelado por esse controle.
O limite adicional é US$ 0,45 de GPU, mais armazenamento temporário. A fila e
os recibos de execução são a fonte de verdade sobre início e conclusão.

Para chamar a biblioteca de melhor que as alternativas, ainda precisamos de
ganhos reproduzidos em uma coorte maior, independente, com intervalos por
paciente, falhas por derivação, morfologia, calibração, desempenho e rejeições
medidos. A aprovação em testes de software e a qualidade visual dos exemplos
sintéticos não substituem essa medição.

O procedimento e os comandos completos estão em
[`reproductions/hybrid/README.md`](../../reproductions/hybrid/README.md).

## Verificação e dados retidos

A suíte integrada executou 523 testes, com 498 aprovações e 25 indisponibilidades
declaradas de CUDA, Mitsuba ou métrica externa. Os 62 testes de reprodução
passaram, incluindo a auditoria da fonte npj fixada. Também passaram os controles
de seleção, amostragem e adaptação, o lint e uma execução real da nova CLI com
o modelo original em CPU a 1500 pixels. A galeria procedural de 16 cenários foi
exportada e inspecionada. Nenhuma dessas verificações é uma nova validação
clínica ou uma execução local em GPU.

[`validation.json`](validation.json) guarda recibos e hashes. O banco de
medições de 351 arquivos inclui as saídas completas, fotografias de origem,
referências de avaliação, malhas anteriores, logs e decisões. A identidade do
arquivo está em [`measurement-archive.json`](measurement-archive.json).

Os controles seguintes de orientação, composição downstream e diagnóstico do
eixo temporal estão na [continuação do estudo](followup/README.md).

## Auditoria do erro e melhorias

A [revisão de 17 fontes primárias e dos componentes](improvement-audit/README.md)
separa a redução de erro no suporte comum da redução obtida por perder amostras.
Inclui decomposição exata, componente experimental de tempo físico, instrumentação
de falhas nativas e experimentos com critérios de avanço.
