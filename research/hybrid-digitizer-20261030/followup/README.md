# Orientação, segunda correção e eixo de tempo

Foram executadas duas novas inferências nas 24 imagens M7 já conhecidas.
Os sinais diretos e os da amostragem adaptativa anterior foram reutilizados
sem alterar seus valores. Não é uma nova validação externa.

O código original npj sempre estima perspectiva, mesmo com o TPS desativado.
Depois pode girar os mapas se a imagem for mais alta que larga. A nossa saída
retificada também pode ficar em retrato quando o OCR não resolve a orientação.
Inspecionamos esse caso na primeira imagem congelada, `pm-9-photos_scans`.

Os controles chamam os módulos publicados de segmentação, extração e identificação
de derivações, com os mesmos checkpoints. O primeiro omite perspectiva, recorte,
rotação e busca de escala. O segundo conserva uma rotação exata de 90° nos mapas
em retrato. Ambos usam a densidade isotrópica inferida pelo nosso amostrador.
A comparação entre eles isola essa rotação. A comparação com o pipeline completo
também muda recorte e escala, portanto não isola somente a perspectiva.

| Todas as 24 imagens, média por paciente | Cobertura impressa | NRMSE bruto | Derivações ausentes | Layouts compatíveis |
| --- | ---: | ---: | ---: | ---: |
| npj direto | 56,57% | 1,2153 | 81/288 | 13/24 |
| RGB adaptativo seguido do npj completo | 51,31% | 1,1395 | 110/288 | 10/24 |
| Módulos downstream, orientação preservada | 37,03% | 1,2798 | 156/288 | 7/24 |
| Módulos downstream, controle de orientação | 55,32% | 1,1996 | 104/288 | 13/24 |
| Seleção conservadora | 56,57% | 1,2153 | 81/288 | 13/24 |

Três imagens não tinham uma malha aceita no banco anterior. Os dois controles
downstream registram essas falhas como NaNs. O braço adaptativo anterior reutiliza
o direto nessas três imagens. Para separar essa diferença, também comparamos os
21 casos com RGB retificado disponível, definidos pelos arquivos congelados antes
da inferência. Esse recorte é secundário e não substitui o denominador de 24 casos.

| Subconjunto pareado de 21 imagens, oito pacientes | Cobertura impressa | NRMSE bruto |
| --- | ---: | ---: |
| npj direto | 58,31% | 1,2231 |
| RGB adaptativo seguido do npj completo | 53,13% | 1,1393 |
| Downstream com orientação preservada | 41,21% | 1,3812 |
| Downstream com controle de orientação | 62,66% | 1,2669 |

O controle de orientação recupera suporte e layouts, mas não vence simultaneamente
em cobertura e erro. A aparente melhora de NRMSE no total de 24 casos também pode
ser influenciada pelo tratamento das falhas. No subconjunto pareado, o erro do
novo controle é maior que o do direto. Nenhum dos novos candidatos passou pela
seleção conservadora. Esses controles permanecem experimentais e não substituem
o caminho downstream do ensaio prospectivo em GPU.

## Diagnóstico temporal com referência

Um diagnóstico separado busca deslocamentos de até 500 ms, sem deslocamento
circular, ajuste de amplitude ou ajuste DC. Ele só recebe referências depois de
todas as predições e decisões estarem salvas. Não modifica predições, política de
seleção nem métricas primárias.

No npj direto, 121 das 288 derivações têm suporte suficiente para esse diagnóstico.
Entre elas, 91 apresentam o melhor deslocamento acima de 50 ms em módulo. A mediana
do deslocamento absoluto é 114 ms. A correlação mediana passa de 0,011 no eixo bruto
para 0,479 no melhor deslocamento. Isso sugere investigar origem e escala temporal,
mas não demonstra que o traçado esteja correto depois de uma simples translação.
Ritmos periódicos podem gerar picos concorrentes. Seus valores são preservados.

## Controle analítico da normalização original

Executamos `LeadIdentifier.normalize` do commit original verificado, sem alterar
seu código, sobre três linhas de pixels com um pico físico fixo em 4 segundos.
A escala é de 20 pixels/mm e 25 mm/s, equivalente a 500 amostras/s. Somente as
observações do início são ocultadas, conservando o sinal e seu intervalo físico.

| Início oculto | Pico retornado | Erro temporal | NaNs na entrada | NaNs na saída |
| --- | ---: | ---: | ---: | ---: |
| 0 ms | 4,000 s | 0 ms | 0 | 0 |
| 200 ms | 3,878 s | -122 ms | 300 | 0 |
| 400 ms | 3,750 s | -250 ms | 600 | 0 |
| 1000 ms | 3,334 s | -666 ms | 1500 | 0 |

O identificador recorta a extensão com pelo menos três linhas válidas e a
interpola para o número de amostras solicitado. O controle demonstra que esse
mecanismo pode mudar o tempo e apagar a informação de ausência. Não é inferência
neural nem prova da causa dos atrasos nas fotografias. A reprodução original
permanece intacta. Uma alternativa deverá conservar a origem física, a escala
pixels/mm e os trechos ausentes e ser comparada em um braço experimental separado.

O script é `reproductions/npj/time_axis_control.py`. O resultado completo, com
commit e hash do módulo original, está em `native-time-axis-control.json`.

## Rastreabilidade e execução

O retificador agora conserva coordenadas e confiança das detecções, IDs relativos,
células, malha completa e conectividade. Uma rejeição conserva as observações que
já existiam e informa a fase incompleta. A CLI e o worker salvam os arrays com hash.

A leitura do pod de treinamento às 14:33 de São Paulo, 17:33 UTC, confirmou `RUNNING`, uma L40 e
US$ 0,82/h. O endpoint não permitiu introspecção de telemetria. Esse recibo não
comprova épocas concluídas nem uso efetivo da GPU. O treinamento não foi alterado.

A geração 5 do piloto mantém as mesmas imagens, modelos publicados e política
congelada. Seu teto adicional continua US$ 0,45 de GPU, mais armazenamento. Ele
passa a ter fila própria para executar em paralelo ao treinamento. A admissão
cancela apenas cópias antigas do piloto que ainda não têm jobs. Um piloto já
iniciado provoca adiamento. Nenhum treinamento é cancelado por esse procedimento.

A execução da geração 3 passou pela admissão, mas falhou antes de alocar GPU
no controle independente de interpolação M4. Seu ambiente instalou OpenCV 5.0,
enquanto o ambiente das medições usa 4.11. A geração 4 fixa as versões numéricas
já verificadas. O teste, os resultados esperados e a tolerância foram mantidos.
Um controle local alterando somente OpenCV reproduziu a falha com 5.0 e passou
os mesmos 21 testes com 4.11. O recibo é `validation-environment-control.json`.
Com o novo controle temporal, os 69 testes de reprodução passam com o código
original disponível. A suíte integrada permanece com 530 testes, dos quais 25
exigem recursos indisponíveis localmente e estão explicitamente pulados.

Na execução `41105577589`, a recuperação do checkpoint terminou com hash correto,
ausência do pod confirmada e volume original preservado. O ensaio seguinte foi
recusado antes de criar GPU porque a API exige saldo mínimo de US$ 5 para criar
um novo volume de rede. Nenhuma inferência desse ensaio foi executada.

A geração 5 reutiliza o artefato de checkpoint dessa execução. A inferência curta
usa somente 40 GB de disco do contêiner, sem criar volume de rede ou volume
persistente de pod. O orçamento e o encerramento automático são mantidos. Os
resultados devem ser copiados e verificados antes da remoção do pod. Uma queda
abrupta antes da cópia pode perder resultados parciais. Esse modo não é permitido
para treinamento e não altera a execução de treinamento já ativa. Os três novos
contratos de armazenamento elevam os testes de reprodução para 72.

## Isolamento das observações entre imagens

Um controle adversarial encerra o prazo logo depois da primeira imagem. Ele
reproduziu uma falha no worker: se a próxima retificação não chegasse a começar,
o recibo podia conter observações da imagem anterior. O worker agora limpa o
relatório e os arrays antes de verificar o prazo da candidata. O teste falhou
antes da correção e passou depois. São 73 testes de reprodução passando com o
código original disponível. Predições e política de seleção não foram alteradas.

O ensaio GPU já iniciado conserva seu commit congelado, anterior a essa correção
de recibos. Seus resultados serão auditados por ID de origem. Uma eventual
observação de outra imagem não será usada como evidência geométrica da imagem
atual, e o recibo original será preservado. O controle está registrado em
`timeout-geometry-regression.json`.

A CLI completa também foi executada com pesos reais, CPU e resolução de 1500
pixels em `pm-9-photos_scans`, uma imagem já conhecida. A candidata ganhou 456
amostras finitas, mas perdeu 375 da leitura direta e foi rejeitada. O resultado
direto foi conservado. Esse é um teste de integração, sem comparação de precisão
com referência. Seu recibo é `known-image-cli-smoke.json`.

Os números completos, os intervalos descritivos por paciente e os protocolos
estão nos arquivos JSON desta pasta. Os novos scripts são
`examples/hybrid_downstream_study.py`, `examples/summarize_downstream_controls.py`
e `examples/audit_digitizer_time_axis.py`.

O banco completo desta continuação tem 275 arquivos, com sinais, referências,
decisões, diagnósticos e logs. Sua identidade está em
[`measurement-archive.json`](measurement-archive.json).
