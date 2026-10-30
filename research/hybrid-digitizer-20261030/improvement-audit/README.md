# Auditoria de melhorias para um digitalizador de ECG de uso geral

Continuação implementada: [integração temporal física e catálogo completo do PM-ECG-ID](../physical-clock/README.md).
Essa etapa inclui controles novos e a tentativa de GPU bloqueada por saldo
insuficiente do Runpod. Os resultados históricos desta auditoria permanecem
preservados.

Revisão de 30/10/2026. Objetivo: reduzir erro de sinal e falhas em ECGs reais, preservando as derivações, a calibração e os trechos observados. Liderança precisa ser demonstrada em avaliações comparáveis e independentes. Resultados em outro banco, scores de classificação e métricas de qualidade visual não estabelecem essa liderança.

Esta revisão questiona a métrica, a composição dos modelos, nossas hipóteses geométricas e o critério de seleção. Não representa uma nova validação externa. As 24 imagens cujos resultados já foram abertos passam a servir ao desenvolvimento. Nenhuma predição original foi alterada nesta auditoria.

## Por que o erro da grade ficou menor

O NRMSE bruto passou de 1,14984 para 0,91332, uma diferença de -0,23652. A média é calculada primeiro por imagem e depois por paciente. A decomposição exata dessa diferença é:

| Partição do mesmo eixo temporal impresso | Fração do suporte impresso | Contribuição para a diferença de NRMSE |
| --- | ---: | ---: |
| Ambos retornam sinal | 65,909% | -0,109954 |
| Apenas o npj direto retorna sinal | 19,379% | -0,127546 |
| Apenas a grade retorna sinal | 2,743% | +0,000984 |
| Ambos ficam ausentes | 11,969% | 0 |

Na segunda linha, o erro da candidata é o erro de zero contra a referência, porque a métrica antiga transforma ausências em zero somente para pontuar. Quando a predição direta era pior que zero, apagar aquela predição reduz o score. Essa partição explica 53,93% da redução líquida de NRMSE. É atribuição matemática do score, não demonstração da causa visual de cada falha.

Existe também uma melhora nos mesmos trechos observados: o RMSE comparável cai de 0,21049 para 0,17452 mV. Esse resultado é condicionado às 19 imagens com suporte comum, abrangendo os oito pacientes. As cinco imagens restantes continuam explicitamente ausentes dessa estatística condicional. Não se pode converter isso em desempenho de todas as imagens.

A grade perde amostras diretas em todas as 24 imagens, mesmo quando ganha outras amostras. A perda líquida é de 16,64 pontos percentuais. O seletor conservador continua retornando o npj direto em todos os casos.

| Condição, oito imagens cada | Cobertura direta | Cobertura com grade | RMSE no suporte comum, direto → grade |
| --- | ---: | ---: | ---: |
| Fotografias de papel dobrado | 78,26% | 30,77% | 0,2871 → 0,1695 mV |
| Fotografias Samsung | 90,70% | 92,81% | 0,1866 → 0,1658 mV |
| Scans fotografados | 86,90% | 82,39% | 0,2106 → 0,1815 mV |

As estatísticas condicionais têm seus denominadores detalhados no JSON. A coluna de papel dobrado é o maior sinal de falha de robustez. A melhora agregada em Samsung é uma hipótese de benefício localizado, sem provar melhora de cada imagem ou justificar roteamento ajustado neste mesmo grupo.

`support-decomposition.json` conserva os resultados por imagem, condição e paciente. O novo `support_audit.py` decompõe os erros sem alterar predições. O resumidor comum agora publica essa auditoria para cada candidata contra a referência direta. Uma candidata completamente ausente deixa de parecer apenas uma linha com um score numérico aparentemente competitivo.

A identidade usada é ΔNRMSE = ΔSSE / [energia_da_referência × (NRMSE_candidata + NRMSE_direta)]. A diferença de SSE é aditiva entre as quatro partições. Assim, a decomposição permanece exata mesmo com a raiz quadrada e a agregação por paciente. Referências de energia zero têm RMSE definido e NRMSE indefinido.

## O que as fontes consultadas acrescentam

As fontes primárias e os limites de acesso estão em `sources.json`. Alegações dos autores permanecem alegações dos autores. A nossa escolha anterior do segundo colocado foi baseada em disponibilidade e complexidade, não em robustez externa demonstrada.

| Trabalho ou recurso | Mecanismo relevante para investigar | Limite da evidência para nosso objetivo |
| --- | --- | --- |
| Open-ECG-Digitizer, npj, 2026 [S1] | Pipeline modular, layout, grade e extração abertos, com dados pareados | Coortes e protocolo diferem do nosso piloto. Treinamento original ainda parcial |
| Powerful Medical, artigo publicado e descrição pública [S2] | Dotter, organização da grade, retificação e reconstrução de sinais são etapas distintas | Arquitetura exata, pesos e banco original de Dotter não estão disponíveis aqui. Nossa grade não é o sistema comercial |
| Patente técnica US12614244B2 [S3] | Organização em células e mapeamento local da imagem | A descrição não fixa todos os detalhes numéricos nem comprova desempenho da nossa reconstrução |
| Kaggle 1º [S4] | Regressão direta, remapeamento da origem em alta resolução e ensemble | Generalização a outros layouts precisa ser medida. A descrição contém ablações em uma imagem e validação pequena |
| Kaggle 2º [S5] | Supervisão consistente com PTB-XL 500 Hz, máscaras esparsas e modelos 2,5D/inteiros | O foco declarado na parte média e fácil segue a agregação do concurso. A etapa zero compartilha dependências de template que falharam nas seis fotos elegíveis |
| Kaggle 3º [S6] | Mapa de baixa resolução aplicado ao original em alta resolução, refinamento parabólico | Depende dos estágios iniciais compartilhados e do layout do concurso |
| Kaggle 4º [S7] | Grade com índice aprendido, recortes por coluna, supervisão auxiliar e pretreino FCMAE | A anotação própria e os templates fazem parte do método. Transferir só o backbone não reproduz o resultado |
| Kaggle 5º e suas ablações [S8] | Pseudoanotação de pontos, registro, homografias locais, heatmap por coluna e recorte com maior faixa de amplitude | Ablações alteram a duração de DII na avaliação. Seus números não são diretamente os nossos ou o leaderboard |
| ECG-Digitiser, vencedor PhysioNet 2024 [S9] | Hough e nnU-Net como comparação aberta adicional | É outra competição. Precisamos executar no mesmo banco para comparar |
| ECGMiner [S10] | Alternativa clássica para extração e calibração | Não usar correlação publicada em renders como prova de robustez fotográfica |
| ECGtizer [S11] | Formatos variados, exportação aECG e visualização do sinal reconstituído | A conclusão neural de derivações parciais é uma tarefa distinta e precisa de proveniência própria |
| VLM-in-the-Loop, preprint 2026 [S12] | Diagnóstico de falhas apoiado em ferramentas e reprocessamento limitado | AUROC de fidelidade 0,64 e PCC pareado global 0,28. As 428 imagens clínicas não têm sinal digital nativo. Qualidade “Excellent” não significa reconstrução fiel |
| PTB-XL-Image-17K [S13] | Máscaras, sinais, caixas e metadados com escalas e velocidades variadas | É sintético e compartilha a fonte PTB-XL. Separar pacientes antes de misturar bancos |
| ECG-Image-Kit [S14] | Gerador adicional para reduzir dependência de um único renderizador | Realismo sintético é hipótese que precisa de validação externa |
| ECGLight, preprint 2026 [S15] | Detecção de regiões, texto e pulso, execução em CPU e interface de revisão | Classificação de IAM e reconstrução de forma de onda são endpoints distintos |
| CU²-NeXt, 2026 [S16] | Segmentação e trajetória com Viterbi, avaliação de medidas de ECG | Precisamos auditar os dados, disponibilidade e protocolo antes de usar como comparação numérica |
| ECG-Image-Database v2 [S17] | Rota pública para Ahus e Emory, além de PTB-XL | Reconciliar páginas PDF, variantes e versões antes de chamar de reprodução exata da coorte do artigo |

## Revisão completa dos componentes

| Prioridade | Limitação atual ou risco verificável | Melhoria e experimento que pode refutá-la |
| --- | --- | --- |
| P0 | Um score pode melhorar com perda de suporte | Publicar suporte comum, perdido, ganho e ausente junto do erro. Medir sucesso por imagem e por derivação, com todos os agendados no denominador |
| P0 | Normalizar pelo primeiro e último ponto visível pode esticar o tempo | Propagar x, origem do recorte e máscara desde a extração. Converter distância física em segundos, mantendo lacunas |
| P0 | Ganho fixo de 10 mm/mV e duração solicitada não são inferência do papel | Detectar texto de velocidade/ganho e pulso de calibração. Manter hipóteses concorrentes e confiança. Usar escala x e y separadas |
| P0 | O especialista Kaggle pode falhar antes de chegar ao modelo de sinal | Salvar saída, orientação, pontos e exceção completa em cada estágio. Verificar semântica de flips, ordem de cantos e homografia em controles positivos conhecidos |
| P0 | A referência impressa não distingue oclusão real de falha do algoritmo | Anotar visibilidade independentemente das predições. Relatar denominadores impresso e visível, sem apresentar imputação como observação |
| P0 | Oito pacientes não estabelecem liderança ou estabilidade | Ampliar bancos pareados reais e congelar novo teste por paciente, dispositivo e template. Agrupar todas as variantes e fontes duplicadas |
| P1 | Grade periódica não informa sozinha orientação semântica, origem ou pitch | Combinar grade com rótulos, separadores e pulsos. Treinar índices locais e checar ciclos, linhas perdidas e troca 1/5 mm |
| P1 | Detector global de 704×1024 perde detalhes e não usa tiles nesta execução | Comparar global, tiles nativos e refinamento local dos pontos. Medir erro físico, identidade dos índices e cobertura, além de Dice/AP |
| P1 | Papel dobrado concentra perda de suporte | Estimar geometria por regiões, excluir só regiões sem evidência e manter o restante. Comparar fronteiras, reflexões e extrapolação em pontos independentes |
| P1 | Uma malha válida pode ter correspondências erradas | Validar com pontos não usados no ajuste, consistência de pitch e reprojeção. Estimar incerteza do mapa local |
| P1 | Retificar externamente e depois aplicar todo o npj pode duplicar transformações | Compor mapas até o RGB original e amostrar uma vez. O bypass anterior não melhorou globalmente, então isolá-lo como ablação |
| P1 | Orçamento de 3000 pixels limita detalhes no documento inteiro | Recortar derivações no RGB original com resolução guiada pelo Jacobiano. Comparar filtros de amostragem sem gerar detalhes aprendidos |
| P1 | Média de pixel/mm dos dois eixos perde anisotropia residual | Estimar calibração de cada eixo e sua incerteza. Testar perspectiva, zoom, crop e escala não uniforme com soluções analíticas |
| P1 | Extração pode saltar de uma derivação para a vizinha em cruzamentos | Aprender propriedade do traço por derivação. Comparar heatmap de coluna e trajetória com continuidade, mantendo alternativas multimodais |
| P1 | Uma máscara grossa não determina centro de linha preciso | Medir o teto sinal→raster→sinal antes do treinamento. Comparar argmax, parábola, Gaussian/JSD e regressão. Usar posições subpixel nas labels |
| P1 | Saturação do recorte pode cortar QRS de alta amplitude | Detectar clipping e ampliar o recorte local. Incluir hipertrofia, marcapasso e amplitude extrema no teste |
| P1 | Reordenar derivações pode trocar polaridade, ritmo longo ou janelas sequenciais | Validar Cabrera, -aVR, derivações repetidas e tiras longas identificadas por texto. Propagar lead e tempo por segmento |
| P1 | Média do traço não equivale à linha isoelétrica | Estimar baseline com evidência do documento e incerteza. Medir preservação de ST e T sem centralização usada para esconder erro |
| P1 | BCE/Dice ou loss fotográfica podem ignorar sinal e calibração | Comparar perdas de coordenada, waveform em mV e derivada temporal, com máscara de visibilidade. Relatar ST, picos e intervalos separadamente |
| P1 | Fidelidade do nosso renderizador não garante transferência | Misturar renderizadores independentes, dados reais e poucos rótulos geométricos revisados. Medir o ganho em domínio não usado para ajuste |
| P1 | Concordar com uma leitura direta errada pode bloquear uma candidata correta | Aprender erro/confiança em validação separada. Avaliar seleção por derivação apoiada em reprojeção e escala, em vez de só concordância com o baseline |
| P1 | Um SQI pode preferir um sinal limpo e incorreto a um ECG patológico | Validar confiança contra erro real. Anormalidades fisiológicas não são critério automático de rejeição |
| P2 | Relações de Einthoven podem ser aplicadas em tempos diferentes | Usar somente trechos simultâneos, derivados compatíveis e mesma filtragem. Preservar saídas antes e depois como ablação |
| P2 | Fourier global pode espalhar erro e ringing através de lacunas | Testar por trecho observado, bordas e não periodicidade. Manter máscara e separar interpolado de observado |
| P2 | Completar 10 s a partir de 2,5 s pode misturar inferência com medição | Marcar explicitamente observado, interpolado e imputado. Comparar digitalização usando somente o que está impresso |
| P2 | PDF pode conter o traçado vetorial original | Detectar conteúdo vetorial e extrair coordenadas antes de rasterizar. Validar a mesma calibração e semântica de leads |
| P2 | Contratos atuais divergem sobre NaNs e sucesso parcial | Unificar API com leads nomeados, janelas, unidades, máscaras, incerteza e proveniência. Exportar WFDB/aECG com suporte explícito |
| P2 | Tempo total inclui custo que forward isolado oculta | Medir cold start, decodificação, geometria, OCR, modelos e exportação. Comparar CPU/GPU, memória e lote com a mesma qualidade |
| P2 | Quantização e AMP podem mover pontos subpixel | Checar paridade de coordenadas, amplitude e tempo antes de exportar ONNX ou quantizar. Manter operações geométricas na precisão necessária |

## Arquitetura a testar

O caminho principal deve ser um modelo de documento com regiões e rótulos de derivações, calibrador físico e mapa geométrico com incerteza, seguido de leitura em alta resolução por derivação. A saída deve reunir sinal, eixo temporal, janela impressa, máscara de observação, confiança e fonte da calibração. A verificação deve reprojetar o traçado sobre a imagem original, com propriedade da derivação e máscara de oclusão. Distância a qualquer tinta isolada não basta em cruzamentos.

A grade é uma fonte de evidência útil, e sua ausência não deveria impedir automaticamente a leitura de um traçado reconhecível. Texto, separadores e pulso podem fornecer outras restrições. Para formatos novos, precisamos manter hipóteses de layout e declarar a ambiguidade em vez de forçar o template do concurso.

Um único modelo bem calibrado é o primeiro experimento. Ensembles, TTA e modelos grandes entram depois de medir onde oferecem ganho. Aumentar a concordância com uma saída errada, suavizar um sinal ou completar segmentos escondidos não satisfaz o objetivo de fidelidade.

## Sequência de experimentos

| ID | Execução | Evidência exigida e critério de avanço |
| --- | --- | --- |
| E0 | Auditoria de suporte no banco salvo | Identidade exata da decomposição, hashes verificados e denominadores explícitos. Executado nesta revisão |
| E1 | Reamostragem física com origem fixa | Pico analítico imóvel sob oclusão, ganho correto com escalas diferentes por eixo, nenhum preenchimento de lacuna. Componente e controles implementados, integração fotográfica pendente |
| E2 | Instrumentar extração e etapa zero | O adaptador agora registra exceções capturadas nos três estágios, com testes e comparação estática do código numérico. Nova inferência para diagnosticar as seis falhas e salvamento de pontos pendentes. Falha de estágio não vira predição zero válida |
| E3 | Integração temporal e calibração | Comparar somente a mudança de coordenadas, mantendo pesos e geometria. Confirmar origem, baseline e janelas antes de pontuar |
| E4 | Retificação e ROI sem perdas adicionais | Comparar uma amostragem do RGB original, tiles e interpoladores. Medir pontos independentes e sinais no mesmo suporte |
| E5 | Codec e modelo de waveform | Comparar teto geométrico, heatmap por coluna e regressão com um backbone, mantendo o mesmo split e faixa de amplitude |
| E6 | Treinamento de robustez | Parear captura física e sinal, incluir domínios/velocidades/ganhos variados, oclusões e patologias. Medir curva de aprendizagem e contribuição de cada fonte |
| E7 | Confiança e seleção por derivação | Calibrar em validação separada. Testar erro versus cobertura, detecção de falha e taxa de aceitação incorreta |
| E8 | Comparação externa congelada | Executar os métodos completos no mesmo manifesto, política de falha e protocolo. Publicar resultados por dispositivo, layout e severidade |
| E9 | Validação orientada à finalidade clínica | Medir ST, amplitude, R, PR/QRS/QT e desempenho de um modelo downstream congelado em sinais originais e digitalizados. Definir tolerâncias previamente com especialistas |
| E10 | Biblioteca e desempenho | API unificada, exemplos, contratos, exportação, CPU/GPU e pacote de pesos com proveniência reproduzível |

Os experimentos neurais devem usar o Runpod solicitado. Primeiro validar o runner e a memória em um lote curto, salvar intermediários e só depois executar o ensaio completo. Cada invocação precisa de plano imutável, limite de tempo/preço, checkpoint e encerramento externo verificável. Esta revisão não iniciou novos recursos pagos, nem alterou o treino original em andamento.

Não definir um score único que esconda derivações perdidas. O resultado candidato deve melhorar erro nos mesmos trechos, preservar suporte, reduzir falhas por exame e manter morfologia no teste novo. Se houver troca entre objetivos, publicar a fronteira erro/cobertura. Relatar média, mediana, cauda inferior, pior grupo e erro de calibração da confiança. A agregação linear de SNR do concurso valoriza casos fáceis com score alto e não substitui avaliação de falhas difíceis. ICs devem usar pacientes ou registros independentes, e não tratar várias fotos do mesmo ECG como amostras independentes.

## Código e reprodução desta revisão

```bash
python -m unittest tests.test_support_audit tests.test_physical_trace -v
python -m examples.compare_trace_time_axes --upstream /dados/Open-ECG-Digitizer --out /dados/controle-temporal.json
python -m examples.audit_digitizer_support \
  --inputs /dados/pacote/original-inputs \
  --run /dados/pacote/completed-evaluation/run \
  --out /dados/auditoria-suporte.json
```

O componente físico recebe coordenadas, baselines e calibração fornecidos explicitamente pelo chamador. Não infere automaticamente esses parâmetros e ainda não foi validado em fotos reais. Não alteramos os fontes/pesos originais para reescrever a reprodução. As medições anteriores e seus pacotes permanecem intactos.

Validação desta revisão: 541 testes integrados, com 25 skips explícitos, 83 testes de reprodução com a fonte original habilitada e 17 testes focados em sinal, suporte e resumo. Os 16 cenários adversos foram exportados e inspecionados e seu painel é idêntico ao anterior. Os comandos e hashes estão em `validation.json`. O log de falha Kaggle foi acrescentado sem alterar o código numérico dos três estágios, verificado por AST.
