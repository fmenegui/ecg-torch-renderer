# M4: fontes externas e alternativas para retificação de ECG

Verificado em 23 de outubro de 2026. Este parecer é uma revisão de mecanismos e hipóteses. Não executou novos modelos nem alterou M3 ou produção. Fontes externas foram lidas sem enviar imagens a serviços de terceiros. Os arquivos desta pasta contêm somente nossa síntese, referências e propostas.

## Decisão recomendada

O candidato mais forte é uma arquitetura que **mede linhas e pontos, mantém hipóteses de identidade e unidade, escolhe a complexidade geométrica por evidência independente e reamostra o RGB original uma vez**. A maior oportunidade atual está em indexação, orientação e preservação de conteúdo. Trocar o interpolador ou adotar uma rede maior não resolve esses três problemas.

Isso é uma recomendação de pesquisa, não uma vitória experimental nem uma afirmação de novidade. A arquitetura precisa vencer H global, M2 e os comparadores autênticos na mesma população, preservando falhas e caudas.

Os rótulos usados abaixo são:

- **Documentado**: mecanismo identificado em fonte primária ou código original. Não implica que o produto comercial atual tenha exatamente esse mecanismo.
- **Inferido**: consequência de engenharia ou limite matemático que nós deduzimos.
- **A testar**: proposta ainda sem resultado nesta rodada.

## O que as referências realmente estabelecem

### npj Digital Medicine

**Documentado [N1, N2, N3].** O artigo é *Digitizing paper ECGs at scale: an open-source algorithm for clinical research*, DOI 10.1038/s41746-025-02327-1. O método principal combina segmentação, duas famílias de linhas em domínio Hough, correção projetiva global, recorte informado por traçado e periodicidade para escala em cada eixo. Isso é diferente de retificação local completa de uma folha dobrada.

A branch oficial `main`, consultada novamente pelo conector GitHub, continua em `97a15087d4abcda843da8c58ee74b1d8f47e6f9a`, versão 1.9.3, commit de 4 de junho de 2026. É a mesma fonte congelada no M3. O README classifica `dewarper.py` como experimental e não formalmente avaliado. Os três presets oficiais mantêm `apply_dewarping: false`.

**Documentado em código [N3].** O braço experimental detecta cruzamentos, organiza/refina pontos e ajusta TPS usando subconjunto aleatório limitado de controles. Possui fallback de identidade se o ajuste falha. Portanto um experimento com esse TPS deve receber nome e seed próprios. Seus resultados não pertencem ao método principal validado no artigo.

**Inferido.** H é um candidato essencial para papel plano porque usa muitas observações para reduzir ruído e representa perspectiva exatamente. Recorte pelo traçado e limites prévios de quantidade de linhas podem restringir a generalização. Esses mecanismos precisam ser avaliados separadamente de falhas do adapter, já auditado no M3.

### Powerful Medical: Dotter, Gridder e operador de células

**Documentado [P1, P2, P3].** Dotter localiza interseções, Gridder associa a rede e infere ausências. Undistortion reamostra células. Leader pertence à extração de sinal posterior. Não são quatro métodos independentes de retificação.

A família de patentes descreve candidatos por vizinhos, classes direcionais de arestas, probabilidades, caminhos de quatro direções que fecham ciclos, seleção de rede por BFS e inferência de vértices ausentes por regressão. As constantes publicadas são exemplos, não parâmetros comprovados do servidor atual. O pedido US20240144417A1 recebeu a publicação de concessão US12614244B2 em 28 de abril de 2026.

**Inferido.** Reciprocidade e ciclos são ferramentas úteis contra arestas isoladas erradas. Um grid regular com linhas alternadas ausentes pode manter ciclos corretos e a unidade errada. Regressão de pontos ausentes não observa uma dobra escondida. O maior componente não representa necessariamente a folha inteira.

**Documentado com limite [P3, P4].** A descrição de transformação de quatro cantos e a palavra “linear” não determinam univocamente as convenções de homografia, bilinear, bordas ou costura. Não há evidência pública suficiente aqui para atribuir TPS ao produto. O schema público auditado no M3 não exporta Dotter masks, vértices Gridder ou mapa denso. Logo nossos controles de quatro cantos continuam sendo controles de primitivas, não execução ou reprodução validada de Powerful Medical.

**Não estabelecido.** Versão comercial atual, checkpoint, interpolador exato, comportamento em margens sem grid e erro geométrico em mm. Ausência de acesso recebe N/A. Nenhum score inventado pode preencher essa linha.

### Kaggle e informação semântica

**Documentado no código auditado no M3 [K1].** Tang, terceiro lugar com base Heng, usa marcadores de derivações e IDs de linhas/colunas vinculados a template. Isso oferece orientação e identidade que uma nuvem de pontos isolada não possui. A mesma vinculação limita o suporte a outros layouts. Este parecer não representa esse braço como todas as soluções Kaggle.

**Inferido.** A ideia útil é identidade semântica e ordenação. Copiar o tamanho fixo de template ou usar os marcadores de derivações como cantos métricos universais seria uma decisão frágil. Texto pode determinar a orientação mesmo quando não fornece coordenadas métricas precisas.

## O que acrescentar à nossa arquitetura

### 1. Decoder conjunto de unidade, saltos e orientação

**A testar, prioridade P0, propostas A01 e A02.** Manter, para cada observação, posição, covariância de localização, família de linha, classe major/minor e identidade relativa. Uma aresta candidata deve permitir salto inteiro, não apenas um vizinho consecutivo. Organizar hipóteses de pitch, D4, deslocamentos inteiros por componente e modelo geométrico. As hipóteses competem usando evidência da imagem. Não consultar gold para escolher por caso.

Restrições independentes devem ter funções distintas:

| Evidência | O que pode resolver | O que não prova sozinha |
|---|---|---|
| Linhas das duas famílias e pontos | Tangentes, cruzamentos, ordem e saltos quando linhas intermediárias aparecem | Unidade absoluta quando só sobra um padrão periódico indistinguível |
| Ciclos e reciprocidade | Inconsistência entre associações locais | Pitch correto ou fase global correta de um grid coerente |
| Major/minor e razão 5:1 | Escala relativa com evidência visual dessas classes | Classe correta se a impressão perdeu a distinção |
| OCR de velocidade, ganho e pulso de calibração reconhecido | Testes independentes de escala e orientação sob convenções verificadas | Que todo pulso tenha 1 mV ou duração padrão, ou que o OCR esteja correto |
| Texto e nomes de derivações | Orientação clínica e reflexão, com escopo de layout | Posições métricas universais dos marcadores |
| Bordas da folha e formato declarado | Contexto e canvas, dimensão se fisicamente conhecida | Escala de um recorte ou de uma impressão redimensionada |
| Linha observada atravessando um furo | Ligação entre componentes e geometria ao longo da linha | Superfície inteira entre essas linhas |

**Inferido.** Há casos não identificáveis: duas imagens byte a byte iguais podem representar pitches nominais diferentes ou rotações clínicas diferentes quando todo texto está ausente. Nenhum algoritmo determinístico consegue responder corretamente às duas verdades com uma única decisão. Uma metodologia superior precisa reconhecer esse limite e devolver hipóteses ou ambiguidade, preservando o conteúdo.

Para componentes desconectados, guardar IDs locais e fase relativa não resolvida. Reconciliação deve usar continuação realmente observada de linhas, um modelo global sustentado em ambos os lados ou semântica válida. Não atribuir fase por proximidade através de uma dobra invisível. Um ajuste global que liga os lados é uma hipótese, não nova observação.

### 2. Medir o desenho inteiro das linhas, além dos centros das interseções

**A testar, P0/P1, A03.** Frontend com duas famílias de linhas e pontos, usando rede para separar grid de traçado/texto e estimativa geométrica para refinar cruzamentos em resolução original. Estimar covariância anisotrópica de cada ponto ou segmento. Contrastar centroides, ajuste local de duas linhas e offsets aprendidos com o mesmo decoder e orçamento.

**Inferido [C1].** Perspectiva e blur tornam a precisão dependente da direção. Um score escalar alto pode coexistir com grande erro ao longo de uma linha. A covariância mede localização sob um modelo, não a chance de a linha ter o ID ou pitch corretos. O ponto deve conservar incertezas separadas de localização, associação e unidade.

Outra hipótese é prever fases contínuas das duas famílias e desenrolar seus índices com pontos/major-minor como âncoras. Isso pode usar linhas interrompidas sem exigir quatro vizinhos perfeitos. Também pode sofrer saltos de fase e harmônicos. Deve competir com um decoder de instâncias de linhas. Não há resultado nem prioridade de novidade atribuída a essa proposta.

### 3. Hierarquia de modelos geométricos com residual regularizado

**A testar, P1, A04.** Comparar nas mesmas observações:

1. H global com ajuste robusto e pesos de localização.
2. `F = T ∘ H`, com residual contínuo compartilhado e intensidade regularizada.
3. Células com curvas de arestas compartilhadas e interpolação transfinita no espaço após H.
4. Modelo físico com dobras explícitas para casos em que a evidência justifique essa complexidade.

Residual zero deve recuperar H exatamente. Penalizar a parte redundante entre H e o residual para evitar que ambos expliquem a mesma deformação. Seleção de regularização pode usar linhas ou regiões inteiras reservadas da própria imagem. Esse holdout mede extrapolação entre observações, mas não detecta um erro coerente de IDs. Unidade e orientação continuam precisando de testes independentes.

**Inferido.** C0, Jacobiano positivo e round-trip são contratos matemáticos úteis. Não certificam unidade, orientação, fidelidade ao traçado ou risco em mm. As falhas do protótipo M3 em ruído e IDs corrompidos impedem promover um residual interpolante universal. Uma superfície com crease pode ser C0 sem ser C1. Suavidade de derivadas através de uma dobra verdadeira não deve ser uma exigência universal.

### 4. A região sem grid continua sendo imagem

**A testar, P0, A05.** O contrato precisa separar RGB e geometria elegível. Uma máscara de grid mede evidência, não propriedade do conteúdo. Sem grid não significa sem texto, traçado, pulso ou margem relevante.

O canvas deve considerar a folha/conteúdo visível, não apenas o casco das interseções. Guardar mapa, origem dos nós e validade de amostragem. Distinguir observação, interpolação em furo, extrapolação, componente sem fase e oclusão. Se não houver geometria defensável, preservar a entrada e declarar o estado. Essa preservação não conta como RGB retificado com sucesso.

Investigar quatro mecanismos diferentes de aparente apagamento: zero escrito por máscara, remap fora da imagem, crop do canvas e perda de contraste por reamostragem. Todos podem parecer semelhantes numa miniatura, mas exigem correções distintas.

### 5. Separar restauro fotométrico de medição geométrica

**A testar, P1, A06.** Pode haver normalização fotométrica para o detector. O RGB entregue pela retificação deve vir do original, por mapa composto e integração apropriada do footprint. Não reconstruir o traçado com um gerador para melhorar aparência. Um modelo generativo de coordenadas ainda pode ser um candidato de warp, sujeito aos mesmos gates. A amostragem original não impede um mapa errado de deformar o sinal.

**Inferido.** Não usar “todos os elementos devem ficar retos” como loss de ECG. Grid e linhas de texto conhecidas como retas podem fornecer esse alvo. O traçado fisiológico tem curvatura própria. Ele deve ser preservado, não aplainado para parecer texto.

## Mecanismos úteis de dewarping de documentos

Os resultados de OCR e semelhança de imagem desses artigos não medem calibração ECG em mm. A tabela seleciona mecanismos com utilidade plausível, sem chamar nenhum deles de vencedor no nosso endpoint.

| Fonte primária | Documentado | Inferência para ECG e limitação |
|---|---|---|
| RDGR, CVPR 2022 [D1] | Otimiza uma grade com restrições de bordas e linhas de texto, em vez de depender apenas de fluxo previsto. Discute interpolação transfinita. | Restrições de linhas podem complementar pontos. Precisamos de IDs e unidade próprios. Não aplicar o prior de retidão ao traçado fisiológico. |
| UVDoc, SIGGRAPH Asia 2023 [D2] | Prediz grid 2D inverso 45×31 e grid 3D como tarefa auxiliar. Reamostra via grid interpolado. Dados usam papel real capturado com pontos UV invisíveis em iluminação normal. Assume visibilidade completa e ausência de foldovers/oclusões. | A aquisição com referência invisível é uma opção forte para ground truth real independente. O grid neural é um prior possível, não observação em região sem evidência. O README contém errata de avaliação e deve acompanhar qualquer reprodução. |
| DocTr++, TMM 2024 [D3] | Encoder-decoder hierárquico e definição de mapeamento para imagens com bordas completas, parciais ou ausentes. | Ataca diretamente recortes, que quebram modelos dependentes de borda completa. A rede não fornece automaticamente escala clínica nem risco calibrado. |
| DocMatcher, WACV 2025 [D4] | Associa linhas estruturais e textuais entre imagem e template. | Identidade pode usar um template reconhecido, em vez de inferir apenas vizinhança. Um template errado também pode produzir correspondências coerentes erradas. Avaliar suporte de layout explicitamente. |
| ForCenNet, ICCV 2025 [D5] | Máscara de conteúdo e loss de curvatura de elementos extraídos da referência. | Supervisionar preservação de pequenos elementos pode ajudar. A raiz oficial consultada contém apenas README e assets, sem inferência ou pesos demonstrados. É referência metodológica, não comparador executado. |
| D2Dewarp, CVPR 2026 [D6] | Segmentação das famílias horizontal/vertical com decoders separados e fusão entre os dois eixos. | É uma inspiração direta para evidência de grid em duas direções. Não resolve sozinho o inteiro de cada linha ou seu pitch. Há código de treino/inferência e links de pesos. O README declara CC BY-NC-ND 4.0 para pesquisa não comercial. |
| DvD, preprint 2025 [D7] | Difusão nas coordenadas do mapa, em vez de difusão no RGB. | Permite estudar múltiplas hipóteses sem redesenhar o sinal. Amostras de um gerador não são intervalos calibrados de erro, nem diversidade prova que incluam a verdade. |
| Isometric Mapping, preprint de Luo e Bo [D8] | Reconstrução/achatamento com desenvolvibilidade e linhas, voltada a creases e dobras grandes. | Modelo com crease explícita merece braço próprio. Uma única foto com oclusão não identifica a superfície escondida. Não prometer recuperação desse conteúdo. |
| SyntheticDoc, ECCV 2026 [D9] | Um milhão de amostras de treino, geometria física e path tracing. O artigo anuncia anotações ricas. O README confirma imagens, albedo, shading, UV, mapa inverso e metadata disponíveis, mas normais/3D ainda por liberar. O artigo registra custo de armazenamento e limitações de câmera/viewpoint. | Usar uma fração ou o gerador para validação física e treino com texturas ECG, mantendo fontes separadas. Mais realismo sintético não estabelece desempenho em fotografias clínicas nem cobre todas as falhas. |

**Disponibilidade, documentada em 23/10/2026.** Repositórios oficiais inspecionados, sem executar modelos:

| Repo | Commit consultado | Resultado da inspeção |
|---|---|---|
| `tanguymagne/UVDoc` | `4c9b82b537057aff2526e6dd118a847cdd072e82` | Código, LICENSE e checkpoint `model/best_model.pkl` de 32.158.393 bytes confirmado em inventário. |
| `xiaomore/D2Dewarp` | `1105c0f0e44404d9a8dedb78f689078dba42221c` | Código de treino/inferência e links de pesos. Licença declarada no README: CC BY-NC-ND 4.0. |
| `caipeng328/ForCenNet` | `d524b96bab518054fbe1c499ee43f0eec774d81d` | README e assets. Não há comparador executável demonstrado nesse snapshot. |
| `felixhertlein/doc-matcher` | `750a1cb1f8c8e8cd2e5bf207ad1f98ba1169ac33` | Código e LICENSE. `models.yaml` aponta pesos SAM/Lineformer/LightGlue para DocMatcher. Não foram baixados. |
| `tanguymagne/SyntheticDoc` | `aabd9cf05f479781df432e3bd7e940f3e6716dbd` | Código de geração/treino e LICENSE. Disponibilidade das anotações é parcial, como descrito acima. |
| `Liu-Shaokai/TextFlow` | `f14329d01511a6b939903cfab02e4559efcbb1ba` | README, assets e `textflow.py`. Não foi executado nem auditado até a saída. |

A ausência de um arquivo na raiz não é uma conclusão jurídica nem prova de que jamais exista licença em outro canal. Antes de integrar código/pesos, verificar as condições correspondentes.

## Casos que devem poder refutar o candidato

O catálogo estruturado em `proposals.json` especifica 24 famílias de falsificação. As prioridades são:

| Família | Construção independente | Decisão que pode ser refutada |
|---|---|---|
| Ambiguidade inevitável | Mesmo raster, dois pitches ou duas orientações verdadeiras, sem qualquer pista extra | Garantia de unidade/orientação universal a partir do grid |
| Harmônico coerente | Sumir com toda segunda major, ou perder minor de modo espacialmente consistente | Uso de ciclos como certificado de escala |
| Perspectiva sem ruído | Pontos e linhas perfeitos sob H forte | Associação por distâncias/ângulos globais fixos |
| Componentes separados | Dois grids corretos, offset inteiro variável e furo que esconde plano ou dobra | Escolha de fase por proximidade ou componente maior |
| Geometrias indistinguíveis no furo | Mesmas bordas observadas e interiores físicos diferentes | Confiança em interpolação/extrapolação sem evidência |
| Ruído e curvatura | Plano ruidoso, curva suave, crease alinhada e oblíqua, corrupção coerente | Residual sempre melhor, TPS sempre melhor ou C1 sempre desejável |
| Aparência adversa | Binarização, JPEG, blur direcional, moiré de monitor, dupla exposição, grid falso maior | Confiança de detector como confiança física |
| Conteúdo sem grid | Texto, spikes, pulsos e traços finos fora do suporte e junto das bordas | Máscara de grid como máscara de RGB, canvas limitado ao grid |
| Câmera e aquisição | Distorção radial, rolling shutter, scanner com escala desigual, tela fotografada | Atribuir toda curvatura ao papel e escolher um H único |
| Oclusão real | Dobra que se encobre ou mão opaca, verdade separando visível/escondido | Reconstruir informação ausente como se fosse medição |

Nenhum catálogo finito cobre todos os casos possíveis. O objetivo é conservar negativos e aumentar a capacidade de refutar nossas escolhas.

## Comparação experimental posterior

Separar três perguntas. Primeiro, operadores geométricos recebem as mesmas observações e IDs. Segundo, pipelines completos recebem a mesma imagem e metadados permitidos, mantendo configurações nativas. Terceiro, waveform usa um extrator comum ou mede o pipeline inteiro, com calibração bruta e alinhada separadas.

Na geometria, conservar queries e domínio de papel fixos, um offset de origem permitido quando necessário, sem escolher escala, D4 ou shear com gold no endpoint principal. Medir G/W/C/R, caudas, falhas, conteúdo visível preservado e custo, estratificados por suporte e origem da evidência. Calibrar política e thresholds fora do teste. Reservar fontes, pacientes, capturas e mecanismos, não apenas pixels ou augmentations.

**Documentado [E1].** O paper *ECG-image-database*, publicado em Physiological Measurement em julho de 2026, descreve a mesma família de 37.191 imagens Ahus/Emory/PTB-XL já associada ao npj e desafios. Não é um holdout novo por ter uma publicação nova. É necessário auditar overlap por ECG/paciente/fonte antes de chamar uma parte de externa.

Para dados reais com ground truth denso, a ideia UV é especialmente atraente: obter referências invisíveis na fotografia normal e visíveis em captura controlada registrada. É uma proposta de aquisição, não um resultado disponível. Medir movimentação entre capturas e erro da referência. A unidade independente inclui a folha/geometria física, além do paciente e da textura.

## O que podemos afirmar sobre superioridade e novidade

**Hoje:** há mecanismos publicados que justificam melhorar nosso decoder e desenho experimental. M3 encontrou limites concretos e preservou negativos. Não demonstra superioridade universal, clínica ou comercial.

**Comercial:** artigo e patente documentam mecanismos. Não identificam a build atual. Nosso controle não se chama Dotter/Gridder. Uma comparação autêntica exigiria acesso autorizado e endpoint compatível, mantendo identificador de build e warnings. Nenhuma chamada de inferência comercial foi feita nesta tarefa.

**Novidade:** detectar pontos, usar ciclos, inferir ausências, combinar H e residual, reamostrar o original e usar informação semântica já têm precedentes. A combinação pode ser boa engenharia. Uma contribuição científica candidata seria mostrar, com protocolo independente e transparência de unidade/ambiguidade/proveniência, uma melhor fronteira entre cobertura correta, risco e preservação em múltiplas fontes. Não é lícito transformar uma busca limitada em garantia de ineditismo ou liberdade de operação.

As URLs e os limites de cada fonte estão em `sources.json`. As propostas A01 a A10 e seus gates estão em `proposals.json`.
