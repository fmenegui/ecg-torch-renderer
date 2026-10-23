# Censo metodológico Kaggle para retificação de ECG

Data de leitura: 23 de outubro de 2026. Este trabalho amplia o M3. Não altera o código do retificador, não executa novos checkpoints e não transforma SNR de digitalização em erro geométrico.

Foram percorridas as seis páginas da listagem oficial do fórum, com 117 tópicos e 117 IDs únicos. Os 24 itens com `isWriteUp=true` foram recuperados integralmente pela API pública oficial e lidos. As 24 URLs da listagem coincidem com as 24 entradas de `census.json`. O índice de busca e um agregador serviram para descoberta, mas as afirmações técnicas usam os autores e seu código. A cobertura é completa para essa listagem recuperada nesta data. Ela não inclui soluções privadas, material removido ou todo notebook público sem writeup.

Os metadados atuais do autor não coincidem com a posição do título em dois casos: sasa_leaf tem título 97 e campo `competitionRanking=93`, Kirderf tem título 98 e campo 91. O outro texto de posição 97 usa o slug `104th-place-solution`, com metadado atual 97. O censo conserva esses fatos separadamente. Não tenta adivinhar a causa.

## O que muda em relação ao M3

Kaggle não corresponde a um único método. Além da família Heng/Tang, há registro denso com matching Húngaro e homografias locais, triangulação de pontos esparsos, indexação densa de linhas, matching clássico SIFT, matching ALIKED/LightGlue, correção neural de pontos em resolução alta e uma solução de recortes por derivação sem uma retificação não linear explícita.

A decisão metodológica mais importante é avaliar essas famílias separadamente. O segundo lugar diz usar os estágios 0 e 1 de Heng sem modificações. Seus avanços de máscara esparsa, fusão de derivações e processamento temporal não são um novo retificador. O M3 executou a adaptação de Tang, não o vencedor nem todos os métodos do Kaggle. [K002](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/2nd-place-solution), [K003](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/3rd-place-solution).

Há precedente explícito para a correção de bordas e da convenção de amostragem: o 15º descreve canvas fixo, normalização por `W−1,H−1` e `align_corners=True`, inclusive no trecho publicado de `grid_sample`. O mesmo trecho ainda usa replicação de borda, recorte da última coluna de nós e offsets empíricos. A medição independente do erro de meio pixel no M3 continua útil, mas a correção não é novidade nossa. [K015](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/15th-place-solution), [repositório](https://github.com/arunodhayan/physionetv1-kaggle).

## Matriz de mecanismos

As posições abaixo são as dos títulos, com as discrepâncias descritas acima. “Sem mudança geométrica” significa que não há uma mudança documentada no texto lido. Não significa que o código completo tenha sido provado idêntico. As condições exatas de ausência de pontos, interpolação e kernels não são publicadas em todos os textos. Os campos desconhecidos estão preservados no JSON.

| Fonte e autores | Evidência e geometria | Decisão reutilizável e limite |
| --- | --- | --- |
| [1º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/1st-place-solution), outrunner, Yijun, Arya, Bob, Shane Hsieh | Heng, centros ponderados, ajuste de superfície, remoção de outliers, remapeamento direto do RGB original | Precisão subpixel e uma única amostragem. Não há certificado de suporte ou de inversibilidade no texto |
| [2º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/2nd-place-solution), Takashi Someya | Heng 0/1 explicitamente sem modificações | Comparador da mesma família geométrica, extrator de sinal diferente |
| [3º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/3rd-place-solution), Tang | H e grade estimadas em resolução menor, aplicação em resolução alta | Preservar original. A cascata nativa e o decoder de grade foram auditados no M3 |
| [4º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/4th-place-solution), tomoon33 | Janelas 704, layout e cruzamentos, orientação pelo layout, 55/43 índices, votação local, campo denso | Resolução alta para detectar, rede separada para identidade. Os índices continuam ligados a um template |
| [5º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/5th-place-solution), Đăng Nguyễn Hồng | 57 âncoras e 2365 cruzamentos, pseudo-rótulos MINIMA-RoMa, matching Húngaro, H local K4/K16 ou spline | Melhor censo público de ablações de warping encontrado. H locais independentes exigem testes de seams |
| [6º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/6th-place-solution), AnnieGo | Base Heng, foco em regressão direta do sinal | FFT e soft-argmax pertencem ao sinal. Não são evidência de melhor mapa da imagem |
| [7º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/7th-place-solution), Harshit Sheoran e Bartley | Rotação, detecção de derivações, recorte em duas passagens, segmentação, regressão e gate OOD | Braço independente útil quando a grade desaparece. Não oferece saída RGB com geometria métrica certificada |
| [8º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/8th-place-solution), liuzhangzhen, D.Imanishi, James Day | Imanishi usa um remapeamento do original. James usa DINOv2, 30 landmarks e H. Liu conserva a base | Três subpipelines distintos no mesmo time. James registra resultados negativos para dewarping não afim e treino de warp fim a fim |
| [9º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/9th-place-solution), YumeNeko | ALIKED/LightGlue e H, quatro rotações, detecção sintética de grade, matching e BFS, piecewise affine | Separar matching inicial de propagação topológica. Moiré e fase incorreta de uma célula são ataques essenciais |
| [10º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/top-10-solution-giba), Giba | YOLOv12 com 22 landmarks. Duas famílias: cruzamentos com Delaunay e U-Net com 54/42 linhas indexadas | Comparar pontos versus curvas indexadas. O texto chama H de linear, mas a transformação descrita é projetiva |
| [11º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/11th-place-solution), zx2, fengxing, zacarioli | Heng e refinamento iterativo por espaçamento igual em x/y | Pode reduzir ruído em scans quase planos. Não impor espaçamento igual na fotografia antes de remover perspectiva |
| [12º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/12th-place-solution), pln | OCR, 17 landmarks, cantos estendidos, Delaunay. Outra rede deforma a máscara do sinal | Triângulos são uma família distinta de interpolação. Poucos nós não observam rugas finas. Warp da máscara não prova fidelidade do RGB |
| [13º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/13th-place-solution-public-lb-21-62-db-privat), Maruichi01 | Pré-processamento Heng | Centroide diferenciável e ResUNet 1D são posteriores à retificação |
| [14º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/14th-place-solution), sshiyu | Checkpoints 0/1 da base, HRNet no sinal | “Perfeitamente alinhada” é descrição do autor, sem validação geométrica publicada |
| [15º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/15th-place-solution), Arunodhayan e FatLiuyun | Canvas e convenção de pixel corrigidos na base | Precedência clara para preservação das colunas de borda e `align_corners` coerente |
| [17º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/17th-place-solution), MaxChen303 | SIFT com template mascarado e RANSAC. Grade em resolução original, centros ponderados, máscara de traço filtra falsos cruzamentos | Alternativa clássica ao detector de âncoras. Sujeita a template, texto repetido, deformação forte e falsos matches |
| [21º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/21st-place-solution), David List | Heng e cascata de duas CNNs em patches 31 para refinar pontos em resolução alta | 800 posições manuais de treino e 180 de validação, erro médio relatado 0,146 px. Não é medição independente nossa |
| [24º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/24th-place-solution), Rudr Pratap Singh e Uday Bhatia | Estágios públicos 0/1 | Atenção entre linhas e convoluções deformáveis pertencem ao extrator |
| [26º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/26th-place-solution), Boredom, Vampire, Shuyang Jiang, Large_Supper, BeeBoo | Heng 0, novo treino de grade, voto 3×3, preenchimento por vetores direcionais | Testar buracos e bordas com os mesmos pontos de entrada. Direção extrapolada deve conservar proveniência e incerteza |
| [27º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/27th-place-solution-only-pre-and-postprocessing), Zhihao Zhu, Van Dong Tran, BanhMiMatOng | Heng com dois caminhos, original e pré-processado, escolhidos por qualidade do estágio 1 | Gate de qualidade é útil. Sua fórmula completa não está no texto e o exemplo contém variáveis inconsistentes |
| [51º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/51th-solution), Ray, COLA KE LIU, HanCheng Li, JjaHe Li, Eddie Sun | H, linhas/cruzamentos e campo denso, sem detalhes suficientes do indexador | Arquitetura de três estágios. A melhoria documentada é preenchimento e suavização do sinal |
| [97º, sasa_leaf](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/physionet-97th-place-solution) | Heng e pré-processamento por fonte, demoiré antes da geometria | Recolorir linhas finas antes do extrator traz risco de alterar traço. Manter RGB original como referência |
| [97º, Evan, Momoko Yan, Cian Ron](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/104th-place-solution) | Pesos públicos, três vistas de intensidade | Diversidade de inferência, sem novo modelo geométrico documentado |
| [98º, Kirderf](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/physionet-98th-place-solution) | Classificador de fonte, pré-processamento e três estágios públicos | Novo pré-processamento não implica nova calibração geométrica |

## Ablação que merece reprodução

O 5º publica uma comparação com o mesmo backbone simples e validação do fold 0. O SNR inclui a derivação II curta, com 12,5 segundos totais para II. Portanto não é exatamente o mesmo agregado da competição. Não há novos resultados nossos nesta tabela. [Fonte primária](https://github.com/dangnh0611/kaggle_ecg_digitization/blob/8358d5b09a5b9519275e9b54c11716da5752a9d6/docs/ABLATION_STUDY.md).

| Warping e kernel | SNR publicado no fold 0 |
| --- | ---: |
| H global localmente selecionada, Lanczos | 21,545642 |
| H por célula K4, Lanczos | 22,929716 |
| H por célula K4, linear | 22,828297 |
| Spline bivariada, Lanczos | 22,846357 |
| H local K16, Lanczos | 23,086725 |

Outra tabela do autor compara K16 com perturbação aleatória de nós e relata 23,195719. Não misturar a diferença inteira com efeito exclusivo de K16. O autor motiva K16 como compromisso entre viés global e sensibilidade ao ruído K4. Isso torna o método um candidato forte. Não resolve sozinho continuidade, extrapolação ou domínio válido.

O 8º e o 21º observam que treinar o extrator com retificações imperfeitas pode ajudá-lo. Isso indica covariância entre retificador e extrator. Por isso o estudo deve ter tanto um extrator comum quanto o pipeline nativo completo. Um extrator que tolera erro não demonstra que o mapa esteja correto.

## Ataque específico a homografias locais e seams

Foi lido o [crop.py fixado](https://github.com/dangnh0611/kaggle_ecg_digitization/blob/8358d5b09a5b9519275e9b54c11716da5752a9d6/src/ecg/utils/crop.py). Para K maior que 4, ele seleciona vizinhos no índice da grade e estima H com USAC_MAGSAC. Se falhar, retorna a H4. O mapa denso escolhe uma H por célula com `searchsorted`. Não faz blend de H adjacentes. O campo pode receber GaussianBlur opcional, com sigma padrão zero. Fora do lattice, os índices são limitados à célula externa e essa H continua sendo avaliada. Não foi encontrada uma prova de continuidade ou de inversibilidade no texto nem nesse trecho.

**Inferência e controle analítico nosso, não resultado de checkpoint:** duas H4 que acertam os mesmos vértices da borda não precisam coincidir no interior dessa borda. Considere as células de destino `[0,1]×[0,1]` e `[1,2]×[0,1]`. A esquerda é identidade. A direita mapeia seus cantos para `(1,0),(2,0),(2.2,1),(1,1)`. Os dois quadriláteros fonte são convexos e partilham exatamente os nós da borda. Uma H da direita é

\[
H_R=\begin{bmatrix}1.2&-0.2&0\\0&1&0\\0&-0.2&1.2\end{bmatrix}.
\]

Na borda comum, `u=(1,0.5)`, a H esquerda produz `(1,0.5)` e a direita `(1,5/11)`. A descontinuidade tem tamanho `1/22`, cerca de 0,04545 unidade de célula. O cálculo com `cv2.getPerspectiveTransform` retornou 0,04545456 em float32. Isso ocorre com os nós exatos, sem erro do detector. K16 também não impõe igualdade entre mapas vizinhos, e pode não interpolar os nós em comum.

Não há motivo para banir H locais. Há motivo para não promovê-las a padrão só pelo SNR da ablação. Precisamos medir limites laterais nas bordas, deformação de traços, jacobiano e sobreposição global. Alternativas comparáveis são H global composta analiticamente com um campo residual contínuo, triângulos com vértices compartilhados e quads bilineares com nós compartilhados. Essas parametrizações resolvem a continuidade de posição quando construídas de modo consistente, mas também precisam de controles contra inversão, interseção global e subamostragem de rugas.

## Decisões de engenharia propostas

1. **RGB e evidência geométrica separados.** Grade ausente reduz observabilidade da métrica. Não autoriza multiplicar RGB pela máscara de grade. Manter imagem e máscara de qualidade independentes, inclusive para cabeçalho, calibradores e texto.
2. **Transformação composta uma vez.** Preservar a H global como transformação analítica e um residual local acoplado. Formar o mapa de amostragem do original uma vez, com origem de pixel explícita e kernel identificado.
3. **Identidade de grade com hipóteses.** Usar detector de cruzamentos, direções/curvas e offsets relativos, com labels absolutos como evidência de template. Manter hipóteses de passo, fase, orientação e layout quando houver ambiguidade. Não resolver D4 apenas pela grade quadrada.
4. **Refinamento subpixel independente.** Centros ponderados e patches de alta resolução são precedentes fortes. Medir erro das coordenadas em referências independentes, incluindo viés de rótulo e nível de borramento. Não selecionar ponto só pelo SNR do sinal.
5. **Extrapolação com proveniência.** Não converter ausência em coordenada fonte `[0,0]`. Separar ponto observado, interpolado, extrapolado e rejeitado. Conservar conteúdo visível por um fallback geometricamente explícito e marcar a confiança, sem chamar extrapolação de observação.
6. **Gate por observabilidade e risco geométrico.** Um seletor deve usar resíduo de observações retidas, mudança de escala, continuidade, dobra e quantidade de suporte. Um grid artificialmente regular ou uma foto visualmente bonita não é critério suficiente.
7. **Braço sem grade.** A solução de recortes do 7º merece comparação em documentos sem grade. A saída desse braço deve explicitar calibração desconhecida quando não houver cue físico suficiente. Não inventar mm a partir de pixels.
8. **Seleção por estrato e falhas no denominador.** Separar formato nativo compatível, mudança de layout e mudança de captura. Separar ganho geométrico de sinal, interpolação temporal, fisiologia e ensemble.

## Ataques e critérios de refutação

| Ataque | O que ele pode expor | Condição para aceitar a solução |
| --- | --- | --- |
| Grade removida localmente e traço preservado | RGB apagado por falta de confiança geométrica | Traço e demais pixels visíveis preservados, confiança independente |
| Buraco interior versus faixa exterior ausente | Interpolação confundida com extrapolação | Proveniência por pixel e curva erro versus cobertura |
| Deslocamento de uma linha inteira em componente desconectado | Erro inteiro de fase apesar de boa suavidade | Detecção de ambiguidade ou ancoragem sem usar o alvo |
| Grade fina versus grossa e pixels/mm anisotrópicos | Troca de passo, escala ou um eixo | Erro em mm por eixo e hipóteses de passo reportadas |
| Quatro rotações e quatro reflexões | Simetria de grade confundida com leitura clínica | Orientação correta com cue explícito ou abstenção |
| Layout 6×2, 12×1, Cabrera, cabeçalho e margens diferentes | Prior de posição confundido com observação | Experimento de formato separado do nativo, sem resgate por gold |
| Traço passando por cruzamentos e texto sobre grade | Falso nó e supressão de informação do sinal | Robustez do detector e preservação de RGB e lead identity |
| Ruído pequeno de nós e curva forte no papel | H4 nervosa ou H16 excessivamente rígida | Mesmo conjunto de observações, erro geométrico e seams medidos |
| Grade exata em duas células trapezoidais | Descontinuidade estrutural de H locais | Limites de cada lado da aresta compatíveis |
| Dobra aguda e auto-oclusão | Suavização apagando crease ou mapa duplicando superfície | Múltiplos charts ou incerteza explícita, sem fingir conteúdo oculto |
| Moiré perto da frequência da grade | Falsa identidade e aliasing após warp | Testar detector e RGB separadamente, medir preservação do traço |
| Deslocamento do traço em relação ao grid no scan | Grade perfeita e sinal desalinhado | Referência de traço e calibração não derivada apenas da aparência da grade |
| Minificação forte e linhas subpixel | Aliasing, ringing e desaparecimento de pico | Mesma geometria com kernels e footprints apropriados comparados |
| Bordas, dimensões ímpares e identidade | Meio pixel, perda da última coluna e padding copiado | Rampas de coordenadas, bordas inclusivas e identidade exata por convenção |
| Detector treinado no mesmo renderer da avaliação | Otimismo sintético | Banco externo ou gerador independente e separação de templates/câmeras |
| Extrator treinado com erro específico de retificação | Ganho de SNR mascarando erro geométrico | Mapa avaliado diretamente e extrator comum em braço independente |

## Disponibilidade e próximos comparadores

Há repositórios públicos dos lugares 2, 3, 4, 5, 6, 7 e 15 identificados. A visibilidade do repositório foi verificada para 2, 4, 5, 7 e 15, além dos checkouts M3 de Heng e Tang. Nem todo repositório possui licença identificada. O segundo alerta que seu repositório refatorado pode divergir do notebook. O 5º anuncia dataset de checkpoints e configs, mas não foi baixado aqui. Nenhum novo checkpoint de equipe foi executado nesta revisão.

Prioridade para um experimento fiel:

1. Manter Tang nativo do M3, sem trocar seu interpolador silenciosamente.
2. Reproduzir o 5º e verificar seus configs de checkpoint. Distinguir a configuração padrão K4 dos experimentos K16 e da configuração final realmente usada.
3. Executar o 4º como comparador de detecção em resolução alta com indexing separado.
4. Adicionar o 7º como braço sem retificação explícita de grade e avaliar saída de sinal em paralelo.
5. Implementar comparações de decoders com as mesmas observações para H global, H4, H16, spline, triângulos e campo residual contínuo. Nomear versões derivadas como ablações próprias.

Os métodos 9, 10, 12, 17 e 21 são referências metodológicas concretas, mas o writeup sozinho não fornece todos os pesos e detalhes necessários para uma reprodução nativa. Uma implementação nossa inspirada no texto não deve receber o nome de reprodução autenticada da equipe.

`sources.json` contém URLs, IDs oficiais, hashes de recebimento e nível de evidência. `census.json` conserva autores, posições, geometria, orientação, escala, suporte, amostragem e disponibilidade. `decision-proposals.json` transforma os pontos acima em hipóteses e critérios refutáveis. O cache de textos externos é apenas recibo interno de leitura e não integra os entregáveis.
