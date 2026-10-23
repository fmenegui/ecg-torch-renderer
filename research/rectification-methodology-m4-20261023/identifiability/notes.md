# Identificabilidade, ameaças e contrato de um retificador de ECG

Data: 23 de outubro de 2026. Escopo exclusivo: `m4/identifiability`. Nenhuma fonte, configuração, predição ou pontuação de M3 foi modificada.

O próximo método deve procurar a melhor decisão sustentada pela imagem, incluindo decidir que uma grandeza continua ambígua. Um retificador de uma única fotografia não consegue garantir uma reconstrução única para qualquer documento. Há casos com o mesmo RGB e diferentes coordenadas materiais, orientação ou conteúdo oculto. Há também falhas recuperáveis: o RGB contém a informação, mas o frontend ou o modelo descartou essa informação. Tratar essas duas situações da mesma forma prejudicaria a engenharia.

Os 16 controles congelados passaram, com 39 verificações. São testemunhas analíticas e controles de contrato, sem detector, sem nova inferência dos comparadores, sem avaliação real e sem estimativa da frequência dos problemas. Não demonstram superioridade, nem implementam o decoder proposto abaixo. Um `PASS` em um contraexemplo confirma a falha da garantia pretendida dentro das hipóteses declaradas.

## 1. Protocolo congelado e reprodução

Antes de executar qualquer controle, foram escritos `controls.py`, `protocol.json` e `protocol-freeze.json`. O congelamento foi registrado em 2026-10-23T11:08:56.078050+00:00. O resultado foi registrado em 2026-10-23T11:09:07.125646+00:00.

- SHA256 de `protocol.json`: `9fd242446e55a0b29071ed78f799967a28f0583954fe4a51b1a44875886d2ac9`.
- SHA256 de `controls.py`: `3a245c823e98c2b918656736c7cff27ea337453b6afd4bb5bcd239f76739fcca`.
- `results.json` retém cada caso, suas condições, seus asserts e hashes dos arrays. Os 16 arquivos `witnesses/ID*.npz` contêm as observações e os alvos das testemunhas. Nenhum caso falhou ou foi removido.
- Python 3.12.14, NumPy 2.3.5, CPU. Não há resultado de GPU, custo de inferência ou desempenho de produto.
- `independent_audit.py` foi escrito separadamente, sem importar ou chamar `controls.py`, e congelado em `audit-protocol.json` antes de sua execução. As 36 verificações passaram em `independent-audit.json`. A auditoria reconstrói os rasters de ID01 e ID04/ID05 a partir de unidades materiais e arcos de segmentos 3D, além de conferir hashes e invariantes. É uma referência analítica independente do código dos controles, escrita pelo mesmo agente, e não revisão por outra pessoa ou simulação fotográfica independente.

```bash
/workspace/scratch/a57599a3e70e/swarm-venv/bin/python \
  /workspace/scratch/a57599a3e70e/m4/identifiability/controls.py \
  --out /workspace/scratch/a57599a3e70e/m4/identifiability/reproduced-results.json \
  --witness-dir /workspace/scratch/a57599a3e70e/m4/identifiability/reproduced-witnesses
```

Os hashes de arrays são estáveis. Um arquivo NPZ comprimido pode ter metadados de contêiner diferentes em outra execução, por isso a auditoria deve comparar também os hashes dos arrays. Não substituir o resultado original ao reproduzir. Mudança de fonte exige novo protocolo e preservação dos resultados anteriores.

Fontes locais lidas: `ecg-torch-renderer/AGENTS.md`, `m3/our-critique/critique.md`, `m3/our-critique/decision-matrix.json`, `m3/benchmark/protocol.json` e `m3/benchmark/final-interpretation-pt.md`. A interpretação preserva a distinção entre D4 escolhido pelo avaliador e orientação inferida automaticamente. Também preserva a distinção entre o painel sintético já inspecionado e um novo holdout.

## 2. O que pode ser identificado

Uma testemunha de não identificabilidade tem dois mundos admissíveis, a mesma observação disponível e alvos diferentes. Nenhum algoritmo determinístico que receba apenas essa observação pode acertar ambos. Um algoritmo aleatório também não adquire informação sobre qual mundo foi observado. Um prior pode preferir o mundo mais comum, mas não transforma essa preferência em uma medição.

O conjunto de mundos admissíveis é essencial. Assumir folha plana, grid quadrado conhecido, câmera calibrada ou texto legível pode eliminar uma ambiguidade. Esses pressupostos precisam ser evidência ou condições explícitas da saída. Não se deve generalizar um contraexemplo de mapas 2D arbitrários para todo papel sem estiramento. Tampouco se deve chamar uma perda do detector de impossibilidade da fotografia.

| Controle | Resultado e implicação | Informação adicional capaz de ajudar |
| --- | --- | --- |
| ID01, unidade 1/5/10 mm | O mesmo grid de passo 20 px representa passos físicos 1, 5 ou 10 mm com magnificação diferente. A consulta `[100,80]` px vale `[5,4]`, `[25,20]` ou `[50,40]` mm | Legenda/calibrador físico, família do papel conhecida, aparência major/minor realmente distinguível, linhas intermediárias |
| ID02, D4 | Oito poses de um grid quadrado puro produzem exatamente o mesmo RGB e oito alvos distintos | Texto assimétrico confiável, fiducial com identidade, direção e quiralidade, metadados de captura |
| ID03, linhas saltadas | Dois ajustes dos mesmos pontos têm resíduo e roundtrip zero. Chamar um salto real de 10 mm de 5 mm produz erro 17,67767 mm na consulta | O RGB deste controle ainda contém a linha intermediária. Um decoder que retorne às linhas pode corrigir a perda do frontend |
| ID04, fase entre ilhas | Uma ponte oculta pode ter comprimento material 20 ou 25 mm e a mesma projeção. A segunda ilha muda exatamente um período de 5 mm | Ponte visível e indexada, geometria 3D conhecida, hipótese plana sustentada que exclua a dobra, outra vista |
| ID05, dobra em área branca | A consulta em uma faixa branca vale 30 ou 32,5 mm em dois papéis desenvolvíveis sob câmera ortográfica e radiância constante | Grid ou marcas na faixa, sombra/iluminação conhecida, limite físico de superfície, informação 3D ou outra vista |
| ID06, interior sem evidência | Uma deformação C1 preserva todas as bordas da célula e desloca seu centro em 1 mm, com determinante maior que 0,37168 | Restrição física mais forte e validada, observações internas, limite de curvatura que produza uma cota de erro |
| ID07, recorte | A forma relativa é idêntica, mas a origem pode diferir por `[15,25]` mm | Proveniência do recorte, marcador temporal, referência de posição, identificação de janela e baseline de cada derivação |
| ID08, texto espelhado ou invertido | O rótulo `II` em fonte simétrica não distingue espelho horizontal nem rotação 180°. Um `F` assimétrico de orientação conhecida tem oito vistas distintas | OCR de texto assimétrico, convenção de orientação sustentada e análise explícita de reflexão |
| ID09, dobra com sobreposição | Dois pontos materiais, u=5 e u=15,62832 mm, projetam no mesmo ponto. Só a camada da frente aparece | Visibilidade por chart, outra vista ou registro anterior. Não há inversa global única para as duas camadas |
| ID10, oclusão opaca | Alterar 900 valores de canais sob oclusor não altera nenhum pixel capturado | Outra observação sem a oclusão ou fonte original. Preencher por IA não recupera evidência observada |
| ID11, rótulos contraditórios | Metadados com conjunto único I/II e leitura confiável II/II produzem zero atribuições exatas e duas melhores atribuições empatadas | Proveniência/layout correto, revisão de OCR, identificadores e janelas das faixas. II repetido pode ser legítimo em uma tira de ritmo |
| ID12, fotocópia | Um grid quadrado de 5 mm esticado 1,1×0,9 e um grid originalmente 5,5×4,5 mm podem ter o mesmo passo 22×18 px | O prior confiável de grid original quadrado de 5 mm resolve a deformação da cópia. Não confundir mm da cópia com mm do documento original |
| ID13, página composta | Dois painéis com passo 20 px, mas calibração conhecida 5 e 10 mm, não admitem uma única escala compartilhada. Melhor ajuste linear erra ±2,5 mm por intervalo | Charts e unidades por painel, layout, namespace de rótulos, vínculo temporal explícito |
| ID14, velocidade e ganho | O mesmo traçado desenhado pode representar velocidades 25/50 mm/s e ganhos 10/20 mm/mV | Legendas, pulso de calibração de amplitude/duração conhecidas e metadados confiáveis. Grid correto não fornece sozinho s/mV |
| ID15, preservação | A caixa sem grid contém 137 pixels de tinta. Usar suporte de grid como máscara RGB apaga todos. Fallback identidade preserva os canais exatamente | Política que separa suporte geométrico de presença de conteúdo |
| ID16, controle positivo | Quatro fiduciais físicos ordenados reduzem 24 candidatos de passo/pose a um só. A consulta resulta `[10,25]` mm com erro numérico menor que 2×10^-14 mm | Aqui os fiduciais são entrada observada. Não são gold usado para escolher o resultado de uma fotografia de ECG |

Limite importante dos arquivos RGB: nos controles de igualdade, o raster observável foi construído para ser igual. A auditoria escrita separadamente reconstrói essa igualdade a partir da métrica material nos controles de unidade e ponte física. Não rodamos um simulador fotográfico independente de papel/iluminação para gerar os dois mundos. A consistência dos mundos é estabelecida pelas equações declaradas, pelo renderer analítico de referência e pelos arrays, não por uma alegação de realismo da captura. ID12 verifica igualdade da observação de lattice, não gera uma fotografia de fotocópia. ID11 e ID13 são controles de restrições/modelo com metadados de entrada, não impossibilidades do RGB.

### Dobras físicas que fundamentam ID04 e ID05

O eixo projetado da ponte vai de x=20 a x=40 mm. A versão plana tem 20 mm de arco. A versão V tem vértices 3D `(20,0,0)`, `(30,0,7.5)`, `(40,0,0)`, extrudidos no eixo y. Cada metade tem comprimento `sqrt(10²+7.5²)=12.5` mm. É uma superfície desenvolvível por partes. A câmera ortográfica observa x e y, por isso os extremos projetados são idênticos. O material à direita ganha 5 mm de arco. Como esse ganho é um período inteiro do grid maior, as linhas visíveis à direita têm a mesma fase de imagem.

ID04 cobre a ponte com um oclusor. ID05 imprime uma faixa branca sem grid e assume radiância constante, de modo que a inclinação da faixa não adiciona evidência fotométrica. No meio da ponte, a coordenada material é `20+1.25×10=32.5` mm na versão dobrada e 30 mm na plana. Uma fotografia com sombras, textura ou reflexo informativo pode distingui-las. Esse controle não diz que todas as áreas sem grid são impossíveis, nem que uma sombra identifica uma superfície única sem hipótese de iluminação.

### Bordas não fixam qualquer interior

No quadrado `[0,5]²`, compare `F0(x,y)=(x,y)` com:

`F1(x,y)=(x + sin²(πx/5)sin²(πy/5), y)`.

Toda borda tem deslocamento zero. O centro desloca 1 mm. O determinante é `1+(π/5)sin(2πx/5)sin²(πy/5)`, logo é pelo menos `1-π/5 > 0`. Como y é preservado e x é estritamente crescente em cada linha, o mapa é injetivo e preserva orientação. A classe aqui inclui deformação material 2D e não impõe papel sem estiramento. Esse contraexemplo impede certificar um interior arbitrário apenas por cantos/roundtrip. Um modelo físico restrito pode produzir uma cota melhor, que precisa ser testada separadamente.

### Calibração física, origem e sinal são variáveis diferentes

O retificador deve distinguir mm no papel original, mm na fotocópia, segundos e mV. O grid original quadrado de unidade conhecida permite corrigir anisotropia de cópia no sistema original. Não revela automaticamente o tamanho físico da nova folha ou os parâmetros da câmera. Velocidade e ganho continuam separados. No traçado de ID14, um período espacial de 25 mm e amplitude espacial de 5 mm correspondem a período temporal 1 ou 0,5 s e amplitude 0,5 ou 0,25 mV, respectivamente.

Uma translação de origem da folha não prova que o sinal mudou de baseline. Ela prova que o zero absoluto não foi observado. Converter uma origem arbitrária de canvas em zero temporal ou em baseline de uma derivação seria um erro adicional do digitizador. O gauge de translação de M3 foi apropriado para comparar forma geométrica e não demonstra identificação automática dessas referências clínicas.

## 3. Escada de evidência

| Nível | Evidência exportável | O que não autoriza afirmar |
| --- | --- | --- |
| E0, fonte | Hash, dimensões, RGB completo decodificável, proveniência, cadeia de recortes | Que o raster retificado conserva tinta subpixel |
| E1, observações | Pontos/linhas/caixas de texto, covariâncias ou alternativas, pixels de suporte e visibilidade | Que um ponto é major de 5 mm ou que seu ID está certo |
| E2, topologia relativa | Famílias de linha, vizinhança, saltos inteiros, ciclos, componentes, hipóteses de fase | Que não há alias coerente, unidade física ou orientação semântica |
| E3, unidades | Candidatos 1/5/10 mm e demais famílias plausíveis, evidência major/minor, calibrador ou metadados com origem verificável | Que a velocidade/ganho ou mm da cópia e do original sejam iguais |
| E4, pose e semântica | Orientação e reflexão, OCR assimétrico, rótulos/lead IDs, janelas temporais, contradições | Que todo texto esteja correto ou que prior de layout possa sobrescrever tinta |
| E5, geometria condicional | Modelo, parâmetros, charts, regiões observadas/interpoladas/extrapoladas, regularidade e erro de validação espacial da entrada | Que validação interna seja erro físico independente ou que conteúdo oculto foi recuperado |
| E6, risco externo | Erros físicos independentes por domínio, cobertura/risco, casos extremos, calibração em dados novos | Que uma distribuição validada certifique qualquer nova fotografia ou represente todos os concorrentes |

Proveniência de evidência deve ser um dado, não apenas uma string `trusted`. Registrar sua localização na entrada, leitura, hipótese alternativa, fonte, hash e regra de utilização permite auditoria. Duas evidências derivadas dos mesmos pontos não são independentes por terem nomes distintos. Um classificador treinado pode fornecer um prior útil, mas ainda precisa de calibração externa e mecanismo para candidatos contraditórios.

## 4. Contrato de saída recomendado

Cada chart e, quando necessário, cada região exporta estados separados para `geometry`, `scale`, `orientation`, `origin`, `signal_calibration` e `support`. O vocabulário para resolução é `resolved`, `ambiguous` ou `unsupported`, acompanhado de condições, razões e candidatos sobreviventes. `resolved` significa resolvido dentro do modelo declarado e da tolerância, com evidência listada. Não significa certificação universal ou garantia clínica. Contradições são uma flag explícita de qualidade de dados e não são escondidas pela escolha de um candidato.

O tipo de suporte espacial é separado do estado: `observed_anchor`, `observed_line`, `interpolated`, `extrapolated`, `unknown` e `occluded`. Incluir distância às evidências, chart, domínio visível e condição de inversão. Não tratar `interpolated` como automaticamente correto nem `no_grid` como equivalente a `occluded`.

Campos mínimos:

- `source_rgb` com hash e pixels originais, cadeia de transformação e estado de exportação. Validar decodificação completa ao salvar.
- `operation`: RGB efetivamente reamostrado, original preservado, coordenadas parciais ou entrada fora do modelo. Uma proposta de mapa não conta como RGB entregue.
- `coordinate_convention`: centros de pixels, x/coluna, y/linha, ordem dos eixos, unidade, D4, origem e qual documento físico a unidade descreve.
- `charts`: limites, relações conhecidas ou ambíguas, fase relativa candidata, visibilidade, unidade e orientação. Não unificar ilhas por conveniência.
- `maps`: direção, suporte, validade numérica, domínio de inversa única e candidatos. Coordenadas em mm ficam ausentes quando a unidade não foi sustentada. Um chart relativo pode ser exportado em unidade abstrata de lattice, com esse nome explícito.
- `evidence`: referências à entrada, classe de grid, saltos, calibradores, OCR/layout, pressupostos físicos e contradições.
- `quality`: checks numéricos internos separados de evidência física e de risco calibrado. Probabilidade sem validação fica marcada como score, não confiança física.
- `sampling`: densidade, filtro, footprint, orçamento e evidência de preservação de traços. Manter RGB original mesmo quando a proposta falha.

Gates que devem rejeitar uma alegação de retificação física resolvida:

1. Unidade escolhida somente por resíduo, convexidade, roundtrip ou ciclos. ID01 e ID03 sobrevivem a esses checks.
2. Pose escolhida apenas pelo grid quadrado. ID02 preserva D4. Um token simétrico não basta, conforme ID08.
3. Fases de ilhas unificadas sem ponte observada ou hipótese global sustentada. ID04 mantém um deslocamento inteiro real.
4. Cota de erro no interior deduzida só de cantos. ID05/ID06 mostram o pressuposto adicional necessário.
5. Inversa única declarada em um domínio de sobreposição ou conteúdo oculto declarado observado. ID09/ID10 exigem visibilidade e perda irrecuperável.
6. Layout ou ganho únicos forçados apesar de contradições e metadados ausentes. ID11/ID14 preservam essas incertezas.
7. Escala compartilhada de página aplicada a painéis incompatíveis. ID13 exige segmentação/modelo diferente.
8. RGB multiplicado pela confiança do grid. ID15 contém conteúdo perfeitamente visível onde o grid não existe.

Fallback: conservar o original e os charts/coordenadas sustentados. Se existir uma proposta visual útil, identificá-la como proposta com seus candidatos/limites, sem elevar sua unidade ou pose a resolvida. Uma orientação arbitrária pode servir à visualização geométrica, desde que permaneça declarada ambígua e não vire leitura clínica. Uma ROI de documento pode definir o footprint de uma saída, mas deve ser explícita e não derivada da ausência de grid como se ausência de grid significasse ausência de conteúdo.

## 5. Decoder conjunto mais forte a investigar

Esta é uma hipótese de arquitetura a testar, não uma nova configuração validada.

**Observação rica e reversível.** O frontend deve exportar interseções, segmentos de linha e suas tangentes, regiões major/minor candidatas, traços pontilhados, papel, oclusão, texto e creases. Guardar patches e covariâncias permite voltar ao RGB quando uma hipótese discreta entra em conflito. Uma frequência de Fourier isolada pode ser harmônico/sub-harmônico. Conservar candidatos de periodicidade e verificar linhas intermediárias evita decidir o inteiro cedo demais.

**Grafo discreto com múltiplas hipóteses.** Os latentes são chart, IDs inteiros de linha/coluna, saltos, família/unidade de grid, pose D4, fase de componente, outlier e visibilidade. Arestas seguem evidência local de linhas e perspectiva, não apenas uma direção global e distâncias Euclidianas fixas. Fatores incluem continuidade de linha, correspondência recíproca, fechamento de ciclos, evidência de major/minor e compatibilidade de layout. Ciclos são necessários, mas a hipótese coerente em meia escala também fecha ciclos. Usar feixe limitado de hipóteses e relabeling é uma aproximação computacional a medir. A distância entre scores não é posterior calibrado por definição.

**Contínuo condicionado ao discreto.** Alternar relabeling/outliers e ajuste geométrico robusto, em vez de congelar IDs e apenas suavizar pontos. Comparar H global para plano, composição compartilhada `T∘H` para residual suave e charts ligados por creases para superfície por partes. A inversa deve usar a mesma composição e a mesma estrutura. Não substituir isso por homografias independentes por célula, pois vértices compartilhados não garantem a mesma parametrização na aresta. Penalizar complexidade e selecionar por evidência da entrada em blocos espaciais ou linhas retidas, sem consultas gold. Resíduo de linhas de entrada continua sujeito a erro comum do detector e não se torna metrologia independente.

**Dobras e oclusão explícitas.** Permitir descontinuidade de derivada nas creases e fragmentação do domínio visível. Restringir regularização onde há evidência, sem alisar universalmente através de uma dobra. Usar charts locais orientados e camada/visibilidade quando a projeção for muitos-para-um. A parte visível pode ser retificada parcialmente. Nenhum warp de RGB único revela a face escondida. Multivista ou profundidade são novas observações e precisam de contratos próprios.

**Pose, layout e calibração em conjunto.** Avaliar rotação/reflexão de palavras assimétricas, relação entre texto e traços, layouts e janelas de tempo. Rótulo II isolado, números simétricos ou uma moldura retangular não resolvem quiralidade. Legendas e calibradores físicos podem vincular unidade e ganho, com procedência explícita. Consistência fisiológica pode ser um prior fraco se as janelas forem comparáveis. Não normalizar polaridade, derivação ou onda para parecer fisiologicamente típica. Uma patologia ou sinal incomum não é autorização para modificar o ECG.

**Preservação como produto próprio.** Amostrar todo conteúdo visível dentro do footprint com filtro e densidade medidos em controles de traços finos, texto e calibração. Grid insuficiente afeta validade geométrica. Não define transparência ou branco do RGB. Medir geometria, operação de raster e fidelidade de traço separadamente. Os controles anteriores de sampling de M3 motivam um candidato, mas não estabelecem uma densidade universal suficiente.

## 6. Anotações e treinamento que faltam

| Anotação nova | Por que é necessária |
| --- | --- |
| Família de papel/grid, major/minor, interseções com IDs e saltos reais | Treinar decisão de unidade e omissão coerente, não apenas localização de pontos |
| Componentes materiais, vínculo entre ilhas e conjunto de fases admissíveis | Distinguir ausência de ponte de componente realmente independente |
| Unidade do papel original, unidade da cópia, impressão/esticamento e metadados de velocidade/ganho separados | Evitar que uma hipótese de impressão vire uma medição do sinal |
| Orientação semântica e reflexão, tokens assimétricos, texto difícil e rótulos contraditórios | Ensinar D4 completo e abstenção em vez de orientação arbitrária de grid |
| Chart, creases, camada visível, oclusor, face escondida e correspondências muitos-para-um | Não treinar uma inversa bijetiva inexistente |
| Suporte observado, interpolado, extrapolado e sem grid com tinta | Aprender geometria sem aprender a apagar margens/legendas |
| Lead ID, faixa, ordem, janela temporal, recortes e baseline de cada faixa | Separar rótulo repetido legítimo de erro e preservar sequência/derivação |
| Geometria física independente em capturas reais, incluindo medidas entre âncoras | Avaliar mm e caudas sem usar o mesmo gerador/modelo como juiz |
| Pares de mundos observacionalmente equivalentes e alvos em conjunto | Não punir uma saída ambígua correta por não escolher o único gold arbitrário |
| Motivos de rejeição e entradas fora de escopo, com anotação de contradição | Medir cobertura/risco e falsa segurança em condições ausentes do treino |

Para cada novo bloco, separar erro de frontend, erro discreto de identidade/unidade, erro contínuo de modelo, oclusão e perda de sampling. Avaliar logits locais não substitui avaliar os IDs e o mapa final. Treinar com alvos em conjunto ou equivalências conhecidas quando a observação não identifica um único alvo. Métricas de probabilidade exigem distribuição explicitada e calibração externa. Um score alto não permite violar uma equivalência exata.

Dividir dados por paciente e também por template, fonte, impressora, câmera, família de grid, família de deformação e mecanismo de cópia. Manter combinações de dificuldades fora do treino, incluindo baixa resolução + ausência de grid + reflexo + rótulos parcialmente cortados. Adicionar um conjunto real independente com quantidade suficiente dentro do escopo de cada comparador. O painel M3 foi consumido como diagnóstico e não se torna holdout após inspirar esses ajustes.

## 7. Ameaças adicionais sem experimento neste pacote

Este pacote não fecha todos os corner cases. `cornercases.json` registra também ameaças ainda não controladas: grid não padrão, padrões pontilhados e duplos, cor removida pela cópia, moiré, line hopping sob blur, rolling shutter, lente forte, papel molhado/esticado, frente/verso vazando, folhas sobrepostas, página colada, ausência total de grid, velocidade/ganho mistos por faixa, dois tempos superpostos, texto manuscrito e calibrador cortado. Nenhum desses itens recebe resultado experimental por estar listado.

A decisão imediata mais forte é atacar unidade, associação discreta e pose, mantendo o bom contrato de fonte/saída e uma hipótese projetiva simples quando sustentada. A avaliação deve tentar refutar cada componente novo com casos negativos. Um método melhor é aquele que corrige mais casos recuperáveis e evita certeza falsa nos irrecuperáveis, sob orçamento e domínio declarados.
