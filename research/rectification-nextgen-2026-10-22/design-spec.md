# Especificação da candidata

Proposta de pesquisa, anterior à implementação de uma nova rede. Este documento define interfaces, hipóteses e experimentos que permitem descartar componentes sem transformar um conjunto de técnicas em complexidade permanente.

## 1. Contrato de dados antes do primeiro treino

Guardar uma transformação explícita entre o sensor original, cada resize/ROI/tile e o plano físico do papel. Centros de pixels, normalização e `align_corners` precisam ter uma convenção única. Verificar composição com rampas de coordenadas, marcas nas bordas e pulsos de calibração, incluindo pixels de meia amostra. As implementações concorrentes não devem ser copiadas sem esse controle.

O schema mínimo de cada cena inclui: RGB original e hash, paciente/registro/layout, parâmetros físicos de impressão, transformações de câmera, domínios visíveis e ocluídos, IDs inteiros de cruzamentos, major/minor, coordenadas físicas e observadas, duas famílias de linhas e validade geométrica. Uma etiqueta de legibilidade ou um proxy de observabilidade deve ser separado da visibilidade geométrica. Blur altera a observação e não desloca o gold do traçado.

O cache atual não exporta por padrão todos esses alvos e o dataset de treino descarta IDs. Acrescentar heads antes de corrigir esse contrato produziria supervisão inconsistente. Exportar apenas labels efetivamente disponíveis. Não chamar um proxy de legibilidade de probabilidade real sem validação.

O renderer 3D atual produz 1024×704 e pode reduzir outra vez ao rotacionar/letterbox. Comparar cenas com a mesma geometria, luz, óptica e papel em aquisição 1024 e 2048, preservando campo de visão. Um braço com upsampling de 1024 controla o efeito de mais pixels sem informação adicional. Ajustar blur e ruído em unidades físicas/sensoriais coerentes, e medir custo. O gerador e a supervisão continuam em torch no device, conforme o contrato do projeto.

## 2. Observação global e local

Uma visão global estima papel/ROI, orientação aproximada, presença de grid e a necessidade de refinamento. A ROI conserva margem. Seus mapas retornam às coordenadas do sensor original. Headers e calibração podem estar fora do grid, portanto não definir saída pela caixa de pontos confiáveis.

O ramo local usa pixels nativos, overlap e fusão de probabilidades/offsets no sistema original antes de decodificar. Começar com interseções subpixel e linhas H/V. As duas famílias são coordenadas do papel e podem aparecer oblíquas ou curvas na fotografia. O traçado ECG nunca recebe loss de retidão.

O detector já é stride 1 e possui offsets subpixel. Priorizar resolução disponível, falsos cruzamentos causados pelo sinal, recall sob compressão e NMS. Comparar raio fixo, raio ligado ao pitch e propostas multiescala. Registrar truncamento por capacidade em cada imagem. Um controle analítico prova supressão de picos próximos, mas não escolhe o melhor decoder em fotos reais.

GroupNorm usa estatísticas espaciais, de modo que frame e tile não são equivalentes mesmo com halo. Treinar ambos os contextos ou comparar uma normalização compatível. Exigir ablação de normalização e contexto, em vez de prometer paridade por sobreposição. O limiar do heatmap deve ser tratado como ranking: BCE ponderada e targets gaussianos não fornecem uma probabilidade calibrada de localização correta.

Adicionar fase periódica, direções locais e vizinhança relativa somente após medir os ganhos de pontos + H/V. Classes absolutas de 55/43 linhas de um layout não atendem automaticamente tiras e páginas arbitrárias. Para labels relativos, manter a ambiguidade de origem e saltos entre componentes.

## 3. Geometria e identidade conjunta

Usar G: coordenadas físicas `(u,v)` em mm → pixels originais para amostrar a imagem. Quando necessário, F deve ser a inversa da mesma representação, sem uma segunda triangulação independente. Uma malha adaptativa com elementos simples é um ponto inicial depurável. B-spline é um braço concorrente, com auditoria digital própria.

Variáveis: vértices/parâmetros de G, IDs inteiros `z_i`, indicadores de outlier `o_i`, pitch/família de grid `s`, orientação e registro relativo entre componentes. Para cada detecção `x_i`, usar covariância de localização estimada/calibrada `Σ_i` e evidência de observabilidade. O score bruto não é uma variância. Manter os pontos como observações suaves permite corrigir ruído. B0 com anchors rígidos permanece congelado como controle.

Uma função objetivo inicial é:

```text
E = Σ_i (1-o_i) ρ((G(s z_i)-x_i)ᵀ Σ_i⁻¹ (G(s z_i)-x_i))
    + λo E_outlier + λl E_linhas + λc E_ciclos
    + λb E_curvatura + λj E_barreira
```

No braço com fase, acrescentar distância circular entre fase prevista e `exp(2πi F_a/s)`. A fase não determina o inteiro ausente. Linhas H/V contribuem somente onde o grid tem evidência, com pesos de observabilidade e perdas robustas. Retirar pontos, linhas ou regiões inteiras do ajuste fornece diagnóstico de generalização interna, embora não descubra um erro comum de escala em toda a folha.

Covariância contínua representa erro de localização dentro de uma hipótese. Não representa a alternativa discreta de um cruzamento ser major em vez de minor, nem o salto de uma linha inteira. Conservar um pequeno conjunto de hipóteses e seu desacordo físico. Textura periódica sem referência adicional pode não identificar o pitch em mm. Nesse caso o sistema precisa declarar a ambiguidade, em vez de inferir certeza pelo menor resíduo.

## 4. Solver com custo delimitado

1. Gerar hipóteses projetivas factíveis por amostragem robusta, incluindo LO-RANSAC/PROSAC/MAGSAC++ como ablações. Não escolher a variante com o gold de teste.
2. Fixar gauge de origem e orientação da representação. Separar essa convenção da tarefa automática de reconhecer lead, polaridade, tempo e escala.
3. No primeiro braço diagnóstico usar os mesmos IDs fixos para comparar least squares, pesos e IRLS. Isso isola o refinamento de posição.
4. No braço automático alternar associação discreta e ajuste contínuo sparse. Permitir outlier e componentes sem registro global identificável. Evitar descartar sempre todos os componentes exceto o maior.
5. Usar trust region/line search e barreira para manter a malha factível. Reavaliar associação quando evidências forem incompatíveis. Regularização não deve forçar uma ligação arbitrária sobre uma região sem observação.
6. Conservar hipóteses competitivas e propagar desacordo para a seleção. Retornar coordenadas parciais quando a observação não determina uma solução global.

Configurar `max_hypotheses`, `max_iterations`, `max_mesh_vertices`, `max_output_pixels` e `timeout_ms` no manifesto antes do benchmark. Estes limites dependem do profiling e não foram fixados por números imaginários. Toda execução registra iterações, hipóteses descartadas, tempo por estágio, motivo de parada e `budget_exhausted`. Timeout e truncamento permanecem no denominador.

## 5. Prior de documentos, dobras e oclusão

UVDoc e o baseline SyntheticDoc são priors opcionais de forma/localização, sujeitos a ablação sem prior. O checkpoint SyntheticDoc foi treinado com papel recortado e o README avisa sobre papel pequeno no frame. Seus meshes começam em A4. Uma ROI pode melhorar a entrada, mas escala e aspecto físicos de ECG devem ser recuperados por referências reais em ambos os eixos. Não tomar uma página visualmente reta como prova de mm corretos.

D2Dewarp fornece anterioridade forte para H/V. AADD já usa alinhamento axial e repetição após ROI. DvD opera em coordenadas, podendo atuar como teacher ou comparador de maior custo. Nenhum desses resultados em documentos estabelece sua qualidade em ECG. Selecionar um prior pela validação geométrica e pelo custo, não pelo ranking de OCR de outro domínio.

Regularização de curvatura deve permitir mudança de derivada em creases se os dados justificarem. Isometria física pertence a uma superfície S:(u,v)→R³ e a uma câmera declarada. Não impor `J_Gᵀ J_G=I` na projeção 2D: perspectiva altera ângulos e comprimentos no sensor.

Se papel se sobrepõe, partes distintas podem projetar no mesmo pixel. Um único mapa bijetivo da superfície inteira não representa essa oclusão. Usar domínios/charts visíveis, com associação e visibilidade explícitas. Não reconstruir sinal escondido como observação. Background fotográfico pode permanecer no frame original sem receber uma coordenada física de papel.

## 6. Sampler e saídas

Compor ROI, pose, prior e refinamento em um único mapa para o RGB original. Auditar a representação realmente usada pelo interpolador: orientação de elementos, degenerescência, borda, interseções e colisões globais. Um determinante positivo calculado por uma diferença central não basta para todas as transformações digitais. Um mapa e sua inversa podem estar perfeitamente consistentes e fisicamente errados.

Separar no resultado:

| Campo | Semântica |
| --- | --- |
| `original_rgb` | Decodificação original arquivada sem alteração, com hash |
| `rectified_proposal` | Reamostragem geométrica opcional e sua densidade/origem |
| `coordinate_map` | Coordenadas físicas estimadas e domínio de charts |
| `sampling_valid` | Fonte existe e pode ser amostrada |
| `evidence_provenance` | Observado, interpolado, extrapolado ou sem evidência |
| `coordinate_eligible` | Decisão calibrada ou heurística explicitamente identificada |
| `hypotheses` | Escala, orientação e IDs concorrentes, com desacordo |
| `status` | Original, parcial, proposta completa, abstinência, timeout ou inválido |

A proveniência não pode zerar RGB. Arquivar o original não garante que um sampler reduzindo a resolução preserve cada traço fino. Medir retenção de conteúdo e frequência de amostragem, além de escolher densidade de saída suficiente e filtro adequado. Sem orçamento, conservar o original e declarar a falta de retificação completa.

## 7. Experimentos que podem matar a candidata

| Experimento | Controle pareado | Critério de decisão |
| --- | --- | --- |
| Aquisição | Mesma cena física em 1024/2048 e upsampling de 1024 | Ganho deve vir de evidência e justificar compute |
| Observação | Pontos atuais, global/local, depois H/V | Precisão/recall por pitch, blur e papel pequeno, mais erro físico final |
| Refinamento | Mesmos pontos e IDs, anchors rígidos versus IRLS | Redução de erro a cobertura fixa, sem esconder detecções ruins |
| Associação | Gold+IDs, gold anônimo, CNN+associação gold, totalmente automático | Localizar gargalo sem usar oracle na trilha principal |
| Prior | Sem prior, UVDoc, SyntheticDoc, opcionais mais caros | Ganho externo em tiras/layouts novos e deformação severa |
| Geometria | Homografia, local K16, TPS, spline e malha com mesmas observações | Acurácia física, retenção e custo, não apenas round-trip |
| Incerteza | Sempre aceitar, gate comum e seleção calibrada | Mais cobertura no mesmo risco e menos risco na mesma cobertura |
| Generalização | Gerador, dispositivo, paciente e layout separados | Ganho não pode existir apenas no próprio renderer |

Rejeitar um componente se ele melhorar apenas resíduos internos, excluir casos difíceis, reduzir cobertura para quase zero, explorar gold na calibração final ou consumir mais recursos sem vantagem útil. Calibrar o candidato somente depois de congelar suas decisões. Qualquer fase M4 é condicional ao padrão de falhas de M2/M3.

## 8. Calibração e comparação externa

Os 96 replays anteriores estão em desenvolvimento. Separar pacientes de treino, escolha de arquitetura, calibração da seleção e teste final. Uma coorte pública pode ter sido usada por concorrentes, portanto registrar sobreposição por método. O subset Ahus já integra a validação do npj. Usá-lo para reprodução, sem apresentá-lo como teste inédito contra esses pesos.

O ECG-Image-Database 2026 e PM-ECG-ID permitem avaliar sinal com captura real. Pares imagem+sinal não fornecem gold denso da deformação. Criar landmarks físicos independentes em regiões com e sem grid, separados dos anchors usados no ajuste, com dupla anotação e incerteza. Não interpolar esses landmarks e chamar o resultado de verdade densa. Um piloto físico de 30 registros × 6 vistas pode orientar custos e poder estatístico, mas ainda não foi coletado e não determina sozinho o tamanho final.

Para seleção usar G, W, C e R=W/C. A forma padrão de Conformal Risk Control exige perda monotônica e unidades intercambiáveis. Ela não transforma automaticamente controle de W em garantia de R, nem cobre mudanças arbitrárias de domínio. Os seis pacientes atuais não sustentam promessas de erro de 1%. Para uma família finita de thresholds, considerar bounds simultâneos de W e C ou um método seletivo apropriado, sempre com os pressupostos declarados.

Fixar o extrator e as janelas/calibração na trilha que isola retificação. Medir os pipelines nativos à parte. Reportar sinal apenas no trecho impresso, inclusive convenção Cabrera/-aVR e páginas sequenciais. Publicar métricas sem ajuste e com o scorer de referência, pois tolerâncias de deslocamento podem esconder erro de tempo e baseline.

O protocolo versionado enumera comparadores, endpoints, falhas, margens propostas e profiling completo. Qualidade, latência e memória formam uma fronteira de compromisso. Se o candidato ganhar apenas em um deles, a afirmação final deve refletir isso.
