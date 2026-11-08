# Crítica independente da metodologia de retificação

Data: 23 de outubro de 2026. Código auditado: `f4f6caef3b59b5830e93284cccadbc6855fc701e`, PR #21 de `fmenegui/ecg-torch-renderer`.

## Parecer

Manter os contratos de preservação de RGB, malha compartilhada e rastreabilidade. Não promover M2 como solução geral de retificação. O ganho observado justifica conservar o refinamento como opção experimental, mas a próxima decisão metodológica deve atacar identidade, pitch e organização da malha antes de simplesmente aumentar a suavização.

A pipeline tem hipóteses fortes em seu estágio discreto. Interseções são tratadas como vizinhas de um grid maior de 5 mm, com uma orientação e um espaçamento globais aproximados. Só o maior componente sobrevive. O refinador modifica posições mantendo os IDs. Uma organização coerente, mas fisicamente errada, pode atravessar todo esse caminho com resíduo pequeno. A proteção de curvatura evita alguns danos de suavização, mas também pode conservar exatamente a falha que precisava ser corrigida.

Essa crítica não compara a qualidade comercial de Dotter ou Gridder e não atribui arquitetura a métodos externos ainda não verificados pelos outros agentes. Ela estabelece os testes que qualquer candidato, inclusive o nosso, precisa enfrentar. Uma implementação inspirada em uma publicação deve ser identificada como reconstrução, nunca como execução do produto original.

## Evidência e escopo

| Tipo | Evidência utilizada | Limite |
| --- | --- | --- |
| Código | Fontes exatas dos módulos geométricos, CLI e avaliador no commit acima | Inspeção estática não mede prevalência de falhas |
| Experimento M2 | README e revisão final congelados no PR #21 | Oito pacientes novos ao estudo anterior, mesmos geradores e detector |
| Controles independentes desta revisão | `structural_controls.py` e `structural-controls.json` | Sem rede, sem RGB, sem comparação com concorrentes |
| Controles anteriores | `research/rectification-nextgen-2026-10-22` e revisão adversarial anterior | Contraexemplos de premissas, sem reivindicação de novidade |
| Propostas | Testes e decisões abaixo | Não são melhorias já validadas |

Os controles novos carregam os módulos auditados em um namespace isolado e registram hashes SHA256 e versões de NumPy, SciPy e Python. Não modificam o checkout de produção nem o estudo M2.

Para reproduzir os controles após integrar o script ao repositório, executar da raiz com `--geometry-dir ecg_gpu/digitization` e `--out` apontando ao JSON desejado. O script recusa fontes cujos hashes diferem dos módulos M2 congelados. Os controles passaram com Python 3.12, NumPy 2.3.5, SciPy 1.17.0 e OpenCV 4.11.0 no ambiente desta revisão. A execução com asserts e a compilação Python passaram. Ruff não estava instalado nesse interpretador, portanto esta revisão não registra aprovação de lint.

## 1. O que deve permanecer

**Preservação da fonte.** A imagem RGB original continua arquivada e é a fonte do sampling. Evidência de grid e elegibilidade geométrica são saídas separadas. Ausência de grid não autoriza apagar texto, traços ou margens. Essa correção é um contrato importante de engenharia e vale para todos os concorrentes.

**Uma mesma malha para forward e inverse.** `CompleteMap` constrói as duas direções com o mesmo complexo de triângulos. Isso elimina uma classe de inconsistência causada por triangulações independentes. Células observadas usam diagonais e IDs explícitos, sem substituir uma aresta conhecida por uma conexão Delaunay através de um buraco.

**Fallback rastreável.** O CLI distingue original preservado, coordenadas parciais e proposta reamostrada. Arquiva detecções originais, candidato refinado e coordenadas efetivamente usadas. Rejeição do refinamento usa checks de entrada, sem consultar gold.

**Avaliação pareada.** M2 mantém detector, IDs, consultas e gauge de B0 comuns entre as forças. G, W, C e R evitam confundir abstinência com acurácia. Os intervalos reamostram pacientes inteiros e não tratam milhares de pixels como milhares de pacientes.

Esses pontos tornam a experiência auditável. Não tornam verdadeira uma correspondência de grid errada nem certificam fidelidade do ECG após reamostragem.

## 2. P0: identidade e escala são o gargalo prioritário

### 2.1 Cinco milímetros são uma premissa, não uma medição

O CLI e o avaliador usam `ids[:, ::-1] * 5.` como coordenadas canônicas. O próprio relatório do CLI chama isso de `assumed_major_grid_mm`. A organização local estima o passo aparente mais próximo e o chama de um incremento inteiro. Não há uma decisão explícita entre grid menor, grid maior e submalha com linhas perdidas.

Um conjunto periódico de pontos não identifica, sozinho, o número de linhas ausentes entre dois pontos. Homografia, convexidade, orientação preservada, ciclos consistentes e round-trip exato podem coexistir com pitch físico incorreto. M2 não pode resolver isso mantendo os IDs fixos.

**Controle executado C01.** Em uma folha plana com grid maior de 5 mm, retiramos todas as linhas e colunas de índice ímpar da observação. Restam 36 pontos perfeitos com passo físico de 10 mm. `organize_lattice` aceita IDs consecutivos, sem conflitos de ciclo. B0, força 0,15 e força 1 conservam a escala errada. Na força 1, o refinador informa `projective observations already consistent`. Uma consulta com verdade `[25, 25]` mm é mapeada para `[12.5, 12.5]` mm, com erro de **17,67767 mm**.

Isso é uma injeção deliberada de omissão coerente no frontend. Não estima a frequência com que nossa rede ou uma concorrente produz tal padrão. Mostra que, se isso ocorrer, as verificações estruturais atuais não detectam a falha física. O problema não é resolvido por um solver contínuo melhor.

**Decisão.** Manter hipóteses de passo inteiro, família major/minor e outliers. Usar evidência independente do próprio ajuste, como presença de linhas intermediárias e aparência major/minor. Se hipóteses fisicamente diferentes continuarem compatíveis com as observações, declarar escala desconhecida ou ambígua. Não pedir ao método que escolha uma unidade impossível de identificar a partir de pontos idênticos.

### 2.2 Maior componente único descarta evidência válida

`organize_lattice` constrói componentes e retém somente o de maior número de pontos. Uma dobra, mancha, recorte ou perda de grid pode separar uma mesma folha em ilhas. A informação de uma ilha menor desaparece do ajuste, mesmo que suas coordenadas locais sejam úteis.

**Controle executado C02.** Duas regiões 7×7 pertencem à mesma folha plana, separadas por quatro colunas sem observações. Todos os pontos restantes são perfeitos. O organizador encontra dois componentes e conserva **49 de 98 pontos**, ou 50%. O RGB não é apagado. A perda é de evidência e coordenadas.

Conservar componentes não significa forçar um registro global entre eles. A origem relativa pode ser ambígua sem linhas intermediárias ou um modelo global suficientemente sustentado. Cada ilha pode precisar de chart e status próprios.

**Decisão.** Reter múltiplos componentes e a incerteza da ligação. Não inventar o inteiro que falta e não descartar automaticamente uma ilha só por ser menor.

### 2.3 Base global e janelas fixas falham antes da rede

O organizador usa uma orientação obtida de `exp(4j*angle)`, espaçamentos globais, bandas fixas de aceitação e até 12 vizinhos por ponto. Isso é uma aproximação razoável para perspectiva moderada. Não é uma organização geral de lattice deformado.

**Controle executado C03.** Aplicamos uma homografia finita a 192 pontos perfeitos, sem ruído, oclusão ou ponto falso. O denominador é `1 + k*x`, positivo em toda a folha. O resultado é:

| k por pixel | Status nativo do lattice | Pontos no maior componente | Componentes | Conflitos de ciclo |
| ---: | --- | ---: | ---: | ---: |
| 0 | aceito | 192/192 | 1 | 0 |
| 0,001 | aceito | 192/192 | 1 | 0 |
| 0,002 | rejeitado | 138/192 | 47 | 17 |
| 0,004 | rejeitado | 116/192 | 72 | 27 |
| 0,008 | rejeitado | 78/192 | 87 | 24 |

O wrapper `locally_supported_cells` ainda pode aproveitar células parciais quando o organizador nativo rejeita. Logo, esses números não são taxas de falha completa de RGB. São uma demonstração de que melhorar somente a detecção não elimina o gargalo de correspondência. A homografia é um controle algébrico, sem pretensão de provar a distribuição de perspectivas de fotografias reais.

**Decisão.** Comparar inferência de orientação e passo locais com crescimento de lattice e múltiplas hipóteses. Medir o organizador com pontos perfeitos e com pontos de rede separadamente. Não usar o F1 de detecção como substituto da qualidade final de retificação.

## 3. P0: elegibilidade não é certificado físico

`CompleteMap` interpola as âncoras e preserva as células conhecidas. Depois, `eligible_geometry` compara esse mapa com `cell_coordinates`, que compartilha as mesmas âncoras e triângulos. Os pontos usados em `point_error` também são vértices interpolados. Esse acordo é um check de consistência da implementação, não uma testemunha independente da localização, ID ou unidade física.

O limite de 0,75 espaçamento para regiões próximas, o casco convexo e a concordância local são heurísticas de suporte. Uma ilha inteira com IDs coerentemente errados pode satisfazê-las. Uma célula chamada observada não possui erro físico certificado.

M2 usa essas heurísticas corretamente como elegibilidade e mede erro físico contra gold separado. A crítica é que não podemos reinterpretar a elegibilidade, o resíduo de ajuste ou o round-trip como confiança calibrada de implantação.

O controle anterior de duas deformações com os mesmos cantos e bordas também continua pertinente. Para mapas 2D C1 suficientemente gerais, o interior pode mudar enquanto os anchors ficam fixos. Esse é um limite da informação observada, não uma impossibilidade universal de reconstrução de papel físico. Restrições de câmera, superfície, curvatura ou linhas menores podem reduzir a ambiguidade, mas precisam ser explicitadas e testadas.

**Decisão.** Manter checks topológicos e de ida/volta como invariantes necessários. Rejeitar seu uso como proxy de acurácia. Calibrar uma política de aceitação em unidades independentes com erros físicos e estados de ambiguidade registrados. Relatar mapas sem filtro e mapas com elegibilidade em endpoints separados.

### 3.1 Uma célula afim por partes não representa exatamente uma homografia

Mesmo com interseções e IDs perfeitos, a transformação física de uma folha plana vista em perspectiva é projetiva no caso de câmera ideal. Interpolar afim por triângulo concorda nos vértices, mas não reproduz em geral o interior da homografia. Triângulos adjacentes compartilham posições na aresta, o que garante continuidade C0. Suas derivadas podem mudar bruscamente. Uma linha fisicamente reta pode adquirir um pequeno kink na diagonal da célula após o warp.

**Controle executado C04.** Construímos 192 correspondências perfeitas em grid de 5 mm. Os pontos de imagem vêm de `x_image = 10*u/(1+k*u)` e `y_image = 10*v/(1+k*u)`. Consultamos 4.096 posições físicas não usadas no ajuste. Comparamos o `forward_cell_map` de produção com uma homografia estimada das mesmas 192 correspondências por `cv2.findHomography(method=0)`. Os IDs são conhecidos neste controle geometry-only. Nem o solver de homografia nem a malha recebem as queries gold durante o ajuste.

| k por mm | RMSE afim por células | Maior erro nas 4.096 queries | Maior erro da H estimada | Kink de segmento reto na primeira diagonal |
| ---: | ---: | ---: | ---: | ---: |
| 0 | aproximadamente zero | aproximadamente zero | aproximadamente zero | 0° |
| 0,01 | 0,03974 mm | 0,08446 mm | 0,00000174 mm | 1,119° |
| 0,02 | 0,06662 mm | 0,16525 mm | 0,00000197 mm | 2,190° |
| 0,04 | 0,10544 mm | 0,31680 mm | 0,00000153 mm | 4,215° |
| 0,08 | 0,15903 mm | 0,58512 mm | 0,00000295 mm | 7,943° |

No stress `k=0,08`, a query física `[2.5, 2.5]` mm vira `[2.91667, 2.91667]` mm na malha, erro de 0,58926 mm. Não há nenhum erro de detecção ou de ID. O erro vem da família de interpolação. O resultado de precisão numérica da homografia depende deste controle ideal sem ruído e não representa uma expectativa de precisão em fotografias reais. A severidade não foi calibrada para prevalência de capturas clínicas. O kink mede geometria de um segmento de teste, não erro clínico demonstrado de uma onda de ECG.

Esse controle não mostra que uma homografia global é melhor em papel dobrado ou curvo. Nesses casos, o mesmo modelo global pode ter viés físico muito maior. Ele mostra que flexibilidade local não deve ser imposta universalmente e que preservar anchors perfeitos não garante retificação perfeita entre eles.

**Decisão.** Incluir um ramo projetivo global como baseline e candidato para folhas aproximadamente planas. Selecioná-lo por evidência disponível na entrada e regras congeladas em desenvolvimento, nunca olhando o erro gold da imagem de teste. Comparar também um residual local sobre H, com regularização que preserve a solução projetiva quando ela for sustentada. Para curvatura real ou creases, manter os controles em que a alternativa global precisa perder ou abster-se. As referências Dotter e Gridder precisam ser caracterizadas pelas fontes verificadas antes de atribuir a elas interpolação ou suavidade específicas.

Essa recomendação não significa ajustar uma homografia independente a cada célula. Duas homografias locais que compartilham apenas os vértices de uma aresta não precisam concordar na parametrização ao longo de toda a aresta. Em deformações não projetivas ou com ruído, isso pode introduzir seams C0 que a malha afim compartilhada evita. A vantagem de H no controle plano não resolve essa troca.

Uma proposta composicional específica é `F(x)=T(H(x))`. H é uma homografia global estimada. T é uma malha residual compartilhada em coordenadas normalizadas, ligando `H(observed)` a `ids[:,::-1]*5`. A inversa usa exatamente `H^-1(T^-1(u))`. Se as observações forem realmente projetivas, T é identidade e a composição mantém H exata entre anchors. Se houver resíduos, T pode representá-los com continuidade C0. Reinterpolar todos os nós novamente em triângulos afins nas coordenadas originais de imagem perderia essa propriedade. Não corrige, por si, os IDs ou a escala e pode continuar ajustando ruído.

**Protótipo de desenvolvimento CP.** Após registrar `composition-protocol.json`, executamos 12 cenas analíticas e cinco braços, sem rede, banco ou RGB. A composição RANSAC reduz RMSE de 0,15903 mm para precisão numérica no controle plano projetivo forte e de 0,08811 para 0,05452 mm na curva suave. Porém, na crease oblíqua, W piora de 0,29297% para 0,39063%, apesar de RMSE menor. Com ruído alto em folha plana, a H global LS obtém RMSE de 0,02617 mm, contra 0,25607 mm da composição que interpola os anchors ruidosos. Com corrupção coerente de metade da folha, RMSE piora de 4,35208 para 4,39216 mm. A maior diferença C0 medida entre lados de uma aresta é 3,18×10^-14 mm. Um H com horizonte atravessando o casco observado e anchors colineares são rejeitados. Todos os 60 resultados, negativos e arrays permanecem em `composition-results`. Ver `composition-analysis.md` para o escopo. São controles motivadores de desenvolvimento, sem promover uma configuração ou demonstrar desempenho de sistema completo.

## 4. P0: ainda se congela observação possivelmente errada

O default B0 preserva exatamente as âncoras observadas. `fit_complete_map` calcula uma homografia RANSAC, mas a construção da malha ainda recebe todas as âncoras e células. O status de inlier não exclui, por si, uma célula fisicamente equivocada. O fitter permite uma fração global de inliers de 0,35 quando há bastante suporte celular, ou 0,60 caso contrário.

Isso foi útil para corrigir o contrato de preservar a malha local. Não deve se tornar uma lei da próxima arquitetura. Preservar o conteúdo original não exige preservar como verdade física a localização ruidosa de toda detecção.

M2 introduz anchors suaves, mas não permite relabeling, outliers estruturais ou mudança da família de grid. Quando o guard detecta conflito, a ação é conservar a observação. Nos oito pacientes novos, o guard atua em 53 de 64 imagens, apenas dez recebem refinamento e o RMSE comum permanece em aproximadamente 2,16 mm. As falhas absolutas de 14,89 mm e 9,82 mm permanecem iguais à B0.

**Decisão.** Separar erro contínuo de localização de erro discreto de ID e de ponto falso. Comparar anchors duros, suaves e com hipóteses/outliers reestimados. Uma ablação que muda vários componentes ao mesmo tempo não permite saber qual decisão resolveu a cauda.

## 5. M2 é uma proteção parcial, não um modelo geral de dobras

O refinador ajusta um prior projetivo robusto e suaviza diferenças locais dos resíduos com IRLS. Limita deslocamentos, evita inversão de quadriláteros inicialmente válidos e mantém IDs. Essas decisões são conservadoras e razoáveis para ruído de localização.

A proteção detecta coerência de segunda diferença em quatro linhas adjacentes. Um único conflito pode fazer toda a imagem se abster. Não distingue dobra física de erro coerente do detector. Pode bloquear denoising útil em uma região distante do conflito. Dobras fracas e ruidosas podem escapar.

Os controles analíticos M2 mostram a troca com clareza. Sem proteção, uma dobra limpa produz 10% de coordenadas erradas na força 0,15 e 22,67% na força 1. Com proteção, esse dano conhecido é evitado. Em uma dobra com ruído, a proteção perde a melhora possível de RMSE de 0,2798 para 0,1787 mm. Isso invalida a decisão de simplesmente aumentar a força ou remover o guard globalmente.

Sete dos dez ajustes efetivos de teste atingiram o limite de iterações. A flag de convergência se refere ao IRLS antes do cap e do backtracking, como já explicita o relatório. Ela não afirma optimalidade do candidato final. É correto manter essa distinção.

**Decisão.** Conservar guard global como controle conservador. Investigar regularização por região ou por aresta, com descontinuidades de derivada e charts para partes visíveis. Comparar global guard, guard local e sem guard com orçamento igual e negativos fortes mantidos. Não chamar uma máscara heurística de detector certificado de dobras.

## 6. P0: saída RGB e campo de coordenadas são produtos distintos

A confiança do grid não apaga mais o RGB, o que deve permanecer. Contudo, arquivar o original não demonstra que a imagem reamostrada preserva todos os traços finos. A amostragem usa `cv2.INTER_LINEAR`. Sob redução forte, um traço subpixel pode perder contraste ou parecer tracejado. Uma malha bijetiva pode preservar conteúdo no domínio contínuo e ainda perder informação na rasterização.

`render_complete` pode devolver o original quando o canvas excede os limites de lado ou de área. O avaliador M2 mede mapas e consultas sem executar a reamostragem completa de cada caso. Portanto, 42 propostas de malha integral não significam 42 imagens corretamente retificadas e entregáveis. Coordenadas locais úteis sobre um fallback continuam sendo um resultado parcial, não retificação global.

O projeto estende um prior ao frame inteiro, inclusive à mesa ou ao fundo. Isso conserva conteúdo e evita um recorte pelo casco do grid. O fundo não pertence à superfície métrica do papel. O fato de existir uma transformação de apresentação de todo o frame não dá significado físico às suas coordenadas. Regiões desconhecidas da própria folha também podem parecer retificadas apesar de o campo ser inferido.

**Decisão.** Manter o original integral e fornecer proposta, regiões de evidência e status. Medir, separadamente, campo utilizável, RGB realmente reamostrado, fallback original, conteúdo retido e fidelidade rasterizada. Recorte de papel, quando oferecido, deve ser uma vista adicional e reversível. Nenhum benchmark pode converter original preservado em sucesso de retificação só porque a imagem não foi danificada.

## 7. P0: o gauge de avaliação limita a afirmação end-to-end

O avaliador associa detecções B0 a interseções gold dentro de dois pixels e usa as correspondências para escolher permutação de eixos, sinais e origem inteira. A transformação é fixada entre as forças. Não ajusta escala, cisalhamento ou uma transformação afim para salvar o candidato.

Esse desenho é adequado para comparar mudanças de forma em um lattice relativo. Não mede orientação absoluta automática, polaridade dos eixos nem origem sem referência. Uma inversão do tempo ou do eixo vertical pode ser retirada pelo gauge e, ainda assim, ser um erro importante para a digitalização do ECG. Esse alinhamento é avaliação assistida por gold, não vazamento de gold para o solver. A distinção precisa permanecer visível.

**Decisão.** Publicar dois endpoints. O primeiro é geometria relativa após exatamente o mesmo gauge permitido a todos. O segundo é saída orientada e calibrada automaticamente, sem D4 ou correção de origem escolhida pelo gold. Se um concorrente só produz RGB, registrar explicitamente o método adicional de avaliação que extrai suas correspondências. Não comparar métrica absoluta de um método com métrica alinhada de outro.

## 8. P0: a comparação deve separar frontend, geometria e sistema completo

Uma solução Kaggle ou npj pode incluir normalização, detecção de papel, orientação, localização de grid, dewarping e extração de sinal. Esses estágios precisam ser inventariados a partir da implementação original e dos pesos disponíveis.

Há três perguntas diferentes:

1. **Nativo end-to-end.** Com a mesma imagem original e configurações documentadas, qual sistema entrega melhor retificação ou sinal, com suas próprias hipóteses de entrada e falhas?
2. **Geometria com observações comuns.** Dado o mesmo conjunto de correspondências, qual família de transformação funciona melhor? Se IDs ou correspondências vêm de gold, este é um controle oracle de geometria.
3. **Contribuição de componente.** Ao substituir somente o detector, organizador ou solver, quanto muda o resultado no mesmo pipeline?

Alimentar um algoritmo externo com os nossos pontos e chamar a saída de execução do método externo completo confunde essas perguntas. Também é injusto atribuir a falha de um modelo treinado para um crop específico à sua geometria, sem registrar a incompatibilidade de entrada. Deve-se avaliar tanto o domínio documentado quanto a generalização, sem excluir silenciosamente as imagens difíceis.

**Decisão.** O benchmark principal usa pesos e wrappers originais verificáveis, quando disponíveis. Controles com nossos anchors recebem rótulos de componente ou reconstrução. Código fechado, acesso indisponível ou produto sem exportação de campo são estados de disponibilidade, não desempenho zero nem prova de inferioridade.

## 9. P0: o teste M2 já foi consumido para novas decisões

M2 separou corretamente seis pacientes de desenvolvimento de oito pacientes novos. Oito pacientes são oito unidades independentes, não 64 unidades independentes porque existem oito vistas por paciente. As vistas incluem os mesmos geradores e pesos. A auditoria do exportador confirma composição dos próprios campos, sem estabelecer verdade física externa.

Os 64 casos de teste M2 já foram abertos, discutidos, ilustrados e usados para localizar gargalos. Eles são válidos para uma comparação retrospectiva pareada. Se novas escolhas de arquitetura, wrappers ou thresholds forem influenciadas por esses resultados, eles deixam de ser um teste confirmatório intocado da próxima metodologia.

**Decisão.** Congelar adapters e protocolo antes de pontuar comparadores. Usar M2 como estudo retrospectivo de comparabilidade e reservar um novo grupo ou domínio não inspecionado para confirmação. Não apresentar uma mudança de nome do split como uma nova validação independente. Fotografias externas ou outro gerador são necessários para ampliar a conclusão para além da família de renderização atual.

## 10. Estatística e objetivo precisam acompanhar a decisão

G correto, W errado, C retornado e R = W/C são complementares. W baixo sem G mínimo permite vencer abstendo-se de tudo. RMSE condicional pode melhorar pela retirada das piores coordenadas. Erro em suporte comum isola um efeito útil, mas não descreve coordenadas adicionais nem todos os casos sem suporte.

A média M2 dá o mesmo peso a pacientes e aos dois backends, apesar de haver seis imagens torch e duas offline por paciente. Isso é uma escolha explícita de utilidade de engenharia. Não estima automaticamente a prevalência de condições de captura na implantação. O ganho global do teste concentra-se em poucos casos 3D. A regressão de desenvolvimento `01025-6`, com W +2,7344 pontos percentuais no grid, mostra que os gates médios permitem danos individuais.

Um limiar radial de 0,5 mm também não substitui erros separados dos eixos. Na configuração de 25 mm/s e 10 mm/mV, 0,5 mm corresponde a 20 ms no eixo horizontal e 0,05 mV no vertical. São conversões geométricas, não limites clínicos validados. Outros ganhos e velocidades exigem suas próprias conversões.

**Decisão.** Relatar métricas por paciente, cenário, backend, escala projetada e disponibilidade de suporte. Incluir distribuições e casos extremos, não só médias. Publicar erros horizontal e vertical, calibração de tempo/amplitude e impacto no sinal com um extrator fixo. Não permitir alinhamento livre de ganho e tempo no scoring que apague justamente o erro de calibração que queremos medir.

## 11. P1: confiança, robustez e recursos

`fit_complete_map` valida `scores`, mas não os usa para ponderar RANSAC ou a malha. O organizador usa scores para ordem de percurso. M2 usa ranking limitado entre 0,25 e 1, sem alegar que seja probabilidade ou covariância. Isso é honesto. Não justifica chamar o sistema de ajuste de incerteza calibrada.

O casco convexo é recalculado em `evidence`. A classificação materializa produtos entre pontos e arestas do casco. `render_complete` pede evidência e depois elegibilidade, que volta a consultar evidência. Com 48 linhas, largura 4096 e 100 arestas, apenas `delta` e `cross` em float64 podem somar cerca de 472 MB antes dos outros buffers. É um bound estático, não uma medição de pico desta rodada.

**Decisão.** Perfilar e limitar esses produtos, pré-computar o casco e compartilhar classificações. Contar rede, pré-processamento, solver, máscaras, sampling, retries e I/O na latência end-to-end. Um método que só termina porque abstém por orçamento deve ter essa decisão e sua perda de cobertura registradas.

## 12. Política para escolher a melhor decisão metodológica

Não selecionar um único vencedor a partir da média de um único banco. A decisão precisa refletir o domínio de entrada e o produto que queremos entregar.

* Papel aproximadamente plano e evidência consistente: comparar uma transformação global simples contra a não rígida. Complexidade extra precisa demonstrar benefício.
* Papel curvo com grid observável: comparar organização local e ajuste não rígido, com ruído e mudança de derivada tratados separadamente.
* Grid ambíguo ou componentes sem ligação identificável: devolver hipóteses, unidades desconhecidas ou campos locais, mantendo o original.
* Oclusão física: não inventar o trecho escondido e não forçar uma inversa global da folha inteira através de auto-oclusão.
* Candidato fora de orçamento ou sem confiança validada: fallback rastreável, contado como abstinência no endpoint de retificação.

Um router só pode usar características disponíveis na imagem ou no processamento. O backend, o cenário e a severidade fornecidos pelo gerador são gold de avaliação e não podem escolher o método em implantação. Escolher o melhor método por imagem olhando o erro gold produz apenas um limite superior oracle, que deve ser rotulado à parte.

O critério prático é uma fronteira entre G, W, cobertura, erro de sinal, falhas e custo. Mudanças são aceitas pela ablação que resolveu o problema visado, preservou os contratos e passou no conjunto reservado. Não há fundamento para afirmar que toda imagem exigirá sempre a mesma família de transformação.

## Experimentos prioritários, antes da seleção de M3

| ID | Pergunta e braços | Observação que muda a decisão |
| --- | --- | --- |
| E01 | Pontos perfeitos vs rede, mesmo solver | Se a falha persiste com pontos perfeitos, priorizar organização ou geometria |
| E02 | Omissão aleatória vs submalha coerente, mesma quantidade de pontos | IDs errados com baixo resíduo exigem hipóteses de passo, não mais smoothing |
| E03 | Um vs múltiplos componentes, com ligação observável e ambígua em controles distintos | Conservar charts sem forçar origem global não identificada |
| E04 | Perspectiva crescente com homografia exata e depois deformação local | Medir perda do organizador e separar de insuficiência do modelo projetivo |
| E04b | Homografia global, afim por célula e residual sobre H com anchors perfeitos e ruidosos | Medir viés interior e seams, incluindo controles em que uma dobra real torna H insuficiente |
| E05 | Sem guard, global guard, guard localizado, mesmas forças e orçamento | Ganho de ativação só vale se não reintroduzir dano em dobras limpas |
| E06 | Anchors duros, suaves e IDs/outliers reestimados | Redução da cauda deve vir de correção de associação verificável |
| E07 | RGB completo, cabeçalho sem grid, margens, traços finos e regiões opacas | Original arquivado e output rasterizado precisam de verificações distintas |
| E08 | Gauge relativo vs orientação/calibração autônoma | Impedir que simetrias removidas pelo scoring escondam falha de extração |
| E09 | Native end-to-end de cada referência e controles geometry-only | Separar método original, componente adaptado e reconstrução inspirada |
| E10 | Mesmo extrator e sinal de referência, sem ajuste livre de ganho/tempo | Verificar amplitudes, intervalos e segmentos extremos após o warp |
| E11 | M2 retrospectivo e novo domínio reservado | Separar diagnóstico de confirmação, com todos os casos e falhas no denominador |

Todos esses experimentos devem ser registrados antes de observar os scores dos novos comparadores. Os quatro controles estruturais desta revisão já estão concluídos e permanecem identificados como diagnóstico, sem serem transformados retrospectivamente em teste de seleção.

## Fontes de código primárias

Todas as URLs abaixo são permalinks do commit auditado:

* [lattice.py](https://github.com/fmenegui/ecg-torch-renderer/blob/f4f6caef3b59b5830e93284cccadbc6855fc701e/ecg_gpu/digitization/lattice.py)
* [local_cells.py](https://github.com/fmenegui/ecg-torch-renderer/blob/f4f6caef3b59b5830e93284cccadbc6855fc701e/ecg_gpu/digitization/local_cells.py)
* [complete_rectifier.py](https://github.com/fmenegui/ecg-torch-renderer/blob/f4f6caef3b59b5830e93284cccadbc6855fc701e/ecg_gpu/digitization/complete_rectifier.py)
* [frame_mesh.py](https://github.com/fmenegui/ecg-torch-renderer/blob/f4f6caef3b59b5830e93284cccadbc6855fc701e/ecg_gpu/digitization/frame_mesh.py)
* [cell_mesh.py](https://github.com/fmenegui/ecg-torch-renderer/blob/f4f6caef3b59b5830e93284cccadbc6855fc701e/ecg_gpu/digitization/cell_mesh.py)
* [robust_refinement.py](https://github.com/fmenegui/ecg-torch-renderer/blob/f4f6caef3b59b5830e93284cccadbc6855fc701e/ecg_gpu/digitization/robust_refinement.py)
* [rectify_denoising_grid.py](https://github.com/fmenegui/ecg-torch-renderer/blob/f4f6caef3b59b5830e93284cccadbc6855fc701e/examples/rectify_denoising_grid.py)
* [evaluate_nextgen_refinement.py](https://github.com/fmenegui/ecg-torch-renderer/blob/f4f6caef3b59b5830e93284cccadbc6855fc701e/examples/evaluate_nextgen_refinement.py)
* [README M2](https://github.com/fmenegui/ecg-torch-renderer/blob/f4f6caef3b59b5830e93284cccadbc6855fc701e/research/rectification-m2-2026-10-22/README.md)
* [Revisão final M2](https://github.com/fmenegui/ecg-torch-renderer/blob/f4f6caef3b59b5830e93284cccadbc6855fc701e/research/rectification-m2-2026-10-22/final-review.md)
