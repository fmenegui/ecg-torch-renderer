# Retificação M4: decisões após o confronto metodológico

23 de outubro de 2026. Revisão por seis agentes e integrador. Componentes de pesquisa, sem promoção do método completo a produção.

O método candidato deve combinar observações de pontos e linhas, hipóteses explícitas de unidade e identidade, seleção da complexidade geométrica e amostragem do RGB original. As evidências novas favorecem essa direção, mas refutam tanto um residual universal quanto uma quadratura fixa universal. O objetivo mensurável é melhorar cobertura fisicamente correta com erro, perda de conteúdo e custo controlados.

## Arquitetura proposta

1. Preservar a captura original, seus metadados e um digest. Normalização de aparência pode servir ao detector. O raster entregue pela retificação deve ser amostrado do original.
2. Detectar folhas/charts, texto e layout, duas famílias de linhas, interseções e marcas de calibração em múltiplas escalas. Refinar posições em resolução original. Separar incerteza de localização, associação e unidade.
3. Construir um grafo com saltos inteiros, classes major/minor e hipóteses de pitch. Manter D4 e fase por componente enquanto faltarem pistas que as distingam. Ciclos e reciprocidade ajudam a detectar contradições, mas não verificam a escala.
4. Comparar explicitamente H global, residual compartilhado e regularizado no domínio normalizado por H e H local K16 nas mesmas observações e orçamento. Usar validação espacial de observações sem consultar a referência física de avaliação. A família e a regra são congeladas antes dos scores. O componente M4 executado ainda não inclui K16, portanto não estabelece sua posição frente ao residual.
5. Reservar um modelo com creases e charts de visibilidade para dobras sustentadas por evidência. Não impor suavidade de derivadas através de uma dobra real. A versão M4 executada ainda não implementa esse braço.
6. Exigir domínio explícito, continuidade, ausência de horizonte, degenerescência, sobreposição e contatos de fronteiras não incidentes, com inversa consistente. Essas verificações são numéricas e condicionais ao modelo, não certificados físicos.
7. Gerar o RGB em uma passagem pelo mapa inverso composto. Não converter confiança de grid em alpha. Considerar a área e anisotropia do footprint para redução. A quadratura por blocos é implementada e verificada, mas pode sofrer alias mesmo em 8×8. Há um controle separado de área exata para afim com eixos alinhados.
8. Devolver mapa, hipóteses, proveniência e suporte separados da imagem. Original preservado, coordenadas parciais, retificação relativa e retificação métrica condicionada à evidência são estados diferentes. Verificar pixels decodificados antes da publicação atômica do PNG.

## Escolhas entre famílias

| Decisão | Evidência e contraponto | Escolha atual |
|---|---|---|
| H global ou flexibilidade local | H é exata para plano sob projeção compatível e pode reduzir ruído com IDs/modelo corretos. Curvatura, distorção de lente e erro coerente são contrapontos. | H sempre candidata, sem forçar residual em toda entrada. |
| Triângulos diretamente na imagem | C0 e custo simples, com viés entre âncoras sob perspectiva forte. | Baseline e fallback explícito, não solução universal. |
| Homografia independente por célula | Representa planos locais. Cantos compartilhados não garantem C0 ao longo da borda. | Comparador, sujeito a seam/inversa. Não adotar só pelo nome “perspectiva”. |
| H local K16 do 5º Kaggle | Autor publica ganho de SNR downstream em seu fold. Código seleciona uma H por célula sem blending padrão. | Candidato autêntico a reproduzir. Não converter o SNR do autor em erro geométrico nosso. |
| Bilinear, spline ou TPS | Mais suave pode reduzir distorção em curva, mas suavidade não prova unidade nem injetividade. Creases pedem outra regularidade. | Ablacionar com as mesmas observações, guardas e orçamento. TPS experimental npj é braço separado. |
| H seguida de residual | Preserva a perspectiva e usa continuidade compartilhada. M4 reduz vários erros, mas piora a dobra oblíqua e falhas coerentes. | Candidato de desenvolvimento com seleção observacional. Não promover como vencedor universal. |
| Decoder do grid | Vizinho detectado nem sempre é uma célula de 5 mm. Linhas ausentes podem manter ciclos perfeitos e pitch errado. | Prioridade P0: saltos inteiros, major/minor, hipóteses de pitch e D4, evidência independente de calibração. Ainda não implementado. |
| Template semântico | Ajuda a indexar e orientar no layout conhecido. Template errado pode criar correspondências coerentes erradas. | Reconhecer suporte do layout antes de usar. Não deformar formato desconhecido para caber em 3×4+r1. |
| Regiões sem grid | Podem conter waveform, texto, pulsos e cabeçalhos. Furo pode esconder geometrias diferentes. | Preservar RGB e distinguir interpolação, extrapolação e ambiguidade. Não afirmar coordenadas observadas onde só há prior. |
| Várias ilhas visíveis | O maior componente perde evidência. A fase entre ilhas pode ser indeterminada sob oclusão. | Guardar todas as ilhas e hipóteses locais. Vincular somente com evidência ou hipótese declarada. |
| Amostragem pontual ou de área | Pontual apaga traços em fases específicas. Área fixa melhora controles M3, mas também falha sob minificação forte. | Implementação por blocos para memória. Filtro/integração orientados pelo footprint são a próxima etapa, sem certificado por concordância 2/4/8. |
| Fluxo neural ou modelo 3D | UVDoc, D2Dewarp e outros oferecem priors e pistas úteis. Resultados de OCR não medem mm de ECG. | Comparadores futuros após pin de código/pesos e avaliação externa. Não acrescentar redes apenas por atualidade. |
| Métrica única ou múltiplos gates | RMSE condicional pode melhorar ao rejeitar tudo, e médias escondem spikes e texto. | G/W/C/R fixos, caudas, objetos pequenos, waveform, arquivo e recursos separados. |
| Declaração de estado da arte | M3 consumiu o conjunto sintético, nosso detector conhece a família de renderer e o estrato Kaggle compatível tem uma captura. | A metodologia pode aspirar a SOTA. Superioridade completa permanece uma hipótese até teste externo congelado. |

## Critérios de adoção e prioridades

P0 é corrigir o contrato de evidência e o decoder, preservar o original e todo conteúdo observado, manter guardas de malha e saída íntegra. Há implementações verificadas de sampling/export e seleção geométrica condicional, mas não um detector novo completo.

P1 é testar o decoder conjunto, observações de linhas e pontos e modelos de crease com dados de desenvolvimento separados. O solver esparso deve preservar a mesma equação e os mesmos resultados do denso, enquanto guards e consultas também precisam escalar. Acelerar uma solução linear não equivale a acelerar todo o retificador.

P2 é executar os comparadores autênticos com um holdout fresco compatível e outro amplo, seguido de fotografias físicas com referência independente. Os 24 writeups públicos retornados pelo fórum Kaggle foram lidos, mas somente o terceiro lugar teve execução nativa na etapa M3. Powerful Medical continua sem execução nesta tarefa. Uma reconstrução inspirada na patente não substitui o produto.

O protocolo externo em `external-benchmark-protocol.json` separa estimador condicional, geometria nativa e preservação de imagem/sinal. Ele inclui partição por paciente/documento/fonte, queries uniformes no material e no pixel visível, objetos pequenos, treinamento sobreposto, falhas no denominador e custo comparável.

## Limite de novidade

Pontos de grid, grafos, ciclos, templates, inferência de ausências, H com deformação local e amostragem direta do original têm precedentes. O 15º Kaggle também documenta correções de canvas e convenção de pixels no Heng. Não reivindicamos essas ideias como novas. A contribuição candidata é uma combinação auditável que demonstre melhor fronteira de cobertura, erro e preservação em fontes independentes.
