# Desempenho do protótipo CPU

O integrador repetiu o benchmark após tornar os scripts portáveis e verificar os blobs Git de B0. Esta é a rodada principal publicada em `evidence_benchmark.json`. O core não foi modificado.

| Fixture sintética | Consultas com bordas | Baseline | Bounded combinado | Razão de tempo | VmHWM baseline | VmHWM bounded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lattice com lacuna | 25,556 | 262.6 ms | 171.3 ms | 1.53× | 90.64 MiB | 91.96 MiB |
| Stress de hull, sem células observadas | 26,364 | 460.0 ms | 235.4 ms | 1.95× | 306.03 MiB | 133.01 MiB |

## O que foi medido

Comparação de `mapping.evidence(points)` mais `mapping.eligible_geometry(points, mapped=...)` com uma chamada `BoundedEvidence.classify(points, mapped=...)`. As coordenadas forward são calculadas antes, como permite a interface do caller de rendering. Tempos excluem inferência neural, ajuste da geometria, sampling RGB, construção das fixtures e construção do wrapper. O setup do wrapper é registrado à parte.

Hardware: Intel Xeon Platinum 8573C, nove CPUs lógicas visíveis em host virtualizado compartilhado. Um thread por worker, OpenCV/OMP/MKL/OpenBLAS. Python 3.12.14, NumPy 2.3.5, OpenCV 4.11.0. Cada variante usa três processos novos e três chamadas cronometradas por processo. Alternamos a ordem das variantes. O resultado é a mediana das medianas por processo, sem um intervalo de confiança de deployment.

O hull é pré-computado e os arrays de produto cruzado são limitados em ambas as dimensões a no máximo 131.072 pares por bloco. A classificação é compartilhada com a elegibilidade. As operações preservam o dtype e a ordem dos operandos de cada produto da baseline. O wrapper não guarda cache de queries. Precisa ser reconstruído se a geometria observada mudar. Os métodos de conveniência chamados separadamente voltam a repetir classificação, portanto a integração futura deve usar `classify()` uma vez.

## Paridade e domínio das fixtures

Oito configurações combinam duas fixtures, float32/float64 e orçamentos 17/131072. Cada uma verifica queries normais, ordem invertida e vazias. Incluem bordas de células, buracos, vértices/arestas do hull, pontos próximos às bordas e triângulos. Evidência uint8 e elegibilidade booleana coincidem exatamente. Os hashes de saída das consultas grandes também coincidem entre todos os workers e variantes de cada fixture.

`cells_and_hole` é um lattice sintético com células e lacuna. `many_hull_vertices` tem 256 pontos em círculo, IDs arbitrários e `cells=[]`. É stress de alocação, não uma malha física de ECG. O baseline materializa 6.749.184 pares de query/hull nessa fixture, contra blocos de até 131.072. O orçamento não limita a memória dos CellMaps, KDTree ou demais etapas.

## Memória e variabilidade

VmHWM e VmRSS vêm de `/proc/self/status` do processo após exec. São medidas do processo inteiro, incluindo imports, construção da fixture e forward prévio. Registramos ambos antes e depois da seção cronometrada. No stress bounded, o high-water já havia sido atingido no setup/forward. Isso não significa que a classificação aloca zero bytes. No lattice, a pequena diferença de memória entre variantes não demonstra vantagem estável.

A primeira rodada usou `getrusage` e revelou um piso de high-water herdado do processo pai. Ela foi preservada em `evidence_benchmark_initial_rusage.json` como diagnóstico histórico, sem usá-la como medida principal de memória. A rodada corrigida do agente está em `evidence_benchmark_agent.json`. As três rodadas variam aproximadamente de 1,53 a 2,20× no lattice e de 1,95 a 2,95× no stress. Isso impede tratar uma única razão como aceleração estável em produção.

Não são FPS da cadeia completa, ganho de acurácia, medição GPU ou superioridade sobre outro retificador. Antes de integrar, repetir no banco nativo congelado e fazer profiling de rendering completo. A melhoria é promissora como redução isolada de trabalho redundante e alocações de hull.

## Validação desta entrega

Os controles adversariais e NMS foram repetidos pelo integrador. O primeiro usa OpenCV 4.11.0.86, dentro do major suportado pelo projeto. O controle NMS executa o decoder fixado com heatmaps analíticos, sem rede neural. A análise estática compila os novos scripts. A suíte completa e a galeria de produção não foram repetidas nesta investigação de arquivos de pesquisa. O commit de produção B0 mantém seu histórico anterior de CI. Não há inferência clínica a partir desses controles.
