# Revisão adversarial da rodada M2

Data: 22 de outubro de 2026. Papel: revisão independente de metodologia e implementação. Estado: módulos e controles analíticos revisados, avaliação nova de pacientes ainda em andamento.

## Critérios de aceitação desta rodada

1. Refinamento usa somente posições detectadas, scores e IDs da organização image-only. Gold e queries de avaliação não entram no ajuste. O score é ranking heurístico, sem alegação de covariância ou probabilidade calibrada.
2. O braço de refinamento recebe exatamente os mesmos pontos/IDs/células da baseline no primeiro contraste. A inferência global/ROI é outro braço, pois muda o conjunto de evidências.
3. A ROI usa RGB e predições. Não pode usar `xy`, `ids`, `valid` ou `reference_mm` do renderer. Transformação de pixels deve conservar centros e voltar às coordenadas do original. Upsampling e recorte não são nova aquisição.
4. Queries físicas, domínio e pesos de paciente/backend ficam fixos entre métodos. G, W, C e W/C acompanham erro em suporte comum e falhas. Não promover ganho de cobertura como ganho de qualidade sem medir o erro acrescentado.
5. Origem inteira, sinais e permutação são gauge de avaliação, sem ajuste de escala ou affine. Para o refinamento com IDs fixos, reutilizar o gauge da B0. Para novas associações, declarar como o gauge é fixado por correspondências independentes das queries avaliadas. Orientação automática continua sendo endpoint separado.
6. Todos os 96 replays antigos são desenvolvimento ou regressão. Ajustar hiperparâmetros neles não cria teste novo. Separar pacientes de seleção, calibração e avaliação nova. Não chamar seed novo do mesmo paciente de paciente independente.
7. Original RGB permanece exato e a malha final continua compartilhada pelo forward e inverse. Proveniência e elegibilidade nunca eliminam RGB. Original fallback é abstinência de retificação completa e coordenadas parciais são avaliadas em trilha distinta.
8. O ajuste deve respeitar limites declarados de deslocamento, iterações e tamanho. Folds, células incompatíveis, limite e timeout têm status explícito. Metadados de `observed_cell_map_changed` precisam refletir que houve refinamento, se esse ramo for composto à saída.
9. Benchmark de performance usa mesmo hardware/entrada/densidade de saída. Medir o custo da visão global + ROI/tiles e do refinamento. Não atribuir apenas o custo do solver à cadeia completa.

## Controles independentes planejados

- Projetiva analítica com ruído e outliers, erro medido em pontos físicos que não são os anchors.
- Grid curvo e crease sem ruído para expor viés de regularização. Melhorar um caso de ruído não implica melhora universal.
- Translação do gauge de IDs, permutação de pontos e input RGB imutável.
- Crop com dimensões ímpares, bordas e coordenadas subpixel, distinguindo x de y.
- Nenhuma alegação de confiança física em função de resíduo interno ou round-trip.

## Correções exigidas pela revisão

- O refinador inicialmente não marcava esgotamento do limite de iterações. Agora separa `converged`, `iteration_limit_reached` e `budget_exhausted`, e declara que a convergência se refere ao iterado IRLS antes do cap e do guard de células.
- A inferência global inicialmente decodificava o canvas inteiro antes de excluir o padding criado pela letterbox. Picos no padding podiam consumir top-k e expulsar um pico válido. O módulo foi corrigido para recortar heat/offsets ao domínio conhecido do resize antes do decoder, sem usar gold.
- Um controle clean de crease mostrou que a suavização irrestrita pode criar erro físico grande em uma geometria válida. Isso motivou guard opcional de conflito com curvatura coerente. O guard abstém do refinamento e conserva as observações. Não certifica que existe uma dobra verdadeira.
- O integrador foi avisado que `observed_cell_map_changed=False` do fitter antigo não descreve corretamente mudanças em relação às detecções originais. O assessor agora registra se as coordenadas foram refinadas e qual foi a observação efetiva usada no fit.
- O assessor deve congelar também os helpers de alinhamento e matching, e o cache deve invalidar se `detect()` mudar. O integrador informou ter acrescentado esses bloqueios antes do reinício e antes de criar qualquer seleção.

## Auditoria dos pacientes

Verifiquei de forma independente `build/recovered/ecg-estudo-resultados/results/cohort/sources.json` contra o hash de `previous_cohort_sha256` do protocolo novo. Os hashes coincidem. São 1.280 grupos anteriores únicos. Os oito grupos novos são únicos, todos do fold 10 e sem interseção com esses 1.280 grupos após normalizar a identidade textual/float do paciente. A interseção com as 77 exclusões explícitas e com as 77 exclusões do source antigo também é zero.

Isso sustenta ausência desses pacientes no cohort dos pesos congelados conforme os manifests auditados. Continua sendo avaliação sintética no mesmo renderer. Não constitui generalização clínica externa.

## Controles executados

Reexecutei os 22 controles B0 de malha completa/células, os dez controles finais do refinador, os nove do detector nativo e dois controles físicos independentes. São 43 testes focados únicos e todos passaram, com OpenCV 4.11 disponível. Os logs finais dos dez testes do refinador e dos dois independentes estão em `build/refinement-controls/guard-tests.log` e `independent-tests.log`. Isso não substitui a suíte completa de integração.

Os dois controles novos em `tests/test_nextgen_independent_controls.py` usam câmera analítica independente do fitter e queries físicas que não são anchors. O segundo verifica invariância de origem inteira, permutação de pontos e imutabilidade dos inputs. Ambos passam, incluindo Ruff E/F.

O estudo reproduzível `examples/refinement_control_study.py` gerou `build/refinement-controls/protocol.json`, `results.json` e dez arquivos de arrays. Foram 10 cenas e 9 braços por cena, B0 mais quatro forças com guard desligado/ligado, totalizando 90 resultados. O protocolo e os hashes dos fontes foram congelados antes da execução. Usou dois threads de OMP/MKL/OpenBLAS. Não houve alteração de parâmetros após observar esses resultados.

| Controle, força 0,15 | RMSE B0 | RMSE sem guard | RMSE com guard | Resultado importante |
| --- | ---: | ---: | ---: | --- |
| Projetiva ruidosa, seed 7919 | 0,52768 mm | 0,27572 mm | 0,27572 mm | G 80,33% para 97,07%, C 100% |
| Projetiva ruidosa, seed 1973 | 0,59661 mm | 0,33510 mm | 0,33510 mm | G 76,53% para 92,93%, C 100% |
| Projetiva ruidosa, seed 104729 | 0,34121 mm | 0,13427 mm | 0,13427 mm | G 90,27% para 99,67%, C 100% |
| Curva clean | 0,03197 mm | 0,04158 mm | 0,03197 mm | Guard conserva B0 e evita viés |
| Crease clean alinhada | aproximadamente zero | 0,24496 mm | aproximadamente zero | Sem guard cria W 10% e p95 0,68214 mm |
| Curva clean rotada | 0,03114 mm | 0,04077 mm | 0,03114 mm | Guard conserva B0 |
| Crease rotada e realocada | aproximadamente zero | 0,10440 mm | aproximadamente zero | Guard conserva B0 |
| Crease fraca | aproximadamente zero | 0,00295 mm | 0,00295 mm | Não detectada pelo guard, pequeno viés permanece |
| Crease com ruído | 0,27978 mm | 0,17866 mm | 0,27978 mm | Guard abstém e perde ganho possível de denoising |
| Crease oblíqua | 0,10763 mm | 0,27440 mm | 0,10763 mm | Sem guard W 1,4% para 11%, guard mantém B0 |

Todos os braços dessa tabela têm C 100% nas queries. Os JSONs também preservam a varredura completa de força. Na crease clean, sem guard, W cresce de 0% em B0 para 5,67%, 10%, 18,93% e 22,67% em forças 0,05, 0,15, 0,5 e 1. Nenhuma dessas regressões foi removida.

O estudo publica métricas tanto no suporte finito B0 fixo como em todas as queries físicas, e o erro no suporte comum. Não há gauge fitting. A classe de ruído/outlier usa ranking de scores controlado, não distribuição empiricamente calibrada de uma CNN. Os casos são controles de desenvolvimento, sem inferência populacional ou alegação de independência entre geometrias relacionadas.

Reprodução dos controles, com diretório novo se qualquer fonte mudar:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python examples/refinement_control_study.py --out build/refinement-controls
```

Verifiquei que `results.json` contém as 90 linhas esperadas, os dez hashes de arrays coincidem e os três hashes de fonte continuam iguais ao protocolo após a execução. A revisão não alterou módulos, runner, pesos, RGB nem parâmetros após os resultados.

## Parecer

Os módulos são aprováveis como braços experimentais opcionais. A B0 deve permanecer o comportamento padrão até a avaliação congelada justificar uma mudança. O guard foi necessário para evitar uma regressão física conhecida, mas também tem custo de oportunidade e não detecta toda curvatura. Pontos/IDs sozinhos não distinguem necessariamente dobra real de erro coerente de detecção.

Ainda não aprovo uma alegação de superioridade geral, calibração de confiança física, fidelidade de waveform ou melhoria clínica. A decisão final sobre o candidato depende das métricas pareadas em pacientes novos, com falhas no denominador, ambos domínios e custos completos. O resultado do estudo analítico é avanço executável de metodologia e diagnóstico, sem substituir esse teste.
