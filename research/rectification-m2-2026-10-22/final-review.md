# Revisão final independente da rodada M2

Data: 23 de outubro de 2026, UTC. Revisor: agente independente de QA. Nenhum módulo, peso, avaliador, pós-processador, seleção ou parâmetro de inferência foi alterado nesta revisão final.

## Decisão

Parecer favorável à publicação dos resultados congelados e ao refinamento como opção experimental. Manter `--refinement-strength 0` como padrão. A rodada demonstra uma melhora limitada de coordenadas no banco sintético novo. Não demonstra superioridade geral, solução da cauda de falhas, fidelidade clínica ou vantagem sobre concorrentes executados nas mesmas condições.

A integração local não recebeu aprovação como execução completa da suíte em um único processo. O processo original terminou sem sumário. A execução independente dos módulos restantes encontrou somente dois arquivos upstream ausentes. Ao fechar este parecer, o integrador informou que os dois workflows de CI, Library contracts e Digitization reference, passaram no commit `0eb6a3f`. Essa é a evidência de integração no checkout completo. A galeria adversa permanece sob responsabilidade do integrador.

## Integridade e seleção

Auditoria reprodutível dos artefatos em `build/nextgen-final-integrity.json`: **8.561 verificações, zero divergências**. Conferi por leitura, sem reinferência:

- Os 14 fontes congelados, checkpoint, protocolo, subset canônico do manifest de desenvolvimento e hash do resultado de validação coincidem com a seleção.
- O recibo final coincide com os hashes do pós-processador, seleção, dois inputs e dois outputs. O recibo declara uma execução de pós-processamento com teste e nenhuma mudança de inferência ou seleção.
- Os hashes dos 112 arquivos de captura coincidem com o manifest.
- Reconstruí os 224 conjuntos de queries dos dois domínios a partir do contrato e dos arquivos originais. Coordenadas, referências e hashes coincidem com os arrays arquivados.
- Recalculei as métricas de 736 combinações caso/método/domínio diretamente das predições arquivadas, incluindo elegibilidade, erro físico, p95/p99 e suporte comum. Todos os valores publicados coincidem.
- Recalculei agregação por paciente e backend, RMSE em suporte comum e os intervalos bootstrap pareados de 20.000 repetições com seed 1973. Não houve divergência.
- Reapliquei os gates de desenvolvimento nas cinco forças, usando somente os 48 casos de desenvolvimento. A seleção continua sendo força 1. O teste contém apenas B0 e esse braço selecionado, sem nova busca de hiperparâmetros.

O desenvolvimento reutiliza seis pacientes previamente inspecionados. O teste contém 64 capturas de oito pacientes novos ao cohort documentado dos pesos congelados, oito vistas por paciente, seis torch e duas offline. Não há interseção de pacientes entre desenvolvimento e teste. A auditoria anterior de seleção de fontes permanece documentada em `build/nextgen-review.md`. Trata-se do mesmo renderer e do mesmo detector antigo congelado. O gauge usa sinais, permutação dos eixos e origem inteira obtidos na B0, fixos entre métodos. Isso não avalia orientação automática absoluta.

## Resultado congelado do teste

G é a fração de queries elegíveis com erro até 0,5 mm. W é a fração elegível acima desse erro, C = G + W e R = W/C. As médias dão o mesmo peso aos pacientes e aos dois backends. As falhas permanecem no denominador das queries fixas.

| Domínio | Métrica | B0 | Refinamento força 1 | Delta em pontos percentuais |
| --- | --- | ---: | ---: | ---: |
| Grid | G | 79,352315% | 79,759216% | +0,406901 |
| Grid | W | 2,857717% | 2,426402% | −0,431315 |
| Grid | C | 82,210032% | 82,185618% | −0,024414 |
| Grid | R | 3,476117% | 2,952344% | −0,523773 |
| Papel visível | G | 70,604706% | 70,925140% | +0,320435 |
| Papel visível | W | 2,992503% | 2,637227% | −0,355275 |
| Papel visível | C | 73,597209% | 73,562368% | −0,034841 |
| Papel visível | R | 4,066055% | 3,585022% | −0,481032 |

O IC descritivo de 95% do delta de G no grid é [+0,056712, +0,974019] pp e o de W é [−1,009115, −0,065867] pp. São intervalos condicionais com apenas oito unidades independentes de paciente e sem correção de multiplicidade. Não fornecem garantia populacional ou clínica.

A cobertura cai ligeiramente. Portanto o ganho de G não é um simples aumento de cobertura. Entretanto a melhora agregada está concentrada no backend offline: delta de G +0,807190 pp e de W −0,849915 pp. No torch-cached, o delta de G é +0,006612 pp. O RMSE no suporte comum permanece praticamente igual, de **2,158140 para 2,156902 mm** no grid e de 2,175994 para 2,174684 mm no papel.

No teste, dez capturas foram efetivamente refinadas, 53 acionaram o guard e uma tinha suporte 2D insuficiente. Sete dos dez ajustes chegaram ao limite de iterações, explicitamente registrado, e três convergiram no critério IRLS anterior ao cap e ao guard de células. O número de propostas de malha completa permanece 42/64 nos dois métodos. Há uma captura sem coordenadas elegíveis. Não houve fallback de construção após tentativa real de refinamento. Os 22 fallbacks registrados são repetições de construção rejeitada quando o refinador já havia conservado a entrada, e não 22 novas falhas do refinamento.

## Cauda e regressões que permanecem

O teste não contém regressões nos limiares diagnósticos graves prefixados. Há quatro casos com ganho de G acima de 1 pp e nenhum com perda acima de 1 pp. Isso descreve deltas, não torna os erros absolutos pequenos.

| Captura e domínio | RMSE | p95 | C | Estado |
| --- | ---: | ---: | ---: | --- |
| 206852-2, offline, grid | 14,889713 mm | 35,201135 mm | 3,491211% | Igual à B0, guard acionado |
| 215864-6, offline, grid | 9,823333 mm | 20,070980 mm | 64,941406% | Igual à B0, guard acionado |
| 211723-6, offline, grid | 4,963844 mm | 5,252676 mm | 25,463867% | Igual à B0, guard acionado |

Na captura 215864-6, R ainda é 48,61%. Preservar a geometria observada pode preservar uma associação errada. O guard não certifica geometria física e tampouco recupera essas falhas. O resíduo interno ou round-trip não substitui esse erro físico independente.

No desenvolvimento, a captura 01025-6 teve aumento de W de **2,734375 pp** no grid e 2,392578 pp no papel. O gate de seleção opera em médias por backend e não proíbe esse tipo de regressão individual. Essa regressão não foi ocultada nem usada para novo ajuste após o teste.

Os controles analíticos completos, incluindo braços sem guard, permanecem em `build/refinement-controls`. A crease limpa sem guard chega a W de 22,67% na força 1, partindo de zero na B0. O guard evita essa regressão conhecida, pode perder um ganho de denoising em crease ruidosa e não captura toda curva fraca. Esses controles não foram apresentados como testes independentes de população.

## Integração e investigação do encerramento local

O log original `build/nextgen-integration.log` termina durante `test_paper.MaterialPaperTests.test_checkpoint_rejects_changed_scene_backend`, com 240 linhas `ok`, sete skips e nenhum ERROR, FAIL, traceback ou sumário. O código de saída 1 foi informado pelo integrador.

O mesmo teste passou isoladamente em 3,146 s com faulthandler. Depois executei todos os 22 módulos de `test_paper.py` até `test_training_adapters.py`, cada um em processo novo com timeout de 180 s e dois threads OMP/MKL/OpenBLAS. Todos produziram sumário e nenhum teve timeout ou novo encerramento abrupto. Foram **190 testes reportados, cinco skips de CUDA e dois erros de arquivos upstream ausentes**. Os outros 21 módulos retornaram zero, incluindo `test_paper.py` completo, o refinador e a integração CLI. Há também um log separado com sete controles CLI aprovados.

Os dois erros de `test_renderer.py` são:

1. `test_compiled_raster_preserves_frozen_masks_and_image_union`: ausência de `tests/fixtures/raster-v021.npz`.
2. `test_raster_matches_frozen_v021_all_30_layouts`: ausência de `references/ecg-gpu-generator-v021.zip`, antes de qualquer comparação.

Não regenerei fixture ou referência. Logs por módulo e sumário estão em `build/nextgen-remaining-tests`. A evidência de módulos isolados não equivale a completar a execução única exigida em `AGENTS.md`, pois pode esconder interação de estado entre módulos.

O cgroup registra limite de memória de 8.589.934.592 bytes, pico de 8.590.315.520 bytes e 1.525.417 eventos `max`, mas zero `oom` e zero `oom_kill`. Isso demonstra pressão de memória durante a sessão, sem provar que ela causou o encerramento. O defeito não foi reproduzido no teste isolado nem no módulo completo. A causa da saída original continua indeterminada. CUDA permaneceu indisponível.

O protocolo usa a imagem RGB original como fonte de amostragem e o CLI conserva detecções originais, candidato e entrada efetiva em arquivos separados. Os controles de integração verificam default zero, guard e fallback por falha de construção. O assessor dos 64 casos mede coordenadas, sem executar uma nova amostragem RGB completa por caso. A revisão final não inspecionou a nova galeria adversa nem reivindica validação externa em fotografias reais.

## Artefatos de auditoria

- `build/nextgen-final-integrity.json`: contagens, hashes conferidos, agregados e CIs independentes, seleção reaplicada, cauda absoluta e estado dos testes locais.
- `build/nextgen-summary/summary.json`, `cases.csv` e `final-postprocessing-receipt.json`: resultados congelados e proveniência do pós-processamento.
- `build/nextgen-paper-checkpoint-isolated.log`, `nextgen-cli-tests.log` e `nextgen-remaining-tests/`: diagnóstico e checks de integração locais.
- `build/nextgen-review.md` e `build/refinement-controls/`: revisão adversarial anterior e resultados analíticos integrais, incluindo regressões.
