# Retificação ECG: investigação e engenharia

Data da revisão: 22 de outubro de 2026. Baseline B0: `b9fd04dd6516e3469d1f36593849249c082458a8`, PR #19. Seis agentes investigaram anterioridade ECG, documentos deformados, detector, inferência geométrica, benchmark e revisão adversarial. O integrador confrontou os pareceres e repetiu os controles no código de B0, verificando os hashes dos arquivos.

## Decisão

**Priorizar evidência em resolução nativa e identificação correta do grid, seguida de ajuste robusto.** Aumentar a liberdade do interpolador antes disso pode apenas tornar erros de escala e correspondência mais convincentes. O candidato recomendado combina visão global/local, interseções e duas famílias de linhas, hipóteses discretas de escala/IDs, geometria refinada por evidência e seleção calibrada pelo erro físico final.

O original RGB permanece preservado. A transformação final deve compor todas as etapas e amostrar esse original uma vez. A região usada para inferência pode ser recortada, mas a ausência de grid não autoriza apagar conteúdo na saída. Geometria estimada, elegibilidade de coordenadas e validade da amostragem são informações diferentes.

O novo estimador ainda é uma hipótese de pesquisa. Esta entrega inclui diagnóstico, especificação implementável, protocolo de comparação, controles executados e um protótipo isolado de desempenho. Não houve novo treino, execução de redes concorrentes ou benchmark novo em fotografias reais. O core de produção e os pesos permanecem em B0.

## O que ficou demonstrado

| Achado | Evidência executada | Implicação |
| --- | --- | --- |
| Scores não influenciam o ajuste atual | O mesmo fit com scores 1 e 0 produz exatamente os mesmos vértices e triângulos | O parâmetro não representa ponderação geométrica nesta versão |
| Consistência não certifica escala | Injetar associação errada de grid 1 mm como 5 mm gera fit aceito, round-trip zero e erros de 14,14 e 19,80 mm | Validar pitch e identidade com evidência adicional, ou manter ambiguidade |
| Anchors e bordas não determinam o interior | Duas transformações C1 bijetivas coincidem no grid e em suas bordas, mas diferem 1,2 mm no centro | O rótulo de célula observada não implica erro físico pequeno em todo o interior |
| NMS pode suprimir cruzamentos próximos | Dois picos separados por 2 px resultam em um ponto com raio 2 e dois com raio 1 | Medir recall por pitch projetado e validar alternativa adaptativa. Raio menor também pode aumentar falsos positivos |
| Há trabalho CPU repetido e alocações grandes | Protótipo combina evidence/eligibility e divide produto consulta × hull em blocos | Ganho de componente com paridade exata nas fixtures, sem alegação de ganho da cadeia inteira |

O controle de escala viola deliberadamente a premissa major-grid informada ao fitter. Não demonstra que o detector comete esse erro com uma frequência conhecida. O controle C1 vale na classe 2D irrestrita descrita em `adversarial_controls.json`. Ele não estabelece impossibilidade para uma câmera/superfície calibrada com restrições adicionais. Também não é uma alegação de teorema novo.

## Como estamos na qualidade

Os números abaixo vêm do relatório anterior versionado, não de uma nova execução neural. São 48 capturas de teste já inspecionadas, derivadas de seis pacientes. Portanto, fazem parte do desenvolvimento e da regressão.

| Campo de coordenadas | G correto ≤0,5 mm | W errado >0,5 mm | C retornado | W/C agregado |
| --- | ---: | ---: | ---: | ---: |
| Pontos selecionados anteriores | 65,33% | 2,43% | 67,76% | 3,59% |
| Elegibilidade atual | 68,86% | 2,94% | 71,80% | 4,10% |
| Campo estimado sem seleção | 71,43% | 15,62% | 87,05% | 17,95% |

G e W usam o mesmo denominador de consultas fixas no domínio visível. Os valores são balanceados por paciente e backend. A coluna W/C é a razão entre médias agregadas publicadas, não a média do risco por paciente. O limiar de 0,5 mm é critério de engenharia. A cobertura correta cresceu, mas a fração errada entre coordenadas retornadas também cresceu. Expandir a máscara não demonstrou calibração de confiança.

Na verdadeira captura 3D, o campo irrestrito teve 26,30% de consultas erradas. Das 48 imagens, 32 aceitaram uma malha completa e 16 conservaram o RGB original com coordenadas parciais. Estas últimas não contam como 16 imagens globalmente retificadas. A auditoria anterior de cobertura do frame e inversão demonstra consistência de implementação, sem converter esses resultados em acurácia física.

## O que os concorrentes já fazem

| Fonte primária | Mecanismo confirmado | Decisão para nossa engenharia |
| --- | --- | --- |
| [npj, Stenhede et al.](https://www.nature.com/articles/s41746-025-02327-1), [código fixado](https://github.com/Ahus-AIM/Open-ECG-Digitizer/tree/97a15087d4abcda843da8c58ee74b1d8f47e6f9a) | Segmentação, Hough em duas etapas, perspectiva e escala por autocorrelação | Baseline global barato e forte. Avaliar separadamente o TPS experimental, desativado e não validado formalmente no README |
| [Dotter/Gridder, US20240144417A1](https://patents.google.com/patent/US20240144417A1/en) | Grafo de interseções, ciclos/células, inferência de vértices ausentes e amostragem por célula | Interpolação de grid e organização topológica têm anterioridade explícita |
| [US20260065633A1](https://patents.google.com/patent/US20260065633A1/en) | Candidatos de waveform por coluna e associação com continuidade e baselines | Extrator de sinal deve ser distinguido da retificação geométrica |
| [Kaggle 1º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/1st-place-solution) | Subpixel ponderado, surface fitting/outliers, preenchimento do grid e remap direto do original | São controles obrigatórios. Não apresentar esses componentes isolados como novidade |
| [Kaggle 4º](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/4th-place-solution), [código](https://github.com/uchiyama33/physionet_4th_place/tree/d62de09e5ce5e707a71099bdd387e9e335ce4ae9) | Janelas nativas 704², sobreposição, pontos/layout e indexação de linhas | Referência direta para global/local. Classes fixas 55/43 exigem atenção em layouts arbitrários |
| [Kaggle 5º](https://github.com/dangnh0611/kaggle_ecg_digitization/tree/8358d5b09a5b9519275e9b54c11716da5752a9d6) | Correspondências densas, registro iterativo e homografias locais robustas K16 | Baseline geométrico prioritário. Sua ablação mede SNR posterior, não erro físico em mm |
| [UVDoc](https://github.com/tanguymagne/UVDoc/tree/4c9b82b537057aff2526e6dd118a847cdd072e82) e [SyntheticDoc](https://arxiv.org/html/2609.15503v1) | Prior de forma/UV e amostragem do original, com dados físicos diversos | Prior opcional para inicialização. Não fornece automaticamente escala ou aspecto métrico de tiras ECG |
| [D2Dewarp](https://arxiv.org/html/2507.08492v2) | Decoders de linhas H/V e mapa inverso | Reforça usar famílias de linhas como evidência. Essa ideia já é publicada |
| [DvD](https://arxiv.org/html/2505.21975v2) | Diffusion de coordenadas, condicionada por imagem/linhas, seguida de sampling | Ablação opcional. Não confundir com geração de RGB nem assumir vantagem em ECG |

Os textos oficiais completos dos lugares 1º, 2º, 3º, 4º, 5º, 9º e 17º foram recuperados. Recibos e hashes constam em `sources.json`, sem redistribuir seus textos. A verificação de código e documentação não equivale a execução de pesos. Métodos fechados ou checkpoints não recuperados devem aparecer como não executados, nunca como falhas que favorecem nosso ranking.

A referência anterior a “Lauder” aponta para **Leader** no histórico do projeto. O [artigo de Demolder](https://doi.org/10.1016/j.jelectrocard.2025.153900) e seu [preprint](https://www.medrxiv.org/content/10.1101/2024.08.31.24312876v1.full) identificam Leader como segmentador dos traçados, distinto de Dotter/Gridder/Undistortion. O preprint liga a normalização ao pedido 18/280116. A família US20260065633A1 descreve extração tecnicamente correspondente, mas não contém o nome Leader. Esse vínculo nominal não foi estabelecido pela patente.

## Implementação progressiva

| Etapa | Entrega concreta | Decisão que deve resolver |
| --- | --- | --- |
| M0 | B0 congelado, controles adversariais e protótipo CPU isolado | Detectar falsas garantias e remover trabalho redundante sem mudar a geometria |
| M1 | Contrato de labels e aquisição pareada 1024/2048 do mesmo cenário físico | Quanto do erro vem da resolução/observabilidade e quanto vem de labels |
| M2 | Global/local com pontos + linhas H/V, ajuste robusto com IDs fixos no braço diagnóstico | Ganho da observação e do refinamento sem misturar erros de associação |
| M3 | Conjunto limitado de hipóteses de pitch, orientação e IDs, prior denso opcional | Corrigir alias discreto e componentes desconectados sob orçamento de custo |
| M4 | Fase, tratamento de creases e charts visíveis, apenas para falhas remanescentes | Justificar complexidade adicional por ganho externo mensurável |

O detalhamento está em [design-spec.md](design-spec.md). Não adicionar todos os heads e priors de uma vez. O primeiro treino novo depende de exportar IDs, domínio e observabilidade corretamente. A resolução maior deve nascer da aquisição, não de upsampling. GroupNorm torna tiles diferentes do frame inteiro mesmo com halo, exigindo treino ou normalização compatível.

## Execução reproduzível

Use Python 3.12 com NumPy 2.3.5, SciPy 1.17.0, Shapely 2.1.2 e OpenCV headless 4.11.0.86. O controle NMS usa Torch 2.8.0 CPU. Os scripts verificam os blobs Git de B0 e falham se o código auditado mudou. Num checkout completo deste commit de pesquisa, os paths padrão resolvem os módulos herdados de B0.

```bash
python research/rectification-nextgen-2026-10-22/adversarial_controls.py
python research/rectification-nextgen-2026-10-22/nms_resolution_control.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python research/rectification-nextgen-2026-10-22/evidence_benchmark.py
```

Para um snapshot separado, use `--geometry-dir /caminho/aos/modulos` e `--decoder /caminho/grid_point_learning.py`. O microbenchmark requer Linux para `/proc/self/status`. Os resultados publicados foram repetidos pelo integrador com esses parâmetros explícitos, pois seu workspace contém um snapshot verificado dos módulos, não um checkout completo.

O protótipo `BoundedEvidence` pré-computa o hull, limita os arrays de produtos cruzados a 131.072 pares por bloco e oferece `classify()` para compartilhar a classificação. É preciso reconstruí-lo se a geometria observada mudar. O orçamento não limita a memória dos CellMaps ou KDTree. Ver [performance.md](performance.md) para medições, hashes e limites.

## O que constitui uma vitória

O protocolo em `benchmark-protocol.json` separa coordenadas parciais, imagem retificada, geometria e sinal. Exige pacientes independentes, domínio comum, falhas no denominador, qualidade em subgrupos e latência de toda a cadeia. Compara extrator congelado comum e pipelines nativos em trilhas diferentes.

A contribuição potencial é demonstrar calibração métrica e associação periódica robusta, com incerteza útil e custo controlado em layouts/capturas diversos. Surface fitting, tiles, UV, subpixel, H/V e uma única reamostragem já existem. A meta é superar comparadores fortes nomeados em um benchmark versionado. Ainda não medimos essa superioridade.
