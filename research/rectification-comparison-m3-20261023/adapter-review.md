# Revisão do orquestrador M3 antes de executar o painel

Data: 23 de outubro de2026. Revisão de código, sem leitura dos resultados dos concorrentes.

SHA256 do arquivo inspecionado: `18fedaf81856d7533a08aff112873fcc1b5c3c19b8d23fc7b62b046d8374e20e`.

## Parecer

Favorável após as correções dos bloqueadores identificados. Esta leitura não substitui a auditoria dos160 resultados nem certifica a execução do painel.

## Bloqueadores corrigidos

1. O filtro de banding recebia apenas a imagem e era chamado antes da detecção piloto. A implementação agora segue a ordem congelada: detecção original, banding com pontos piloto, segunda detecção somente se o filtro for aplicado.
2. O tempo normalizado da nossa pipeline omitia a organização topológica. Ele agora soma detecção nova, organização e geometria/RGB. A reutilização de CNN entre B0/M2 é declarada, sem vender o wall time curto do segundo braço como latência independente.
3. As cópias de protocolo e schedule utilizadas pela execução não eram verificadas. Agora possuem hashes próprios dentro do lock.
4. Era possível fornecer caminhos de adaptadores diferentes dos arquivos verificados. Os argumentos de execução agora precisam coincidir com os bindings congelados.
5. O lock cobria somente parte dos códigos upstream. Ele agora inclui as fontes Python e os presets, além de commits, pesos e versões relevantes.
6. O resume aceitava qualquer result.json existente. Ele agora confere lock, hash das predições e identidade completa do caso/método.
7. Uma falha na detecção compartilhada poderia permitir nova tentativa apenas no segundo braço. A falha compartilhada agora é memorizada para ambos.
8. Falha de carregamento poderia encerrar a rodada sem linhas de falha por caso. O estado de model-load é registrado e cada caso programado permanece com seu status.

## Contratos preservados

- Inferência recebe a imagem original, não as referências em milímetros.
- Consultas e âncoras são carregadas após o contrato da inferência, usadas somente para consultar os mapas e pontuar.
- A nossa geometria bruta vem dos IDs e convenção fixa row/column. Predições M2 previamente corrigidas por D4 gold não são reutilizadas como raw.
- Falhas geram arrays de queries explicitamente não elegíveis e permanecem no denominador.
- NaN declarado elegível é tratado como erro pelo scorer aprovado, não escondido por nanmean.
- Coordenadas parciais úteis com original-preserved não são uma imagem retificada. O canal RGB operacional do pós-processador exige resampling real.
- A cobertura nativa e a confiança não são intercambiáveis. O npj usa footprint de recorte. O Kaggle usa um inverso geométrico validado, sem confiança nativa. As métricas de campo finito precisam manter essa distinção.

## Limites que devem aparecer no relatório

O timer SIGALRM é cooperativo. Uma chamada nativa pode ultrapassar o orçamento antes de o Python processar a exceção, portanto elapsed e overruns precisam permanecer publicados.

Os samplers nativos e arquivos diagnósticos não fazem a mesma quantidade de I/O. O tempo medido deve ser descrito como a implementação executada, com fontes de timing e compartilhamento explícitos. Não é um teste controlado de throughput de kernels ou hardware.

Um inverso validado pode excluir células dobradas que a imagem nativa efetivamente desenha. Não se pode certificar o RGB completo por W baixo somente nessa parte inversível. O relatório deve mostrar os totais de células inválidas, holes e zeros de interpolação.

O pool tem56 consultas de capturas, mas o agregado principal é somente o split diagnóstico completo de32, cinco métodos,160 resultados. Uma execução prefixada de development serve à feasibility e não se transforma em agregado de24 casos.

O reset de RNG uniforme por caso/método foi recomendado antes do freeze para evitar dependência de estado compartilhado em rotinas estocásticas. Se não fizer parte da fonte congelada final, relatar a limitação de replay sem alterar a fonte após abrir resultados.

## Testes direcionados recomendados

A QA deve manter controles de mapa parcial versus RGB, falhas programadas, argumentos fora do lock, alteração das cópias, resume adulterado e contagem de NaNs. O scorer passou20 controles independentes. O pós-processador passou14 assertivas sintéticas, com160 resultados artificiais e1280 linhas de métricas, sem abrir dados reais de inferência.
