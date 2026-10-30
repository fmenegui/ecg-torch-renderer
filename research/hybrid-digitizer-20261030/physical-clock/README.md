# Integração temporal física e benchmark PM-ECG-ID completo

Avanço de 30/10/2026. Código e controles executados. A comparação em GPU desta
alteração foi bloqueada pelo Runpod antes da inferência, por saldo insuficiente
na conta, com HTTP 402. Nenhum resultado de precisão em GPU foi produzido por
essa tentativa. As medições anteriores continuam preservadas.

As 24 referências usadas na comparação anterior foram conferidas contra o
catálogo completo. Seus sinais e suportes impressos são idênticos, com o
manifesto original e os hashes verificados em `prior-reference-audit.json`.

## O que foi implementado e verificado

- Catálogo de 6.000 variantes de exame e 6.600 páginas, a partir de 100 ECGs
  de 97 pacientes. São 60 condições e 700 variantes de sinal de referência.
- Verificação das 1.200 janelas das derivações contra os sinais originais do
  PTB-XL, incluindo calibração e checksums WFDB. Maior resíduo após comparar
  deslocamento DC constante: 4,04e-7 mV, abaixo do limite de 1e-6 mV. O
  deslocamento usado nessa conferência não altera a referência pontuada.
- Referências próprias para as seis variantes de ruído de sinal. Não pontuar
  uma imagem impressa com ruído contra um ECG limpo diferente.
- Associação explícita das páginas. Há 5.640 entradas de uma página, 240 de
  duas páginas e 120 de quatro páginas. O nome do formato sozinho não fornece
  a duração ou as janelas corretas.
- Exclusão dos 97 pacientes para candidatos novos, incluindo seus outros ECGs
  e todas as renderizações. A ausência de sobreposição dos pesos publicados e
  do treinamento histórico em andamento permanece não certificada.
- Observação das coordenadas antes da reamostragem do npj, com restauração das
  funções originais mesmo quando há exceção. A saída nativa se mantém idêntica.
- Leitura alternativa com escala horizontal em pixel/mm e velocidade física,
  sem esticar o primeiro e o último ponto sobreviventes, sem extrapolar e sem
  preencher lacunas. A calibração vertical e a centralização por linha originais
  foram mantidas para isolar a alteração temporal.

O controle analítico com 200 colunas iniciais ocultas mantém o pico em 4,000 s.
A reamostragem nativa desloca esse pico para 3,750 s. Isso verifica uma propriedade
do componente. Não estabelece ganho em fotografias.

Uma imagem real de desenvolvimento foi executada em CPU a 1.500 pixels, em um
perfil distinto do ensaio GPU. A saída observada foi bit a bit igual ao controle
nativo sem observação. A execução observada levou 31,29 s nesse host. O recibo
preserva parâmetros, hashes e a limitação de não ser uma medição em GPU.

## Limitações preservadas no contrato

A origem x=0 é a origem da imagem alinhada, ainda sem validação independente
como início impresso do sinal. Portanto, a integração física ainda não é um
detector validado da origem temporal. A média de uma linha tampouco valida a
linha isoelétrica ou a preservação do segmento ST. A alternativa recusa
orientações ou associações de ritmo ambíguas.

O catálogo de referências está completo. A aquisição das 6.600 imagens e a
avaliação de todas elas não estão completas. O catálogo inicial tem 24 imagens
adquiridas, e outro manifesto isolado contém as 22 páginas originais preparadas
para este novo ensaio. Esse banco público contém exemplos já vistos no
desenvolvimento e não é um teste independente novo.

O procedimento exato de falha e alinhamento dos autores ainda não foi reproduzido.
O relatório novo exige erro bruto no eixo físico e decomposição de suporte.
Diagnósticos centrados em DC são secundários. Não há alegação de superar
PMcardio, o npj ou vencedores do Kaggle.

## Ensaio Runpod preparado e tentativa bloqueada

O plano congelado tem 12 entradas de exame, seis pacientes, 22 páginas, duas
condições e todos os perfis de uma, duas e quatro páginas. A seleção usa SHA256
e perfis de página, sem escolher por erro. O worker recebe imagens e eixos
públicos. Não recebe sinal de referência, paciente ou formato verdadeiro.

A tentativa [41133083365](https://github.com/fmenegui/ecg-torch-renderer/actions/runs/41133083365)
passou pela admissão e pelos testes antes da alocação. O catálogo vivo indicou
A40 a USD 0,59/h, dentro do teto separado de USD 0,30 de GPU e 1.800 segundos.
O provedor recusou a criação por saldo insuficiente. Não houve ID de pod
retornado nem início de inferência. A limpeza da execução terminou e a chave
SSH temporária foi removida. Esse registro não é um recibo de exclusão de um
pod, pois a criação foi recusada.

O mesmo job de benchmark, ID 124330441527, pode ser reexecutado quando a conta
permitir alocação. Não é necessário alterar a seleção, ampliar o orçamento ou
criar outra geração de plano. Os resultados só poderão ser avaliados depois
do selo de predições, com todos os exames e páginas no denominador.

Uma consulta direta e sem alterações ao Runpod em 30/10/2026, 21:51:56 UTC,
não encontrou um pod com o nome exato do treinamento original. O Actions ainda
marcava a etapa 2 como em andamento, mas seus logs não estavam disponíveis.
Essa combinação não comprova progresso de otimizador. O último checkpoint
recuperado e verificado permanece na época 13, com cópia durável e hash em
`training-status.json`. A consulta não interrompeu nem alterou recursos.

O avaliador verifica hashes, identidade das imagens, taxa de amostragem,
paridade do observador e correspondência da montagem por páginas. Os controles
rejeitam a remoção de um exame, a troca de imagens, alteração de arquivos e
montagem inconsistente. Um caso adversarial demonstra que apagar todas as
predições pode reduzir o NRMSE antigo, mas aparece explicitamente como perda
de 100% do suporte.

## Reprodução e evidências

Instruções executáveis estão em `reproductions/pm_benchmark/README.md`.
`catalog-summary.json`, `cpu-integration.json`, `validation.json`,
`reference-bundle.json` e `runpod-attempt.json` registram os controles e hashes.

O [pacote completo de referências](https://chatgpt.com/library?file_id=libfile_4180411680e48191abe9bba1466f21b9&account_id=personal)
contém 915 arquivos, incluindo sinais originais, bancos fornecidos, catálogo,
metadados, índice de imagens e controle em CPU. Tamanho: 157.552.136 bytes.
SHA256: `b3bf69af5aee92969ae065ce168cd07e5b1d65d69f2963502ad50539352951df`.

O [pacote desta integração e da tentativa Runpod](https://chatgpt.com/library?file_id=libfile_1cd4bfd22f948191b8e4923998ad1871&account_id=personal)
contém as 22 páginas originais, o controle em CPU, fontes da integração, logs
de testes, galeria adversa e recibos da recusa de alocação e da consulta de
treinamento. São 63 arquivos e 71.516.457 bytes, sem predições novas em GPU.
SHA256: `aa1b3ca7642741e75bc24cdb232c639abbafcfda52cfae3b4c794e6abe31f831`.
Os pacotes preservam os snapshots anteriores à inclusão dos seus próprios
links e hashes no relatório.

Validação local: 541 testes integrados, com 25 skips explícitos, e 101 testes
de reprodução, com o código original habilitado. Os 16 cenários adversos foram
exportados e inspecionados. Nenhum peso ou fonte original do npj foi alterado.
