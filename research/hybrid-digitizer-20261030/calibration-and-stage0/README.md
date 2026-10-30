# Calibração explícita, orientação de imagem e diagnóstico do Kaggle

Etapa de 30/10/2026. O código avança a leitura física e a preparação de dados.
Não há treinamento novo nem resultado novo de precisão em GPU nesta etapa.
O Runpod voltou a recusar a criação por saldo insuficiente, HTTP 402.

## Entrada de imagem com geometria rastreável

Os JPEGs de scan dos dois casos de desenvolvimento usados no diagnóstico
Kaggle possuem EXIF Orientation 6 e 8. A conversão anterior
`Image.open(...).convert('RGB').save(...)` preservava a matriz armazenada,
descartando a orientação de apresentação ao criar o PNG. Os PNGs do diagnóstico
original foram reconstruídos e são byte a byte idênticos aos da execução GPU
anterior. As medições históricas e seus arquivos permanecem intactos.

`ecg_gpu.digitization.image_io.decode_ecg_rgb` aplica a orientação EXIF uma vez
e registra as matrizes inversas entre centros de pixels armazenados e pixels
orientados. Não interpola o traçado nem altera o arquivo original. As oito
orientações, inclusive reflexões, têm controles de correspondência de pixels.
O novo `NativeNpjBackend.predict_file` utiliza essa entrada e preserva o hash
do arquivo original. A API existente que recebe RGB conserva seu comportamento.

O comportamento de `ImageOps.exif_transpose` e a remoção da tag aplicada são
documentados pelo [Pillow](https://pillow.readthedocs.io/en/stable/_modules/PIL/ImageOps.html).
O quadro de coordenadas da imagem orientada é diferente do quadro armazenado.
Baselines, origem temporal e pontos de grade precisam acompanhar essa mudança.

## Calibração a partir de evidências

`ecg_gpu.digitization.calibration` implementa:

- Leitura de velocidade e ganho a partir de texto com unidades, incluindo
  decimal com vírgula e ganho inverso em mV/cm. Valores ausentes, negativos,
  alternativos ou conflitantes não recebem defaults.
- Medição de um pulso retangular em uma ROI binária previamente identificada,
  com amplitude e duração fornecidas de evidência independente. Pulsos
  desconectados, cortados ou incompatíveis são recusados. A ROI e os rótulos
  do pulso ainda precisam de um detector ou anotação externa.
- Baseline obtida do pé do pulso ou de uma linha isoelétrica fornecida. A média
  do traçado não é aceita como evidência de baseline nessa nova API.
- Associação de origem temporal a um marcador fornecido, com tempo e pixel.
  O fim de um pulso ou o primeiro ponto visível não estabelece o início do ECG.
- Conferência da identidade da imagem e do quadro de coordenadas. Um parâmetro
  espacial de outra imagem ou de outro quadro produz erro explícito.
- Reamostragem física mantendo lacunas, sem extrapolação ou ajuste do eixo à
  extensão visível. Evidências incompletas produzem estado não resolvido.

O controle analítico usa um pulso medido, duas escalas diferentes por eixo e
um marcador temporal independente. Com um segundo inicial oculto, o pico fica
em 4,000 s e o deslocamento de amplitude de 0,2 mV é mantido. Isso verifica o
componente, não valida preservação clínica de ST em fotografias.

Foram executados OCR local e análise das 22 páginas do ensaio congelado. O
perfil com EXIF, cabeçalho/rodapé e tamanho limitado a 2.400 pixels retornou
três pares candidatos de velocidade e ganho. Os demais permanecem ausentes
ou ambíguos. Nenhuma das 22 páginas foi declarada completamente calibrada,
pois faltam grade, baseline e marcador temporal independentes. O OCR não foi
tratado como verdade clínica. Seus textos, palavras, caixas e hashes estão
preservados. Corrigiu-se também a interpretação de aspas literais no TSV.

A execução de OCR em páginas inteiras orientadas excedeu o tempo disponível
em uma imagem e não completou um recibo de lote. Seus arquivos parciais estão
preservados separadamente. O perfil limitado registra timeouts por imagem,
mantém todos os agendados e não aceita texto parcial como calibração.

## Diagnóstico da etapa zero original

Os quatro assets da etapa zero são conferidos por SHA256. O modelo e os
processadores originais são executados sem alterações nos seus fontes ou pesos.
O controle usa CPU e float32. As quatro TTA são mantidas, executadas uma de cada
vez para limitar memória. A tentativa com lote quatro excedeu a memória do host
de 8 GiB e foi preservada. O controle pequeno de lote quatro versus execução
serial difere no máximo 1,20e-7 nas probabilidades. Não se declara equivalência
bit a bit com a inferência GPU float16.

| Mesmas entradas da execução GPU anterior | Pontos nativos diferentes de zero | Resultado em CPU |
| --- | ---: | --- |
| pm-46 scans | 2 de 9 | Homografia ausente, erro no warp |
| pm-46 Samsung | 0 de 9 | Homografia ausente, erro no warp |
| pm-46 dobrado | 0 de 9 | Homografia ausente, erro no warp |
| pm-80 scans | 1 de 9 | Homografia ausente, erro no warp |
| pm-80 Samsung | 0 de 9 | Homografia ausente, erro no warp |
| pm-80 dobrado | 0 de 9 | Homografia ausente, erro no warp |

Os pontos ausentes recebem [0,0] no processador original. Nesses casos existem
menos posições distintas que as necessárias para determinar a homografia.
O controle com nove pontos independentes passa com nove correspondências e
erro máximo de reprojeção de 0,215 pixel. Portanto, o processamento geométrico
funciona quando recebe referências suficientes. A disponibilidade dos pontos
nesse domínio é a limitação observada, e não foi reparada usando o sinal de
referência ou pontos conhecidos do benchmark.

Os dois scans também foram repetidos após aplicar EXIF. Ambos ficaram sem
pontos válidos e continuaram falhando. Corrigir EXIF não resolveu esse modelo
de localização. Esses resultados são diagnósticos em CPU, sem geração de
waveforms e sem comparação com o score do Kaggle.

## Fonte de dados para o próximo leitor

`reproductions.pm_benchmark.candidate_split` congela a partição por paciente,
com semente explícita e hashes dos metadados e das exclusões. Todos os registros
e todas as futuras renderizações de um paciente herdam a mesma partição.
São removidos 152 registros dos 97 pacientes do benchmark, incluindo registros
adicionais que não são os 100 ECGs-base do PM-ECG-ID.

| Partição de fontes de sinal | Registros | Pacientes |
| --- | ---: | ---: |
| Treino | 17.291 | 14.986 |
| Validação | 2.174 | 1.890 |
| Reserva de fontes | 2.182 | 1.896 |

O manifesto prepara identidades de sinal. A aquisição de todos esses sinais,
o treinamento de um novo leitor e um teste clínico pareado independente
continuam pendentes. A reserva de fontes ou suas imagens sintéticas não é
apresentada como validação clínica independente.

## Reprodução

```bash
python -m reproductions.kaggle.stage0_diagnose \
  --assets /data/kaggle-assets --inputs /data/image-only-manifest.json \
  --root /data/png-inputs --out /data/stage0 --device cpu --tta-chunk-size 1

python -m examples.inspect_ecg_print_settings \
  --inputs /data/image-only-manifest.json --root /data/images \
  --out /data/print-settings --border-view

python -m reproductions.pm_benchmark.candidate_split \
  --metadata /data/ptbxl_database.csv \
  --exclusions reproductions/pm_benchmark/training-exclusions.json \
  --out /data/candidate-source-split.json
```

O OCR requer Tesseract com o modelo eng. A execução usada foi 5.3.4, com hash
do modelo e perfil registrados. O diagnóstico neural requer o ambiente Kaggle
separado, com timm 1.0.22. O dispositivo e a precisão são explícitos.

Validação local: 541 testes integrados, com 25 skips explícitos, e 120 testes
de reprodução com fontes npj e assets Kaggle habilitados, sem skips.
Os 16 cenários adversos foram exportados e inspecionados. A consulta ao treino
foi corrigida para incluir pods parados e conferir os metadados do volume
exato, sem iniciar, parar ou remover recursos.
