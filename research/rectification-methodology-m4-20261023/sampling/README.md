# Amostragem M4 e publicação PNG

O componente remove o cache de todos os mapas de subpixels e preserva os valores do candidato M3. Não muda o estimador geométrico nem o padrão de produção. Os 280 casos arquivados, seis variantes e dois tamanhos de bloco produziram **3.360 saídas com igualdade exata em 16.944.480 valores RGB**. A comparação abrange todos os pixels do ROI de avaliação 41×41 do M3, e não constitui uma comparação oculta do quadro inteiro. Identidade, bordas e quadros parciais têm controles separados.

O resultado adversarial decisivo é outro: uma minificação 32:1 faz as três grades fixas 2×2, 4×4 e 8×8 perderem um traço de um pixel fonte. Portanto, a implementação por blocos é uma melhoria de memória, e a metodologia ainda precisa considerar a área e a anisotropia do footprint. Concordância entre essas grades não certifica preservação.

## Interface e convenções

```python
from streaming_sampler import OutputGrid, iter_area_tiles, sample_image
from pngio import write_png_atomic

grid = OutputGrid(
    shape_hw=(768, 1024),
    origin_mm=(2.0, 2.0),       # centro físico do pixel de saída (0,0)
    spacing_mm=(0.25, 0.25),   # dx,dy em mm por pixel
)

# inverse recebe Nx2 coordenadas físicas (x,y) e retorna Nx2 coordenadas
# dos centros dos pixels da imagem fonte original, pixel (row,col)=(col,row).
result = sample_image(
    original_rgb_u8, inverse, grid,
    subgrid=4,
    reconstruction="bilinear",  # ou boxhold
    max_tile_pixels=16384,
    trust=None,                 # desconhecida, não presumida verdadeira
)
receipt = write_png_atomic("rectified.png", result.image)
```

`iter_area_tiles` permite consumir e descartar cada bloco, sem guardar a saída inteira. `sample_image` é a conveniência que coleta imagem e evidências. Ambos mantêm uma cópia float32 da fonte. A memória de imagens fonte, saída, evidências e buffers internos do callback está separada da memória das coordenadas. Para limitar a memória total, o consumidor também precisa descartar os blocos. O escritor PNG atual recebe uma imagem completa e não é um codificador PNG por linhas.

Cada célula de saída é centrada no ponto físico especificado e tem extensão `dx × dy`. A integração usa os subpontos médios fixos, em ordem y depois x, compõe cada um pela inversa diretamente à fonte original e acumula em float32. Arredonda uma única vez. Não há uma imagem retificada intermediária. Os callbacks precisam ser determinísticos por ponto, independentemente do tamanho do lote.

As coordenadas são convertidas para float32 antes do `cv2.remap`, com sua tabela de interpolação. Isto mantém a compatibilidade numérica do M3, sem afirmar uma integração analítica do tent bilinear. O processamento é de códigos RGB, sem linearização sRGB e sem alegação de conservação radiométrica. Fontes e blocos obedecem ao limite de dimensão do OpenCV menor que 32767.

As evidências nunca apagam a imagem:

| Campo | Significado |
| --- | --- |
| `finite_geometry_fraction` | Fração dos subpontos cuja inversa retornou duas coordenadas finitas. Não certifica ausência de dobras, boa calibração ou confiança geométrica. |
| `source_cell_fraction` | Fração dos subpontos dentro de `[-.5,W-.5) × [-.5,H-.5)`, domínio das células fonte. É contagem da quadratura, não uma área exata. |
| `full_reconstruction_support_fraction` | Fração dos subpontos com suporte completo no domínio fonte. Bilinear usa os limites dos centros fonte de maneira conservadora. A tabela do OpenCV pode arredondar um ponto quase na borda para o centro. Boxhold usa os índices arredondados. |
| `trusted_geometry_fraction` | Fração finita aceita pelo callback Booleano `trust`, ou `None` quando desconhecida. Não modifica RGB nem alpha. |

Não se usa um sentinela finito como sinal de validade. Pontos não finitos ou muito distantes recebem coordenadas técnicas seguras somente para a chamada do OpenCV. A máscara explícita força o fundo depois da chamada, impedindo que esses pontos usem acidentalmente o pixel fonte `(0,0)`.

RGB usa fundo constante explícito, branco por padrão, e não renormaliza os subpontos que ficaram dentro da fonte. RGBA usa fundo transparente e integra os códigos RGB premultiplicados por alpha, antes de desmultiplicar uma vez. RGB escondido em pixels transparentes não contamina as cores. Alpha de conteúdo não representa confiança geométrica. RGB final de alpha arredondado zero é zero.

## Custo e memória

Checkpoint final em AMD EPYC 9V74 80-Core Processor, 9 CPUs visíveis, quota cgroup de 8 CPUs, Python 3.12.14, NumPy 2.3.5 e OpenCV 4.11.0. OpenCV, OpenBLAS, OMP e MKL com uma thread cada. Fonte 1400×1100 RGB e saída 1024×768, callback afim conhecido, três repetições em processo novo para cada variante. Inclui conversão da fonte, construção dos mapas, amostragem e coleta. O host era compartilhado, inclusive com outras tarefas do swarm. São tempos de componente CPU, sem exclusividade, estatística de latência ou afirmação sobre GPU.

| Subgrade | Total cache M3, mediana | Total por blocos, mediana | RSS cache M3 | RSS por blocos |
| --- | ---: | ---: | ---: | ---: |
| 2×2 | 0,171 s | 0,287 s | 173,3 MB | 81,5 MB |
| 4×4 | 0,623 s | 1,066 s | 337,3 MB | 81,1 MB |
| 8×8 | 2,185 s | 3,813 s | 941,3 MB | 81,5 MB |

A implementação por blocos foi **mais lenta** nesse teste. Ela também calcula as evidências de suporte, ausentes no candidato antigo, portanto a tabela não isola o custo puro de divisão em blocos. Não é uma alegação de superioridade de throughput. O benefício comprovado é retirar o cache `n² × H × W × 2`.

O cache antigo possui 25,2, 100,7 e 402,7 MB de coordenadas nesse quadro. Com orçamento 16384, as quatro famílias explícitas de coordenadas do componente somam no máximo 917504 bytes, iguais em n=2,4,8. É um contador de arrays, não um limite de RSS. Temporários, OpenCV e callbacks podem alocar mais. A fonte float32 custa 18,48 MB e a coleta RGB com três planos de evidência custa 11,80 MB. Para 16 milhões de pixels e n=8, só o cache antigo seria 8,192 GB. O componente por blocos mantém a dependência no orçamento do bloco, sem fingir que as próprias imagens deixam de ocupar memória.

Os resultados iniciais sob outro momento do host permanecem em `results/benchmark.json`. O checkpoint final está em `results/final-cost/benchmark.json`. Todos os hashes de imagens coincidem entre as duas implementações e entre repetições. Não se escolheu a execução mais rápida para substituir a inicial.

## Achados negativos preservados

Nos 140 casos M3 de faixa de 0,1 mm, bilinear 2×2/4×4/8×8 continuam com zero desaparecimentos, mas o MAE médio passa de 0,001325 para 0,001528 e 0,001604. O erro relativo absoluto médio da energia cai de 1,184% para 0,514% e 0,159%. O erro médio do pico é 0,03169, 0,02263 e 0,02499. Mais subpontos não melhoram monotonicamente cada critério. Boxhold 2×2 continua com dois desaparecimentos. Os números arquivados são reproduzidos integralmente, sem escolher uma configuração por imagem.

Mesmo uma inversa identidade com bilinear mais integração altera uma imagem de alta frequência e mistura fundo nas bordas. O teste escalar independente verifica esse comportamento. Apenas boxhold identidade é exatamente a imagem fonte em todos os pixels, inclusive nas bordas. Nenhuma variante recupera informação perdida na observação original, nem certifica forma de onda, duração QRS ou amplitude em mV.

## Controle positivo orientado pelo ataque 32:1

`axis_aligned_area.py` implementa uma rota diferente e estritamente limitada: integração de área da reconstrução constante por célula fonte, para inversa afim diagonal conhecida. A soma acumulada contínua trata bordas fracionárias, minificação anisotrópica, eixos negativos, RGB/RGBA e fundo fora da fonte. **Qualquer coeficiente fora da diagonal, mesmo pequeno, é rejeitado.** Não usa `cv2.remap INTER_AREA` e não o apresenta como integração de área.

```python
from axis_aligned_area import AxisAlignedAreaSampler

sampler = AxisAlignedAreaSampler(
    original_rgb_u8,
    inverse_matrix=[[32., 0.], [0., 32.]],
    inverse_offset=[32., 32.],
)
image, encoded_mean, source_area_fraction = sampler.sample(
    OutputGrid((1, 1), (0., 0.), (1., 1.)),
)
```

Na imagem branca 65×65 com coluna preta em x=32, o footprint fonte é `[16,48]²`. A média exata dos códigos sob o modelo é `255 × 31/32 = 247,03125`, arredondada para 247. As três grades bilineares fixas retornam 255 e apagam o traço. A concordância 2/4/8 falha como certificado de suficiência.

O protocolo próprio foi congelado depois de conhecer esse ataque e antes de implementar e avaliar o controle positivo. É desenvolvimento orientado por falha, não avaliação externa fresca. A família predefinida contém 60 casos, com larguras 1/2/5 pixels fonte, escalas 2/4/8/16/32 e fases 0/0,25/0,5/0,75. Área exata tem zero desaparecimentos e erro da média zero nesses controles fechados. As grades bilineares 2/4/8 têm 29/17/9 desaparecimentos. Isto compara modelos de reconstrução diferentes em faixas quantizadas, sem converter o resultado numa vitória universal.

Há custo de tabela fonte float64, 37,02 MB no teste 1400×1100. O teste separado de custo dessa rota teve mediana de 0,445 s para construir e coletar saída 1024×768, RSS 156,4 MB, com 27,53 MB de imagem, média float64 e área coletadas. Não é uma comparação de velocidade para a mesma qualidade do tent bilinear. O host também era compartilhado.

O primeiro teste RGBA encontrou resíduo de cancelamento de prefixos em alpha zero, amplificado pela desmultiplicação. O erro foi preservado em `axis-tests-initial.log`. A implementação final mantém `RGB×alpha` e `alpha×255` inteiros durante as somas, usa uma margem numérica para zero e rejeita footprints pequenos demais para sua precisão de prefixos. O guard é de aritmética, não de confiança geométrica ou erro óptico. Os testes de sobreposição escalar são independentes da soma acumulada. Não há garantia numérica universal para todos os tamanhos extremos ou fundos fracionários.

A decisão metodológica que o ataque exige é medir o footprint e o Jacobiano locais, em vez de escolher um n fixo como solução geral. A rota exata retangular é admissível somente no seu domínio diagonal. Para afim geral ou mapas não lineares, a próxima rota precisa de integração de polígonos/células fonte ou filtro de área anisotrópico validado, com critérios explícitos de erro e custo. Dobras, orientação, regiões sem identificação geométrica e extrapolação continuam sendo decisões separadas. O sampler apenas compõe a inversa fornecida e não resolve essas questões.

## Publicação e verificação de PNG

`write_png_atomic` escreve um arquivo temporário único no diretório de destino, sincroniza os bytes, verifica CRC de todos os chunks e IEND terminal, decodifica a imagem completa e exige igualdade de modo, forma e cada pixel antes de publicar.

Por padrão, uma publicação atômica por hard link recusa um destino existente, inclusive se outro processo o criar durante a codificação. A substituição de arquivo exige `expected_existing_sha256` explícito e correspondente, lock cooperativo por destino e nova verificação imediatamente antes de `os.replace`. Um processo externo que ignore o lock ainda pode criar uma corrida na substituição explícita. Não há garantia contra truncamento externo posterior ou falha do armazenamento. O suporte a fsync do diretório é registrado no recibo.

Os testes injetam PNG temporário truncado, falha de decode, corrida de destino, alteração de arquivo antes da substituição e lock ocupado. Em todos esses casos o código deixa de publicar o temporário ou mantém o arquivo que já venceu a corrida. O decoder permissivo `LOAD_TRUNCATED_IMAGES=True` é recusado. O incidente M3 não foi alterado.

## Reprodução e rastreabilidade

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../../swarm-venv/bin/python -m unittest discover -s . -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../../swarm-venv/bin/python evaluate.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../../swarm-venv/bin/python benchmark.py --out results/repeated-cost
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../../swarm-venv/bin/python axis_evaluate.py
```

Executar neste diretório. Os protocolos e seus SHA256 estão em `protocol.json` e `axis_area_protocol.json`. `results/parity.json` contém todos os 3360 registros. A qualidade usa as referências arquivadas, sem regenerá-las pela implementação nova. Os testes finais completos desse diretório estão em `all-tests.log`, 25 testes aprovados.

O primeiro replay e o primeiro benchmark carregaram a implementação cujo SHA256 é `b0331288ce2a660ef0a6e39f1bde5a6db13bf4a6d809f10c620de5b3c33b1448`, preservada em `results/streaming_sampler_at_first_run.py`. Durante o replay foram endurecidas somente a validação de fonte e a exigência de n inteiro. Nenhuma operação da rota numérica usada pelos casos mudou. Os hashes em `parity.json` foram lidos na finalização do relatório e identificam o arquivo no disco nesse momento. O checkpoint final de custo e todos os testes usam a versão final. A distinção entre fonte carregada e arquivo na finalização está explicitada em `receipt.json`.

Nenhum arquivo M3, fixture upstream ou fonte de produção foi modificado. Este é um componente offline NumPy/OpenCV e não entra no caminho quente torch de treino. As comparações com detectores, patentes, Kaggle, npj e validação externa pertencem à integração do swarm.
