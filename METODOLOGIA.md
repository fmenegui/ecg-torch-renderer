# Gerador de ECG em PyTorch v2, pacote 0.3.0

O pacote independente `ecg-torch-renderer` oferece `ECGImageLayer`, um `nn.Module` que transforma sinais `[B,12,T]` em imagens RGB `[B,3,H,W]` e entra diretamente antes de uma rede de imagem. `ECGRenderer` acrescenta referências e alvos alinhados quando necessários. A v2 inclui 30 layouts, papel com relevo, iluminação por normais, materiais de mesa e mapas externos de documentos 3D. O código foi executado em CPU. CUDA, correspondência com fotos clínicas e superioridade sobre outros geradores ainda não foram demonstradas.

## Biblioteca independente e nn.Sequential

Extraia `ecg-gpu-generator-v030.zip` e instale com `python -m pip install -e .`. O pacote tem seu próprio `pyproject.toml`, licença MIT, testes, exemplos e CLI. Não depende do site, do treinador de ECG Replot, de uma arquitetura de rede específica ou de pesos pré-treinados. No pacote, use o namespace `ecg_gpu`. Os caminhos `training` nos exemplos do projeto apontam para as mesmas fontes.

```python
import torch
from torch import nn
from ecg_gpu.gpu_module import ECGImageLayer
from ecg_gpu.gpu_renderer import RenderConfig
from ecg_gpu.gpu_parameters import Uniform, Choice, Bernoulli

images = ECGImageLayer(
    fs=500,
    config=RenderConfig(height=384, width=512, profile='mixed'),
    parameters={
        'layout': Choice(['3x4+r3', '6x2', '12x1']),
        'angle': Uniform(-.15, .15),
        'header_present': Bernoulli(.9),
        'gain_index': 1,
    },
    seed=1973,
)
cnn = nn.Sequential(nn.Conv2d(3, 16, 5, stride=4), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 2))
pipeline = nn.Sequential(images, cnn).cuda()

logits = pipeline(signals_cuda)  # [B,12,T], mV
loss = nn.functional.cross_entropy(logits, classes_cuda)
loss.backward()
```

A camada não tem pesos aprendidos. A renderização é uma transformação de dados sem gradiente para o sinal original. O `backward` treina normalmente a CNN subsequente. `stage='image'` evita alocar/reter máscaras de traçado, baseline, labels e vértices que uma rede de classificação não precisa. O `forward` retorna apenas o tensor RGB em `[0,1]`. Normalização adicional pode entrar depois dele conforme a rede escolhida.

Desde a versão 0.2.1, o renderizador mantém float32 mesmo dentro de `torch.autocast`. A CNN recebe normalmente a precisão mista do contexto externo. Mova o módulo com `.to(device)` ou `.cuda()`. `.half()` ou `.bfloat16()` no renderizador não são suportados. Para treinar a CNN em CUDA float16, use `torch.amp.GradScaler`. A taxa padrão fica em buffer e o RNG por dispositivo é reutilizado, preservando a sequência e o contrato de checkpoint.

`PERFORMANCE_GPU_V030.md` apresenta a comparação pareada em CPU. `compile_raster=True` funde curvas e calibração, com primeira chamada de compilação medida separadamente. `ecg-renderer-profile` exporta operações, kernels, alocações, trace e ZIP. O notebook `notebooks/colab_gpu_profile.ipynb` executa o ensaio completo em um runtime GPU do Colab. CUDA ainda não foi medido localmente. A disponibilidade de CUDA não demonstra, por si só, vazão suficiente para uma determinada CNN.

Parâmetros ausentes seguem os sorteios do perfil/severidade. Valores constantes ficam em buffers. `Uniform(low, high)` controla intervalos contínuos, `Choice(values)` sorteia constantes ou nomes e `Bernoulli(p)` controla presença/ausência. As distribuições também ficam em buffers e sorteiam na GPU. Para RGB ou slots por derivação, os limites/probabilidades podem ser vetores compatíveis. `Choice` seleciona um valor ou vetor completo por imagem. Use `images.renderer.sample_parameters(B)` para inspecionar todas as chaves, formas e tipos disponíveis.

Em treino, uma seed definida muda a captura a cada chamada. `.eval()` repete a mesma seed e `randomize=False` fixa a captura também em treino. `state_dict` guarda buffers, configuração compatível e contador para reproduzir o próximo lote após retomada. Reproduzir pixels exige também o mesmo sinal, tamanho de lote, plataforma e versão de PyTorch. Em DDP, use `seed=base_seed+rank`. Não há promessa de pixels idênticos CPU/CUDA. Com `seed=None`, o treino usa o RNG global do PyTorch, que o chamador deve restaurar no checkpoint.

Para taxas e comprimentos variáveis ou parâmetros próprios por amostra, chame `images(signals, fs=rates, lengths=lengths, parameters=overrides)`. Os overrides aceitam tensores no dispositivo e têm prioridade sobre a configuração da camada. `images.render_batch(..., stage='all')` fornece os alvos pareados. `nn.Sequential` usa a taxa padrão definida no construtor.

## Converter sinais pela CLI

```bash
python -m ecg_gpu.gpu_cli --input sinal.npz --output ecg.png \
  --device cuda --layout 3x4+r3 --metadata ecg.json

python -m ecg_gpu.gpu_cli --input sinal.npy --fs 500 --units mV \
  --output ecg.png --device cpu

python -m ecg_gpu.gpu_cli --input lote.npz --output imagens --device cuda \
  --parameters '{"layout":{"choice":["6x2","12x1"]},"angle":{"uniform":[-0.15,0.15]}}'
```

No pacote instalado, o comando direto é `ecg-render`. Entrada NPY tem forma `[12,T]` ou `[B,12,T]`, ordem padrão e `--fs` obrigatório. NPZ contém `signals` e `fs`, com `lead_names`, `units` e `lengths` opcionais. Taxas NPZ podem ser escalares ou `[B]`. A CLI reordena os nomes padrão e converte mV/uV/V para mV. Sem unidade declarada, assume mV. Um lote requer pasta de saída. Uma única amostra também aceita arquivo `.png`, `.jpg` ou `.jpeg`.

Com o extra `wfdb`, a entrada `.hea` lê o registro e as unidades de cada canal. `--units` aplica-se somente a NPY/NPZ. A verificação WFDB usa um leitor simulado, sem acervo WFDB real. `--font`, `--document-assets`, dimensões, perfil, severidade, layout, ganho, velocidade, relevo e sombras são configuráveis. O JSON opcional registra hashes, seed, escala e parâmetros realizados. PNG/JPEG e JSON são codificados na CPU depois da geração, fora do laço de treino.

## Usar durante o treinamento

Instale Python 3.10+, PyTorch 2.6+ com a distribuição CUDA apropriada, NumPy e Pillow 10.1+. As verificações locais usam PyTorch 2.14.1+cpu. O pacote instala NumPy e Pillow. Instale a distribuição apropriada do PyTorch antes dele.

```bash
python -m pip install -e .
```

```python
import torch
from ecg_gpu.gpu_renderer import ECGRenderer, RenderConfig
from ecg_gpu.models_v05 import build_model, task_loss

device = torch.device('cuda')
renderer = ECGRenderer(RenderConfig(height=768, width=1024)).to(device)
model = build_model('geometry').to(device)

# sinais_mv: [B,12,T], em mV, já carregados
signals = sinais_mv.to(device, non_blocking=True)
rng = torch.Generator(device=device).manual_seed(1973)
batch = renderer(signals, fs=500, stage='geometry', generator=rng)
loss = task_loss('geometry', model(batch.image), batch.targets)
loss.backward()
```

A ordem é `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`. Para taxas ou comprimentos diferentes, passe `fs` e `lengths` como tensores `[B]`. Preencha as séries até um comprimento comum. São impressos até dez segundos por padrão, configuráveis por `max_duration_s`. Os sinais fisiológicos não são modificados pelo gerador.

Fontes são rasterizadas uma vez na CPU. O atlas e os bancos de texturas/mapas acompanham o módulo à GPU. A renderização por lote usa PyTorch, sem PIL, SciPy, NumPy, PNGs ou cópias `.cpu()`/`.numpy()`/`.item()`. Leitura de NPZ, inicialização de fontes e exportação de exemplos continuam sendo tarefas de armazenamento na CPU.

## Layouts e faixas longas

| Família | Variantes |
|---|---|
| 3×4 | Sem ritmo e com 1, 2 ou 3 faixas longas |
| 6×2 | Sem ritmo e com 1, 2 ou 3 faixas longas |
| 12×1 | Sem ritmo e com 1, 2 ou 3 faixas longas |
| 2×6 | Sem ritmo e com 1, 2 ou 3 faixas longas |
| 4×3 | Sem ritmo e com 1, 2 ou 3 faixas longas |
| Cabrera 3×4 | Sem ritmo e com 1, 2 ou 3 faixas longas |
| Cabrera 6×2 e 12×1 | Ordem de Cabrera, incluindo `-aVR` |
| Somente faixas longas | `1x1-long`, `2x1-long`, `3x1-long` |
| 1×12 | Doze segmentos em uma única linha |

As faixas padrão são II, V1 e V5. A seleção pode mudar por amostra com `rhythm_leads`, um tensor `[B,3]` de IDs na ordem padrão. `-1` conserva a derivação definida pelo layout. Uma lista estática de três nomes também é aceita. As janelas curtas podem ser sequenciais ou simultâneas.

```python
batch = renderer(signals, fs=500, stage='all', return_instances=True,
                 return_scene=True, generator=rng, parameters={
    'layout': '3x4+r3',
    'rhythm_leads': ['II', 'V1', 'V5'],
    'surface_mode': 'crumpled',
    'background_type': 'wood',
    'paper_palette': 1,
    'object_shadow': 'hand',
})
```

`layout` também aceita IDs `[B]` para misturar layouts no lote. Use `renderer.layout_names` para verificar a ordem. O construtor aceita arranjos próprios, inclusive linhas com números diferentes de colunas e subconjuntos de derivações.

```python
from ecg_gpu.gpu_renderer import LayoutSpec

custom = LayoutSpec(
    'meu_layout',
    rows=(('I', 'II', 'III'), ('V1',), ('aVR', 'V5')),
    rhythm=('V6', 'II'),
)
renderer = ECGRenderer(RenderConfig(), layouts=[custom]).cuda()
```

Derivações curtas não se repetem. Repetições ficam nas faixas de ritmo, limitadas a três. Um layout personalizado recebe classe `unknown` na rede de metadata, a menos que seja exatamente um layout padrão. Isso permite gerar a imagem sem atribuir um nome de classe incorreto. O espaço de layouts continua extensível, sem afirmar que uma lista finita cobre todos os equipamentos existentes.

## Papel, geometria e iluminação

`surface_mode` oferece `flat`, `curl`, `fold`, `accordion`, `crumpled`, `wrinkled`, `rolled` e `corner`. O perfil `mixed` sorteia entre eles. `off` no RenderConfig desativa o relevo procedural. Altura, gradientes e normais são construídos na GPU com campos multiescala, rugas localizadas, dobras e curvatura. O mesmo relevo governa a projeção e a iluminação.

O modelo rápido é **2.5D**. A superfície tem uma altura por coordenada do papel. Sua projeção de relevo é invertida por iterações de Newton e composta com perspectiva, rotações, zoom, cortes e distorção radial. Regiões sem convergência ou com projeção local não injetiva recebem visibilidade zero. Ele não inventa observações nessas regiões. Esse modelo não representa integralmente uma folha dobrada sobre si mesma. Para isso, use o banco de mapas 3D abaixo.

A iluminação inclui três luzes coloridas, componente ambiente, reflexão difusa por normais, brilho aproximado, flash, oclusão ambiente e sombras de relevo por amostragem de raios na superfície de altura. Sombras de contato e sombras deslocadas escurecem o fundo da mesa. Mão, celular, persiana, janela, borda e elipse oferecem famílias de sombras suaves na cena. Essas sombras de objetos usam formas analíticas, sem modelos 3D de mãos/celulares.

O fundo procedural oferece madeira, mármore, laminado, tecido, metal, bancada clínica, fundo liso e couro. Orientação, escala e tonalidade variam. O papel tem oito paletas, do branco ao rosa, creme, amarelo, azul, verde, cinza e envelhecido. `paper_color` multiplica a paleta por RGB `[B,3]`. Use paleta zero para controlar a cor diretamente com pequena modulação de textura.

A impressão varia em fonte, cor/espessura de tinta, densidade, grade vermelha/azul/verde/cinza/laranja, quatro estilos de grade, ausência de grade, pulsos opcionais, labels e unidades ausentes. Grade 0 é contínua, 1 tem somente linhas de 5 mm, 2 tem pontos menores e linhas maiores, 3 usa pontos. Também há desgaste térmico, bandas de varredura, manchas, anéis de café e transparência aproximada da impressão do verso.

Rasgos retiram papel e visibilidade juntos. Adesivos, rasuras, caneta e até quatro textos inclinados podem ocluir o traçado. `artifact_texts` no construtor define os textos, preparados no atlas. Fontes manuscritas TTF/OTF fornecidas em `fonts` podem ser selecionadas por `artifact_font`. O pacote inclui textos neutros, sem nomes de pacientes. Não há geração automática de descrições clínicas.

Óptica/captura incluem desfoque e movimento por PSF anisotrópica, exposição, iluminação desigual, vinheta, balanço de branco, gamma e ruído dependente da intensidade. A simulação JPEG usa RGB/YCbCr, subamostragem 4:2:0, DCT 8×8 e tabelas de quantização. Não escreve um bitstream JPEG nem replica integralmente um codec.

## Usar documentos 3D e texturas externas

Doc3D e Inv3D motivam o uso de mapas inversos UV, normais, visibilidade e altura separados da textura do documento. Assim, um cenário de papel pode receber muitos ECGs diferentes, sem repetir a simulação física em cada iteração. A implementação abaixo é independente. Não copia seus renderizadores Blender e não inclui os respectivos acervos.

```python
from ecg_gpu.gpu_renderer import DocumentAssets

assets = DocumentAssets.from_npz('/assets/document-scenes.npz')
renderer = ECGRenderer(RenderConfig(), document_assets=assets).cuda()
```

| Campo NPZ | Contrato |
|---|---|
| `schema` | Texto escalar `ecg-document-assets-v1` |
| `inverse_field` | `[K,H,W,2]`, coordenadas inversas normalizadas para `grid_sample` |
| `mask` | `[K,1,H,W]`, visibilidade do papel em `[0,1]`, após oclusão 3D |
| `normals` | Opcional `[K,3,H,W]`, normais não nulas no papel visível |
| `height` | Opcional `[K,1,H,W]`, altura não negativa em meias larguras da página |
| `shading` | Opcional `[K,1,H,W]`, iluminação multiplicativa capturada offline |

O campo relaciona o pixel da cena à **imagem canônica do ECG**, com `align_corners=False`. Não é um mapa direto papel para câmera. Converta convenções de UV, ordem de canais, eixo vertical e unidades dos arquivos originais explicitamente antes de salvar o NPZ. O carregador exige esse schema para evitar aceitar silenciosamente um mapa incompatível. O banco pode representar superfícies sobrepostas quando seu mapa inverso e sua máscara já incorporam a seleção da superfície visível.

Por padrão, 65% das fotos elegíveis usam o banco, quando fornecido. Controle `asset_active` e `asset_index` para ablações. Sem banco, todas as cenas são procedurais. O teste executado usa mapas artificiais conhecidos e oclusão explícita, não um acervo real Doc3D. As normais procedurais estão no referencial local do papel. As normais externas conservam o referencial de câmera declarado pelo banco.

Papéis em branco e fundos fotográficos autorizados entram em `paper_textures` e `backgrounds`, ambos RGB `[K,3,H,W]` em `[0,1]`. Os bancos são carregados uma vez. Sorteio, recorte e amostragem ficam na GPU. A especificação registra hashes de todos os bancos.

```python
renderer = ECGRenderer(
    RenderConfig(profile='photo', severity=.8),
    paper_textures=paper_rgb,
    backgrounds=table_rgb,
    document_assets=assets,
    fonts=['/assets/fonts/sans.ttf', '/assets/fonts/handwriting.ttf'],
).cuda()
```

Não há downloads automáticos de fotos, pacientes, fontes ou meshes. Os exemplos entregues usam sinais inteiramente artificiais e texturas procedurais. A aparência precisa ser comparada com fotos externas para demonstrar realismo, mesmo quando se usam bancos fotográficos.

## Saídas e referência física

| Saída | Conteúdo |
|---|---|
| `image` | RGB float32 `[B,3,H,W]`, intervalo `[0,1]` |
| `reference` | Mesmo tensor de sinais recebido, sem alteração fisiológica |
| `targets` | Alvos da etapa nas coordenadas da imagem |
| `metadata` | IDs, sinais de amplitude, escala, janelas, visibilidade e parâmetros |
| `inverse_field` | Campo inverso com `return_field=True` |
| `canonical_image` | Impressão antes da câmera com `return_canonical=True` |
| `scene` | Normais, altura, luz, sombras e visibilidade com `return_scene=True` |

`stage='image'` fornece imagem e metadata, com `targets` vazio e `unit_text_visible=None`, pois não mede suporte dos glifos de unidades. `geometry` fornece vértices, grade e visibilidade. `trace` fornece 12 máscaras de traçado e 12 baselines. `metadata` fornece nomes, orientação, layout, velocidade e ganho. `all` reúne as tarefas e usa mais memória. Selecione apenas os alvos necessários no treino.

`return_instances=True` acrescenta 15 máscaras de instâncias e caixas `instance_boxes_xyxy` em pixels, com limites superiores exclusivos e zero para instâncias não observadas. As primeiras 12 instâncias seguem a ordem padrão. As últimas três são as faixas longas. `lead_id`, `amplitude_sign`, `active`, `start_s` e `end_s` têm forma `[B,15]`. Não considere slots inativos como dados impressos. O formato anterior tinha 13 slots.

Uma escala isotrópica em pixels/mm governa grade, tempo, amplitude e pulsos de 1 mV por 0,2 s. O desenho usa distância a segmentos limitados, amostragem em subpixels e envelope de todas as amostras originais, para evitar apagar extremos estreitos. A ocupação dos pixels é aproximada e a resolução limita a informação recuperável.

Imagem e alvos compartilham o mesmo campo inverso. Para coordenadas normalizadas, `x=(field_x+1)*W/2-0.5` e `y=(field_y+1)*H/2-0.5`. Na instância correspondente, `tempo=start_s+(x-x0)/(speed*ppm)` e `mV=amplitude_sign*(baseline-y)/(gain*ppm)`. O sinal de amplitude explica a inversão de aVR no Cabrera, mantendo a referência original.

Cruzamentos e faixas repetidas mantêm canais independentes por derivação. Baseline significa zero de tensão do desenho, sem remoção de deriva fisiológica. Cortes, rasgos e oclusões opacas têm `valid=0`. Unidades são classificadas como desconhecidas quando faltam ou perdem suporte de tinta suficiente. Essa regra não comprova legibilidade por OCR. Sombras e desfoque não tornam um sinal oculto recuperável.

## Integração no projeto ECG Replot completo

Esta seção descreve o treinador CSV do projeto completo. Ele não está incluído no pacote mínimo, que oferece o renderizador e as arquiteturas para uso no seu próprio laço de treino.

O treinador aceita `--signal-manifest` em lugar de `--manifest`. O CSV exige `patient_id,record_id,signals_path,split,source,manufacturer`. NPZs ficam abaixo de sua pasta e contêm `signals`, `fs` e `lead_names`, em mV. Pacientes, IDs, partições e hashes são auditados. Sinais `test` nunca são abertos como exemplos de treino/validação.

```bash
python -m training.prepare_ptbxl_v05 \
  --root /datasets/ptb-xl/1.0.3 --out /datasets/ecg-signals

python -m training.train_v05 \
  --config training/configs/v05-online.json \
  --signal-manifest /datasets/ecg-signals/signals.csv \
  --stage geometry --out /checkpoints/ecg-online
```

Repita para `trace` e `metadata`. A etapa trace usa `photo=False`, pois recebe imagens retificadas. Ela mantém variações de impressão e captura, sem geometria residual. O carregador completa a altura com branco até a resolução quadrada do modelo, preservando proporções.

Para controlar custo/qualidade use `--renderer-surface`, `--renderer-surface-resolution`, `--renderer-inverse-iterations` e `--renderer-shadow-samples`. `--renderer-assets` carrega o banco NPZ acima. A configuração inicial usa resolução de relevo 96, cinco iterações inversas e oito amostras por luz. Um caminho mais leve usa superfície `off` e zero amostras de sombra de relevo. Custo e qualidade devem ser medidos no hardware alvo.

Workers transportam arrays NumPy. Tensores e memória pinned são criados no processo principal. O modo `--signal-cache ram` usa zero workers efetivos, evitando duplicar todo o acervo. Com múltiplos GPUs, cada rank tem seu renderizador. Slurm aceita `ECG_SIGNAL_MANIFEST`. GPU/DDP/cluster ainda não foram executados aqui.

Seeds de treino variam por época, rank e lote. A validação sintética mantém as mesmas imagens. A retomada exige fontes, bancos, código, dados, versão/tipo de dispositivo, batch e ranks consistentes. Uma seed CPU não promete os mesmos pixels que CUDA. Os IDs iniciais de layouts foram preservados, mas a cabeça de metadata passou de 6 para 31 classes. Pesos e checkpoints anteriores exigem tratamento explícito. A assinatura de retomada rejeita mudanças de código/configuração.

## Medir e gerar uma galeria

```bash
python -m ecg_gpu.benchmark \
  --device cuda --height 768 --width 1024 --batch-size 4 \
  --stage image --steps 50 --no-examples --out /benchmarks/ecg-renderer

python -m ecg_gpu.gallery \
  --device cuda --out /examples/ecg-scenes --zip /examples/ecg-scenes.zip
```

O benchmark registra configuração, GPU, PyTorch, mediana/P90, imagens/s e pico de memória CUDA. Inclui desenho e captura, com os alvos selecionados por `--stage`. `image` mede o caminho sem máscaras densas. Exclui dados em disco, transferência de sinais, fontes, PNG e execução da rede. `--canonical` mede a impressão retificada. `--signals` aceita NPZ autorizado, carregado antes da medição. Sem ele, o ECG é inteiramente artificial.

A galeria entrega 16 cenas controladas, os 30 layouts, mapas de altura/normais e uma sobreposição das máscaras. As fotos são exportadas em JPEG qualidade 95 por padrão, fora do treino. `wood-crumpled.png` mantém um exemplo sem perdas para auditoria. Use `--format png` para exportar todas as cenas sem perdas. Ela usa parâmetros fixados para permitir inspeção e não representa uma distribuição clínica validada. Os resultados executados estão em `GPU_RENDERER_VERIFICATION.json`. A cobertura funcional e as lacunas estão em `GPU_RENDERER_FEATURES_V2.json`.

## Referências e limites da comparação

O ECG-Image-Kit é referência para impressão de ECG, manuscritos, rugas, dobras e perspectiva. Doc3D/DewarpNet e Inv3D são referências para separar geometria, UV, normais, visibilidade, albedo e iluminação. Esta é uma implementação própria. Não há paridade verificada com todos os argumentos/formatos da CLI do ECG-Image-Kit, nem reprodução de seu renderizador ou dos renderizadores Blender.

Para demonstrar maior robustez, treine com orçamento, pacientes, arquitetura e otimização iguais e compare a digitalização em fotos reais pareadas externas. Avalie fabricantes, layouts, captura e artefatos, incluindo falhas, cobertura, SNR e distorção da onda. Congele a reserva antes de ajustar o gerador a suas fotos. Diversidade implementada e testes de software não estabelecem realismo superior ou acurácia clínica.

Fontes primárias: [ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit), [versão do desafio 2024](https://zenodo.org/records/12731152), [artigo DOI 10.1088/1361-6579/ad4954](https://doi.org/10.1088/1361-6579/ad4954), [DewarpNet](https://www3.cs.stonybrook.edu/~cvl/projects/dewarpnet/storage/paper.pdf), [Doc3D renderer](https://github.com/sagniklp/doc3D-renderer), [Inv3D generator](https://github.com/FelixHertlein/inv3d-generator), [PyTorch grid_sample](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) e [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/).
