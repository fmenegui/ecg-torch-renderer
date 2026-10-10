# ECG Torch Renderer

Biblioteca independente v0.3.0 de impressões e fotografias de ECG em PyTorch.
Entrada [B,12,T], em mV, ordem I, II, III, aVR, aVL, aVF, V1..V6.
Fontes são preparadas uma vez na CPU. Renderização por lote ocorre no
dispositivo escolhido. A execução CUDA ainda precisa ser medida.

```bash
python -m pip install -e .
```

Instale uma build do PyTorch com CUDA adequada ao seu computador usando
https://pytorch.org/get-started/locally/. O pacote não instala drivers.

```python
import torch
from torch import nn
from ecg_gpu import ECGImageLayer, RenderConfig, Uniform, Choice, Bernoulli

images = ECGImageLayer(fs=500, config=RenderConfig(height=384, width=512),
    parameters={
        'layout': Choice(['3x4+r3', '6x2', '12x1']),
        'angle': Uniform(-.15, .15),
        'header_present': Bernoulli(.9),
        'gain_index': 1,
    }, seed=1973)
cnn = nn.Sequential(nn.Conv2d(3, 16, 5, stride=4), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 2))
pipeline = nn.Sequential(images, cnn).cuda()

# Dentro do treino, signals_cuda tem forma [B,12,T]
logits = pipeline(signals_cuda)
loss = nn.functional.cross_entropy(logits, classes_cuda)
loss.backward()
```

A camada conserva a renderização em float32 quando a CNN usa autocast.
Mova a pipeline com .to(device) ou .cuda(), sem .half() no renderizador.
No treino CUDA com float16, use torch.amp.GradScaler para a CNN.

Para fundir os blocos de curvas e calibração durante treinos longos, passe
ECGImageLayer(..., compile_raster=True). A primeira chamada pode compilar.
Sorteios, contadores e câmera permanecem fora desses blocos. A versão sem
compilação continua disponível. O backend deve ser igual ao retomar um checkpoint.
Esse modo foi executado em CPU. O ganho CUDA precisa de benchmark na sua placa.

## Colab e profiling detalhado

Abra notebooks/colab_gpu_profile.ipynb no Google Colab, selecione GPU e execute
todas as células. O notebook instala este pacote, compara a referência 0.2.1,
o backend eager e o backend compilado, executa os testes e gera exemplos.
No final, baixa um único ZIP com os resultados, mesmo se um benchmark falhar.

```bash
ecg-renderer-profile --device cuda --preset full --batch-size 4 \
  --compile-raster --steps 30 --warmup 8 --out profile-gpu
ecg-renderer-profile --device cuda --preset balanced --batch-size 4 \
  --compile-raster --train-step --out profile-training
```

report.json, operators.csv, kernels.csv, tables.txt e trace.json.gz ficam
na pasta indicada e no ZIP correspondente. --memory-timeline inclui stacks
e uma linha do tempo das alocações. Descompacte o trace e abra no Perfetto.
O tempo instrumentado não é usado como latência do benchmark. A primeira
chamada, incluindo compilação/cache, é registrada separadamente.
Se CUDA/CUPTI não coletar kernels, o relatório registra essa limitação.

```python
from ecg_gpu import profile_layer
# Use uma instância descartável. Warmup e medição avançam o RNG e o treino.
report = profile_layer(images, signals_cuda, 'profile-custom')
# Para a sua rede, workload pode executar forward, backward e optimizer.step.
```

O teste --train-step usa uma CNN pequena e artificial. Para estimar a sua
vazão de treino, passe seu próprio workload. Inputs devem estar no dispositivo.
Valores do sinal e metadados de pacientes não são exportados no relatório.

ECGImageLayer devolve somente o tensor RGB [B,3,H,W] em [0,1].
Não grava imagens durante o treino e dispensa PIL/NumPy no forward.
O passo de renderização não é diferenciável em relação ao sinal de entrada.
O backward treina normalmente a rede de imagem que vem depois dele.

Parâmetros ausentes são sorteados pelo perfil. Valores fixos, Uniform,
Choice e Bernoulli são compilados para buffers e acompanham .to(device).
Use .eval() ou randomize=False para repetir as imagens. state_dict inclui
o contador de chamadas e as distribuições para retomar a sequência.
Em DDP, passe uma seed diferente por rank. CPU e CUDA não prometem pixels iguais.
Taxas/comprimentos por amostra e overrides tensoriais entram por
images(signals, fs=rates, lengths=lengths, parameters=overrides).

CLI para seu sinal NPY/NPZ ou WFDB opcional:

```bash
ecg-render --input sinal.npz --output ecg.png --device cuda --layout 3x4+r3
ecg-render --input sinal.npy --fs 500 --units mV --output ecg.png --device cpu
ecg-render --input lote.npz --output imagens --device cuda \
  --parameters '{"layout":{"choice":["6x2","12x1"]},"angle":{"uniform":[-0.15,0.15]}}'
```

NPZ contém signals, fs e, opcionalmente, lead_names, units e lengths.
A CLI reordena derivações e converte uV/V para mV. NPY requer --fs.
Para WFDB, instale pip install '.[wfdb]' e passe o arquivo .hea.
Os arquivos WFDB conservam suas unidades declaradas por canal.
Exportação PNG/JPEG e JSON ocorre na CPU, após a renderização.
Também funciona como python -m ecg_gpu.gpu_cli.

São 30 layouts e LayoutSpec para arranjos personalizados. Inclui 1, 2 ou 3
faixas longas com seleção de derivações, Cabrera com -aVR, oito superfícies
procedurais com relevo, oito materiais de mesa, cores de papel, sombras de
mão/celular/persiana/janela, rasgos, manchas, anotações e transparência do verso.

```python
renderer = images.renderer
batch = renderer(signals_cuda, fs=500, stage='all', return_scene=True,
                 return_instances=True, parameters={
                     'layout': '3x4+r3',
                     'rhythm_leads': ['II', 'V1', 'V5'],
                     'surface_mode': 'crumpled',
                     'background_type': 'wood',
                     'object_shadow': 'hand',
                 })
```

A geração procedural usa relevo 2.5D e iluminação por normais. Não é uma
simulação de tecido ou path tracing. DocumentAssets aceita mapas inversos
UV, visibilidade, normais e altura previamente renderizados, incluindo
superfícies 3D com papel sobreposto. O pacote não inclui acervos Doc3D/Inv3D.

ECGRenderer retorna RenderBatch com referência, metadata e alvos alinhados.
images.render_batch oferece a mesma saída usando seus parâmetros configurados.
As redes auxiliares incluídas são arquiteturas, sem pesos treinados. Para treinar o
modelo de traçado após retificação, use stage='trace', photo=False e
build_model('trace'). Ganho/velocidade/layout/orientação usam stage='metadata'.
stage='all' fornece todas as máscaras e exige mais memória.

Benchmark e exemplos, fora do laço de codificação de imagens:

```bash
python -m ecg_gpu.benchmark --device cuda --out benchmark --batch-size 4 --stage image --steps 50 --no-examples
python -m ecg_gpu.gallery --device cuda --out gallery --zip examples.zip
ecg-renderer-audit --device cuda --out audit --batches 1 4 --steps 50 --warmup 10
ecg-renderer-audit --device cuda --out audit-fused --batches 1 4 --steps 50 --warmup 10 --compile-raster
```

Sem --signals, o benchmark usa um ECG inteiramente artificial. --signals
aceita NPZ com signals [12,T], fs e lead_names em mV. --device cpu permite
verificação sem GPU. Para traçado retificado, acrescente --canonical.
O JSON registra mediana/P90, imagens/s e memória CUDA quando disponível.
O audit mede o forward público de ECGImageLayer, incluindo seu RNG, nos
presets fast, balanced e full. Profiling de tensores ocorre fora da medição.
PERFORMANCE.md explica a comparação pareada em CPU e os limites da medida.

```bash
python -m unittest discover -s tests -v
```

Os testes portáteis verificam unidades, tempo, extremos estreitos,
cruzamentos, deformações, oclusões, compressão, nn.Sequential, replay de
checkpoint e conversão CLI. WFDB foi exercitado com um leitor simulado.

A referência conserva os sinais originais. Somente janelas efetivamente
impressas são observáveis. Cruzamentos preservam identidades independentes.
Oclusões/cortes têm máscaras de visibilidade. Pixel/tempo/mV e parâmetros
ficam no metadata. Não preencha segmentos não observados como se fossem
recuperados. Consulte METODOLOGIA.md para o contrato e as aproximações.

VERIFICACAO.json contém resultados CPU. Não há benchmark CUDA, comparação
de desempenho com ECG-Image-Kit ou validação clínica demonstrados.
Os testes e o ensaio de treinamento usaram sinais artificiais.

Consulte COBERTURA.json para recursos implementados, aproximações e lacunas.

O treinador completo do projeto ECG Replot já aceita CSV de sinais com
partições por paciente e gera imagens em cada lote. Essa integração usa
training.train_v05 --signal-manifest e não integra este pacote mínimo.

Implementação original MIT. Referência de geração: ECG-Image-Kit utilizado
no PhysioNet 2024, https://zenodo.org/records/12731152.
