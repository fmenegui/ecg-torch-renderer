# Desempenho e profiling, versão 0.3.0

O backend opcional `compile_raster=True` funde a cobertura de curvas e os pulsos
de calibração com `torch.compile`, `fullgraph=True` e formas dinâmicas. A câmera,
as deformações, o RNG e os contadores permanecem fora desses blocos. A amostragem
dos subsegmentos de antialiasing ocorre em lote. A união pode reduzir as derivações
dentro do bloco compilado quando só a imagem é solicitada. Não se corta uma banda
fixa ao redor da baseline, que poderia eliminar picos altos e cruzamentos.

Os cálculos conservam float32, inclusive quando a CNN seguinte usa AMP. A fusão
pode reduzir launches e intermediários, mas não demonstra ganho em uma GPU sem
executar o benchmark nessa GPU. Não foi executado CUDA nesta versão local.

## Comparação pareada executada em CPU

PyTorch 2.14.1+cpu, dois threads, sinais artificiais [B,12,5000], fs=500 Hz,
perfil mixed, severidade 0,6. Dez chamadas medidas após primeira chamada e três
warmups. Cada backend recebe a mesma sequência de seeds. A ordem é alternada
para reduzir o viés de execução. A referência 0.2.1 é o ZIP congelado da entrega
anterior. Mediu-se o `ECGImageLayer.forward` completo com RNG, deformações e câmera.
Leitura/transferência do sinal, preparação das fontes, PNG/JPEG em arquivo e CNN
ficam fora do tempo. A primeira chamada compilada foi registrada separadamente.

| Preset | Imagem | B | 0.2.1 ms | 0.3.0 eager ms | Compilado ms | Compilado imagens/s | Ganho pela mediana |
|---|---|---:|---:|---:|---:|---:|---:|
| fast | 384×256 | 1 | 76,30 | 73,79 | 71,42 | 15,11 | 1,07× |
| fast | 384×256 | 4 | 192,63 | 191,61 | 157,04 | 23,82 | 1,23× |
| balanced | 512×384 | 1 | 143,63 | 140,02 | 128,84 | 8,69 | 1,11× |
| balanced | 512×384 | 4 | 522,70 | 617,88 | 408,05 | 10,28 | 1,28× |
| full | 1024×768 | 1 | 637,97 | 567,57 | 451,01 | 2,29 | 1,41× |
| full | 1024×768 | 4 | 4862,73 | 5237,13 | 1788,64 | 2,23 | 2,72× |

Imagens/s utiliza a soma dos tempos, enquanto ganho utiliza a mediana. Por isso,
as razões dessas colunas não precisam coincidir. Sem compilação, o ganho não foi
consistente. O novo eager ficou cerca de 18% mais lento em balanced B4 e 8% em
full B4 nessa execução CPU. Não há garantia de aceleração para todo dispositivo,
tamanho de lote ou resolução. O backend eager permanece como opção de compatibilidade.

Os JSONs completos em `GPU_RASTER_COMPARISON.json` no projeto ou
`PERFORMANCE_CPU.json` no pacote incluem p90, cold start, hashes e configurações.
Essas medidas não são estimativas de FPS CUDA. Os números da auditoria anterior
0.2.1 em `PERFORMANCE_GPU.md` pertencem a outra execução e não são uma comparação
controlada com 0.3.0.

## Fidelidade e integração

As máscaras eager reproduzem bit a bit a referência congelada nos 30 layouts,
com taxas/comprimentos variáveis, traçados ausentes, larguras diferentes,
extremos estreitos, amplitudes elevadas e antialiasing de 1, 2, 3, 4 ou 8 amostras.
Os testes compilados de máscaras abrangem seis layouts, com tolerância absoluta
de 0,000002. Na comparação pública de RGB, os seis casos CPU tiveram erro máximo
0 nas imagens amostradas. Isso verifica regressão de software, não qualidade
clínica nem equivalência universal entre CPU e CUDA.

Mudanças de antialiasing ou formas podem exigir outra compilação. Use dimensões
e configurações estáveis no treino e preserve `compile_raster` no checkpoint.
Não se compila o renderizador completo e não se aplica `.half()` a seus buffers.

## Profiling detalhado em um comando

```bash
ecg-renderer-profile --device cuda --preset full --batch-size 4 \
  --compile-raster --steps 30 --warmup 8 --profile-steps 3 --out profile-gpu
```

No projeto completo, use `python -m scripts.profile_gpu_renderer`. Em CPU, troque
`--device cuda` por `--device cpu`. A pasta precisa estar vazia. O resultado é
também arquivado em `profile-gpu.zip`. `--input` aceita NPY/NPZ/WFDB, com `--fs`
e `--units` quando necessários. Os dados já estão no dispositivo antes do timing.

| Arquivo | Conteúdo |
|---|---|
| report.json | Ambiente, configuração, hashes, primeira chamada, mediana/p90, vazão, memória CUDA, regiões e limitações |
| operators.csv | Operadores agrupados por shape, tempos próprios/totais e deltas de memória |
| kernels.csv | Atividades CUDA coletadas, nomes, chamadas e durações agregadas |
| tables.txt | Tabelas de operações com maior tempo próprio |
| trace.json.gz | Trace Chrome/Perfetto de CPU e, quando disponível, GPU |
| memory-timeline.json.gz | Linha do tempo de alocações, solicitada com --memory-timeline |

O timing sem instrumentação e os CUDA Events usam sincronização explícita.
Warmup/compilação precedem o profiling. As iterações instrumentadas ocorrem
separadamente e têm overhead. Stacks ficam desligados por padrão. `--stacks` ou
`--memory-timeline` os habilitam. Não some regiões pais/filhas como se fossem
tempos independentes. Eventos GPU concorrentes também podem se sobrepor.

A memória de pico do allocator não equivale a tráfego de DRAM ou à VRAM total
do processo. O relatório não mede ocupação de SMs, utilização ou largura de
banda efetiva. Se CUPTI não coletar kernels, a ausência fica explícita em
`cuda_kernel_activity_captured`. Para esses contadores, use Nsight em uma GPU
compatível após identificar os blocos mais caros no trace.

## Treino on the fly e Colab

`--train-step` inclui renderização, CNN pequena, AMP, backward e AdamW. É uma
verificação funcional com sinais artificiais. Não representa uma rede clínica.
Para medir sua rede, use `profile_layer(layer, signals, out, workload=step)`.
Use uma instância descartável, pois RNG, pesos e optimizer avançam durante
warmup, medição e profiling.

`notebooks/colab_gpu_profile.ipynb` prepara o ambiente GPU, instala a biblioteca
conservando a build de Torch do Colab, executa testes, compara os três backends,
exporta profiling eager/compilado e treino, gera imagens e baixa um ZIP completo.
Falhas de subprocessos são preservadas em logs e em failures.json. Ainda não
houve execução deste notebook em um runtime Colab nesta entrega.

Referências primárias: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html,
https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html e
https://research.google.com/colaboratory/intl/pt-BR/faq.html.
