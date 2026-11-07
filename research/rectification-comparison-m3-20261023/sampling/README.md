# Amostragem: controle separado a 4 px/mm

A integração da área do pixel eliminou os desaparecimentos completos observados no sampler pontual nos controles declarados, mantendo a mesma densidade de saída. Isso é um resultado de componente: não houve detector, imagem do banco, extração de waveform ou mudança de M2/M3.

O protocolo foi congelado antes dos scores. Foram executadas 280 cenas: dois mapas afins exatos, larguras de faixa de 0,1 e 1 mm, sete orientações e dez fases; cada uma recebeu todos os sete braços. O baseline chama o `render_complete` original, com hash verificado, a 4 px/mm. Os candidatos compõem cada subponto físico diretamente com a inversa do mapa e amostram o RGB original; não usam uma imagem intermediária já deformada. Quadraturas midpoint 2×2, 4×4 e 8×8 foram declaradas tanto com reconstrução bilinear quanto com pixel constante na célula de origem (`boxhold`).

## Referência e contrato fotométrico

A referência integra uma faixa ideal sobre o quadrado físico de cada pixel de saída, por fórmula fechada da soma de duas uniformes, com limite exato de sobreposição de intervalos nas direções dos eixos. A observação de origem é a média exata da faixa em cada célula de pixel-fonte, quantizada uma vez para uint8. Portanto erro de formação da observação, quantização e reconstrução permanece visível contra essa referência ideal.

Neste experimento sintético, cinza é definido numericamente como `255 × (1 − cobertura)`. Os candidatos calculam a média dos **códigos RGB** fornecidos. Não foi feita conversão de sRGB para luz linear; não reivindicamos conservação de radiância, uma câmera física ou superioridade radiométrica. Imagens reais exigiriam um contrato fotométrico próprio. Uma observação já perdida na origem não pode ser recuperada pela quadratura.

O ROI físico `[2,12] × [2,12]` mm é fixo, possui 41×41 centros por mapa e está inteiramente dentro da origem, inclusive para todos os subpontos. Os braços não aplicam máscara de confiança ao RGB. O segundo mapa testa anisotropia e shear, mantendo geometria afim exata; não representa deformação não linear de papel.

## Faixas de 0,1 mm

Cada linha agrega os mesmos 140 casos. Energia é soma de escuridão vezes área física do pixel; o erro relativo é calculado por cena, antes da média absoluta. MAE e erro de pico são de escuridão normalizada em `[0,1]`, não erro geométrico em mm.

| Braço | Desaparecimentos | MAE médio por pixel | Erro absoluto médio de energia | Pior erro absoluto de energia | Erro absoluto médio do pico |
|---|---:|---:|---:|---:|---:|
| Bilinear pontual nativo | 4/140 | 0,004629 | 16,9329% | 150,0000% | 0,33469 |
| Área bilinear 2×2 | 0/140 | 0,001325 | 1,1839% | 9,8039% | 0,03169 |
| Área bilinear 4×4 | 0/140 | 0,001528 | 0,5139% | 5,8824% | 0,02263 |
| Área bilinear 8×8 | 0/140 | 0,001604 | 0,1594% | 0,9804% | 0,02499 |
| Área boxhold 2×2 | 2/140 | 0,003642 | 6,8860% | 100,0000% | 0,11876 |
| Área boxhold 4×4 | 0/140 | 0,001758 | 3,3629% | 37,2549% | 0,07436 |
| Área boxhold 8×8 | 0/140 | 0,001421 | 1,4289% | 24,5098% | 0,02542 |

Essas contagens descrevem o painel analítico determinado; não são prevalência em ECGs reais. O pior erro de energia de 150% significa excesso de energia por alias, além dos casos de desaparecimento. Integrar uma reconstrução bilinear preservou algum contraste em todas as fases/orientações testadas. A quadratura boxhold 2×2 ainda perdeu faixas por alias e permanece como controle negativo.

## Controles de 1 mm e cor constante

| Braço | Desaparecimentos | MAE médio por pixel | Erro absoluto médio de energia | Pior erro absoluto de energia |
|---|---:|---:|---:|---:|
| Bilinear pontual nativo | 0/140 | 0,005065 | 0,25775% | 2,72315% |
| Área bilinear 2×2 | 0/140 | 0,001568 | 0,01977% | 0,09804% |
| Área bilinear 4×4 | 0/140 | 0,001975 | 0,01352% | 0,09804% |
| Área bilinear 8×8 | 0/140 | 0,002116 | 0,01477% | 0,19608% |
| Área boxhold 2×2 | 0/140 | 0,004672 | 0,11009% | 0,81022% |
| Área boxhold 4×4 | 0/140 | 0,002132 | 0,04145% | 0,26508% |
| Área boxhold 8×8 | 0/140 | 0,001809 | 0,01793% | 0,10394% |

Todos os braços mantiveram pico 1 nas faixas de 1 mm. Os 14 controles de cor constante `[47,128,219]`, em dois mapas e sete braços, devolveram exatamente a cor no ROI. Os 31 controles numéricos incluem formas fechadas simples, inversa do mapa contra álgebra afim independente, bounds e origem nativa.

A auditoria independente passou **32.558 verificações**: 4.276 integrações por recorte de polígonos validam a referência por outro algoritmo; 4.200 verificam a formação dos pixels-fonte; todos os scores e agregados foram recalculados; os 280 resultados nativos e 120 quadraturas selecionadas foram reproduzidos por inversa afim direta. Houve 78 comparações de quantização em empate de meio LSB com arredondamento oposto, sempre dentro de 1 LSB e registradas na auditoria. Os controles constantes foram verificados por metadata, sem imagens constantes independentes salvas. Fonte, protocolo e resultados tiveram seus hashes confirmados em `../qa/sampling-candidate-review.json`.

## Negativos, custo e decisão

- Mais subpontos **não melhoram todos os endpoints**: na família bilinear, o MAE aumenta de 2×2 para 8×8 nas duas larguras. Em 1 mm, 8×8 também piora ligeiramente o erro de energia em relação a 4×4. A aproximação converge para uma reconstrução que já tem blur, não necessariamente para a faixa ideal.
- Os dois desaparecimentos de boxhold 2×2 foram mantidos. Mesmo boxhold 8×8 tem pior erro de energia de 24,51% na faixa fina. O fato de reter algum contraste não garante fidelidade da amplitude ou largura.
- A quadratura exige 4, 16 ou 64 amostras e avaliações da inversa por pixel. Preparar os mapas custou aproximadamente 27–29 ms, 119–152 ms e 480–625 ms, respectivamente, nestes pequenos frames. O cache variou de 0,22 MB a 4,22 MB. A 16 milhões de pixels, um cache completo 8×8 de coordenadas float32 exigiria cerca de 8,2 GB; este protótipo precisa de execução em blocos para escalar.
- A aplicação bilinear ao RGB com mapas prontos levou medianas aproximadas de 0,35/1,07/4,23 ms na faixa fina. O baseline completo levou 13,49 ms, incluindo mapa/evidência. **Não são escopos de timing iguais**; o custo de preparação não pode desaparecer numa comparação de velocidade. A máquina compartilhava a inferência do coordenador.
- O primeiro run tinha OpenCV limitado a uma thread, mas OpenBLAS permitia nove. Ele foi preservado em `results-initial-unbounded-blas/`. A execução final fixa OpenBLAS/OMP/MKL e OpenCV em uma thread. Os 1.960 registros de qualidade e os dois NPZ são exatamente idênticos entre as execuções; somente tempos mudaram. `timing-rerun-receipt.json` documenta o desvio e a repetição.
- A cobertura de uma faixa inteira não prova retenção de cada segmento curto, spike, caractere ou detalhe arbitrariamente fino. Fases contínuas, ruído, redução mais forte, campo não linear, bordas, cor e dispositivos reais permanecem fora deste painel. Nem esta quadratura nem 8 px/mm têm suficiência universal demonstrada.

O resultado justifica investigar um sampler integrado em área, com contrato explícito de reconstrução e custo. Não selecionamos automaticamente um braço nem mudamos um default. A próxima integração deve ser em blocos, preservar exatamente a composição geométrica e ser validada com traçado e texto conhecidos, além do teste externo.

## Artefatos e reprodução

- `protocol.json`: planejamento congelado, SHA256 `85adf1e4fe3f37b14fff7d6f32aea98e1db2828917e732bd7e3404653e5cd58a`.
- `sampling_component.py`: protótipo e referência analítica. Importa o sampler nativo com hash; nenhum módulo congelado foi alterado.
- `results/results.json`: todos os 1.960 resultados e 14 agregados, custos e controles; `summary.csv` expõe todos os agregados.
- `results/affine-isotropic.npz` e `results/affine-anisotropic-shear.npz`: imagens-fonte, verdade, predições de todos os braços e coordenadas do ROI, sem casos removidos.
- `results/contact-sheet.png`: três cenas e crop declarados antes dos scores, referência mais todos os braços. Ampliação visual nearest somente para mostrar os pixels de saída.
- `receipt.json`: hashes dos artefatos e da revisão independente `../qa/sampling-candidate-review.json`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python sampling_component.py --geometry-dir ../../ecg-torch-renderer/ecg_gpu/digitization --out reproduction
```

Executar a partir deste diretório com NumPy, SciPy, OpenCV e Pillow disponíveis. Os valores quantitativos são de componente CPU; não há resultado de GPU ou de pipeline clínico.
