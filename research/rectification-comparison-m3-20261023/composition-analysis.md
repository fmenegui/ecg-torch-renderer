# Protótipo composicional: desenvolvimento analítico

Protocolo congelado antes da primeira execução: `cb7051bd233d3e45c8642c933b0321debde9dbbdaebaec3cfdccc1fc19b30371`.

Foram executadas 12 cenas e cinco braços, com 4.096 queries físicas por cena. O protocolo não consulta banco M2/M3, detector ou RGB. IDs e correspondências são fornecidos. Os mapas são parciais, restritos ao mesmo suporte de células observadas. A composição usa F=T∘H e a inversa H^-1∘T^-1, com T e T^-1 no mesmo complexo C0. Não é uma reprodução de Powerful Medical.

As geometrias são mapas 2D analíticos. Não são uma simulação calibrada de câmera/papel desenvolvível nem fotografias externas. Nenhum braço é escolhido por erro gold. LS e RANSAC são os dois inicializadores predeclared, sem busca posterior de parâmetros.

## RMSE físico, em mm

| Controle | Células M2 | H global LS | H global RANSAC | Composição RANSAC |
| --- | ---: | ---: | ---: | ---: |
| flat-affine | 4.69403e-15 | 1.34534e-14 | 1.34534e-14 | 5.52618e-15 |
| flat-projective | 0.0666196 | 7.2078e-07 | 7.2078e-07 | 2.66955e-09 |
| flat-projective-strong | 0.159034 | 1.20554e-06 | 1.20554e-06 | 2.57699e-09 |
| smooth | 0.0881139 | 1.65033 | 2.20367 | 0.0545206 |
| crease-aligned | 0.0666196 | 3.02794 | 6.62571 | 1.80951e-09 |
| crease-oblique | 0.0934773 | 2.44078 | 6.23173 | 0.0761998 |
| flat-noise-low | 0.131916 | 0.0231028 | 0.0231028 | 0.106051 |
| flat-noise-high | 0.264066 | 0.0261732 | 0.0261732 | 0.256073 |
| smooth-noise | 0.222938 | 1.64804 | 2.25408 | 0.208532 |
| crease-oblique-noise | 0.234441 | 2.43733 | 11.4043 | 0.22478 |
| flat-coherent-row-error | 1.26467 | 0.755375 | 1.07772 | 1.2607 |
| flat-coherent-half-error | 4.35208 | 4.23984 | 4.42177 | 4.39216 |

## O que a composição acrescenta

Na projetiva plana forte, o viés de interpolação por células cai de 0,15903 mm para aproximadamente 2,58×10^-9 mm. Esse valor de precisão numérica só vale para o caso ideal de correspondências exatas. Na curva suave, o RMSE passa de 0,08811 para 0,05452 mm. Na crease alinhada à malha, 0,06662 para aproximadamente 1,81×10^-9 mm. O prior projetivo deixa de ser substituído por uma aproximação afim no interior da célula.

Os dois braços globais, sem residual, erram fortemente em curva e crease. Isso é a insuficiência esperada do modelo global para essas geometrias, não evidência de que um produto concorrente real se comportaria assim.

## Negativos mantidos

* Na crease oblíqua limpa, RMSE melhora de 0,09348 para 0,07620 mm, mas W aumenta de 0,29297% para 0,39063%. O efeito não pode ser vendido como melhora em todos os endpoints.
* Na plana com ruído alto, W aumenta de 5,83496% para 6,17676%, apesar de um pequeno ganho de RMSE. H global LS obtém RMSE de 0,02617 mm, enquanto a composição tem 0,25607 mm. Preservar exatamente os anchors ruidosos continua sendo uma limitação.
* No erro coerente por linhas, W aumenta de 72,60742% para 73,55957%.
* Na corrupção coerente de metade da folha, RMSE piora de 4,35208 para 4,39216 mm e W de 70,97168% para 71,50879%. A composição conserva o erro do frontend, não recupera IDs ou posições verdadeiras.

## Continuidade, condicionamento e limites

A maior diferença direta entre os dois lados de uma aresta compartilhada foi 3,18×10^-14 mm. A diferença inversa foi zero nas verificações explícitas. Isso é continuidade C0 no suporte observado, não continuidade de derivadas, precisão física ou garantia global fora do domínio.

Uma injeção de correspondências com os dois extremos inferiores trocados produz uma H ajustada com denominador de -1 a 1. O guard rejeita o caso por horizonte cruzando o casco observado. Anchors colineares também são rejeitados. Essas são verificações unitárias do protótipo, não imagens do benchmark.

O status de H `accepted` significa somente que passou por verificações numéricas e de horizonte. Não existe calibração de risco nem gate mínimo da fração de inliers neste protótipo. O limiar RANSAC de 1,5 mm é um parâmetro de ajuste em coordenadas canônicas, não uma tolerância clínica. O caso com anchors coerentemente errados prova que passar pelo guard não assegura boa geometria.

Não há extrapolação de frame, orientação autônoma, decisão de escala, reamostragem de RGB, latência end-to-end ou modelo de rede. A política de composição ainda precisaria resolver esses contratos antes de ser candidata a um sistema completo. Não houve alteração de M2 nem seleção de configuração a partir dos scores. O próximo teste de método não pode tratar estas cenas motivadoras como confirmação independente.

Reprodução a partir da raiz do repositório: executar `composition_probe.py --geometry-dir ecg_gpu/digitization --out build/composition-probe`. O arquivo `composition-protocol.json` deve ficar ao lado do script. O módulo `cell_mesh.py` é verificado por SHA256 antes da execução.
