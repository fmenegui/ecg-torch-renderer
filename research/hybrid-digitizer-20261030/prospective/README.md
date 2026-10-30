# Ensaio prospectivo em GPU e complementos posteriores

Executamos o npj original, nosso detector de grade seguido do npj e o notebook original do segundo colocado do Kaggle em uma A40. São 24 fotografias, oito pacientes e quatro layouts. Os pacientes não participaram dos nossos grupos anteriores. O compartilhamento com o treinamento dos autores permanece desconhecido. Modelos, pesos, escala de 3000 pixels e política foram congelados antes da inferência. O worker recebe imagens e não recebe referências.

| Média por paciente, todas as 24 imagens | Cobertura impressa | NRMSE bruto |
| --- | ---: | ---: |
| npj direto, ensaio primário | 81,43% | 1,1276 |
| Grade seguida do npj, ensaio primário | 64,72% | 0,9178 |
| npj direto, após completar a última imagem | 85,29% | 1,1498 |
| Grade seguida do npj, após completar a última imagem | 68,65% | 0,9133 |
| Seleção conservadora, após o complemento | 85,29% | 1,1498 |

O seletor mantém exatamente o sinal direto em todas as imagens. A grade tem menor erro agregado, mas perde 16,64 pontos percentuais de cobertura. Isso não demonstra superioridade. Cobertura é suporte numérico finito, não correção. O erro bruto mantém a origem temporal e usa zero somente dentro do cálculo da métrica para ausências. Os sinais salvos conservam NaNs. Não há alinhamento, ajuste de ganho ou de linha de base nos resultados primários. Diagnósticos alinhados permanecem separados.

A execução primária [41106791069](https://github.com/fmenegui/ecg-torch-renderer/actions/runs/41106791069) terminou normalmente, mas seu limite interno deixou a última imagem sem leitura direta e impediu o início do batch Kaggle. Esses casos continuam no denominador original em `primary-summary.json`. A continuação [41113820943](https://github.com/fmenegui/ecg-torch-renderer/actions/runs/41113820943) concluiu a imagem interrompida. A avaliação combinada é um complemento após a abertura dos resultados primários, não um segundo teste independente.

## Kaggle: execução completa e falha de normalização

O layout inferido e congelado tornou seis imagens de dois pacientes elegíveis. As outras 18 permanecem explicitamente fora do escopo do especialista. Na primeira continuação, a normalização falhou e um bloco de demonstração gráfica depois do tratamento de erro encerrou o notebook: esperava cinco campos onde havia quatro. Essa execução e seu erro estão preservados.

Revisamos os três blocos de demonstração gráfica do notebook v6. A adaptação omite somente esses blocos, reconhecidos por hashes de AST. Um bloco diferente é recusado. Cálculos numéricos, checkpoints e listas nativas de falhas são mantidos. Os testes verificam que as chamadas numéricas e o ensemble permanecem no código.

A execução [41116353783](https://github.com/fmenegui/ecg-torch-renderer/actions/runs/41116353783) completou o notebook na GPU. Os oito modelos tiveram todas as chaves dos pesos carregadas, com timm 1.0.22 no subprocesso nativo. As seis imagens falharam na etapa zero, portanto as etapas seguintes não processaram imagens válidas. Quatro emitiram mensagens de orientação desconhecida. O tratamento de erro original não fornece exceções completas, então a causa de cada falha não está estabelecida. Nenhum arquivo nativo de sinal foi produzido.

O CSV final contém o fallback de zeros do notebook, preservado para auditoria. Esses zeros não contam como amostras observadas. Não há medida de precisão Kaggle válida neste grupo. Isso tampouco reproduz ou refuta o score publicado, que usa outra coorte e outro protocolo. O sucesso anterior em uma imagem pública de desenvolvimento continua sendo apenas uma verificação de execução.

## Integridade, reprodução e recursos

`examples/merge_hybrid_inference.py` reconstrói as decisões com a política congelada. Primeiro verifica que as 24 seleções originais são idênticas às salvas. Depois usa os complementos, sela as decisões e só então abre referências para avaliação. Os 96 hashes das predições combinadas foram verificados. As 24 seleções finais são idênticas às leituras diretas, inclusive as ausências.

Um recibo primário de `pm-99-photos_bents`, interrompida antes da retificação, reteve observações geométricas da imagem anterior. Seus bytes foram preservados e a evidência foi colocada em quarentena. Isso não alterou predições ou métricas. A correção de isolamento tem teste adversarial.

Os três pods de inferência foram excluídos e a ausência de cada um foi verificada. O guardião externo primário não precisou executar DELETE. O envelope planejado acumulado foi US$ 0,4437 de tempo de GPU, abaixo dos US$ 0,45 originais. Esse envelope conservador não é uma auditoria de fatura. Armazenamento e arredondamento ficam separados. O volume do treinamento em andamento foi preservado.

Controles independentes corrigiram o encerramento de descendentes após timeout e a instalação do guardião antes de transferir código. Eles não provam a causa de qualquer demora. As informações de etapas recebidas durante a execução primária estavam atrasadas, enquanto os logs completos mostram uma execução normal.

O pacote de medições conserva entradas, referências, detector selecionado, três arquivos brutos de execução, predições, recibos, falhas, relatórios e hashes. Extraia os arquivos brutos em diretórios separados e execute, da raiz do repositório:

```bash
python -m examples.merge_hybrid_inference \
  --primary /dados/primary \
  --repair /dados/repair \
  --specialist /dados/specialist \
  --data /dados/pacote/original-inputs \
  --out /dados/reavaliacao-nova
```

Os caminhos `primary`, `repair` e `specialist` são os diretórios de conteúdo de cada `.tar.gz`. O destino deve ser novo. O script não faz novas inferências. `completion-summary.json` contém resultados compactos e limites. O pacote também tem resultados por imagem, paciente, condição e layout. A chave histórica `cpu_max_side` nos resumos completos é mantida por compatibilidade. Essas execuções usaram CUDA e 3000 pixels.

## Treinamento e próxima hipótese

O primeiro segmento do treinamento original concluiu 13 das 45 épocas do modelo principal, usando a liberação pública completa de 72 shards. O checkpoint foi baixado, verificado e inspecionado com carregamento seguro. Inclui otimizador, scheduler, scaler e estados aleatórios para retomada. O segundo segmento segue em execução. O banco histórico de fontes dos autores está indisponível e sua substituição foi registrada. O treino e a reprodução dos resultados do artigo continuam incompletos.

O controle analítico da normalização original mostrou que uma ocultação de 400 ms deslocou um pico físico de 4,000 s para 3,750 s e eliminou as ausências da saída. Isso demonstra um mecanismo no controle, sem estabelecer a causa de todas as falhas reais. A próxima hipótese é preservar coordenadas temporais e máscaras de visibilidade na extração, seguida de avaliação separada de amplitude e normalização geométrica. O algoritmo original não foi alterado para melhorar esta reprodução.

Os 81 testes de reprodução e 27 contratos de geometria e seleção passaram no commit `c266e20e71049e7bb056cf103ad75689442d328c`. A CI da biblioteca passou em três combinações de Python/Torch. A suíte integrada tem 530 testes e 25 skips explícitos. A galeria de 16 casos permanece idêntica à versão inspecionada.

## Arquivos preservados

- [Pacote completo de medições](https://chatgpt.com/library?file_id=libfile_afaf32e0928081919ce3b158829fa671&account_id=personal), 281 MB, 188 arquivos.
- [Checkpoint original após 13 épocas](https://chatgpt.com/library?file_id=libfile_5d6801b46d0481918fc4e5ad8f12c16a&account_id=personal), 248 MB.

Identidades, tamanhos e SHA256 estão em `measurement-archive.json`.
