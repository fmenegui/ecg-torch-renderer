# Ensaio prospectivo em GPU e conclusão das etapas interrompidas

A execução 41106791069 usou uma A40 real com Torch 2.7.0/CUDA 12.8, 3000 pixels,
os pesos originais e a política previamente congelada. São 24 fotografias de
oito pacientes fora dos nossos grupos anteriores. O compartilhamento com dados
de treinamento dos autores permanece desconhecido. Todas as imagens continuam
no denominador, incluindo as que excederam o prazo.

| Todas as 24 imagens, média por paciente | Cobertura impressa | NRMSE bruto |
| --- | ---: | ---: |
| npj direto | 81,43% | 1,1276 |
| Grade e RGB adaptativo seguidos do npj | 64,72% | 0,9178 |
| Seleção conservadora | 81,43% | 1,1276 |

O retificador reduz o erro agregado, mas perde suporte. O seletor não aceitou
candidatas: 16 decisões citaram extrapolação geométrica, duas identidade de layout
e seis ausência de candidata aceita. Isso não demonstra uma biblioteca melhor.
A inferência direta retornou em 23 imagens. A retificada retornou em 18, com
cinco rejeições de malha e uma interrupção por prazo. A última imagem também
ficou sem leitura direta por prazo.

A solução Kaggle tinha seis imagens elegíveis pelo layout inferido e congelado,
mas seu batch não começou por falta de tempo. Seus zeros de cobertura e NRMSE
igual a 1 representam ausência de predição, e não medem a precisão do algoritmo.
Os 18 casos fora do escopo também permanecem explícitos.

A continuação executa somente essa imagem interrompida e o especialista nos seis
casos já elegíveis. Se a leitura reparada inferir o layout elegível, ela também
entra no batch conforme a regra original. Modelos, pesos, escala e política são
mantidos. Nenhuma referência é adquirida pelo worker. O ensaio original e suas
falhas são preservados. A conclusão posterior é reportada separadamente, depois
da abertura dos resultados primários, sem alegar um segundo teste independente.

O teto planejado de GPU da continuação é US$ 0,17, dentro dos US$ 0,45 originais
quando somado ao limite conservador de tempo e preço dos recursos já encerrados.
Esse cálculo não é uma auditoria da fatura. Armazenamento temporário fica separado.

A auditoria encontrou um recibo com observações da imagem anterior em
`pm-99-photos_bents`, que não chegou à retificação por prazo. Essas observações
não são usadas como evidência da imagem atual. Seus hashes e o arquivo original
foram conservados. A correção de isolamento já tem teste adversarial.

A ausência do pod foi verificada pelo controlador e por uma leitura externa.
A leitura externa não precisou executar DELETE. Os detalhes de etapas recebidos
durante a execução estavam atrasados e não provaram travamento do pod.

Os controles adicionais de infraestrutura demonstraram duas lacunas independentes:
um descendente podia continuar após o timeout de seu processo pai, e uma falha
na cópia de código podia ocorrer antes de armar o guardião. O controlador agora
encerra o grupo de processos e arma o guardião antes das transferências. Esses
controles não demonstram a causa de qualquer demora do ensaio concluído.


A execução de continuação `41113820943` concluiu a leitura direta e retificada
que faltava. O notebook Kaggle iniciou, carregou o checkpoint da primeira etapa
e encontrou rotação desconhecida. O código nativo registrou a falha, mas sua
demonstração gráfica depois do tratamento de erro esperou cinco campos onde
havia quatro e encerrou o batch antes de processar os outros casos.

Foram revisados os três blocos de demonstração gráfica do notebook v6. Eles
apenas desenham e exibem intermediários, inclusive intermediários indefinidos
quando uma etapa falha. A adaptação agora omite somente esses blocos, reconhecidos
por seus hashes de AST. Cálculos numéricos, checkpoints e listas nativas de falhas
são conservados. Um bloco diferente é recusado. O erro original permanece no
arquivo da primeira continuação.

A nova invocação executa apenas as seis imagens elegíveis e reutiliza a leitura
reparada. Seu limite é 660 segundos e US$ 0,11 de tempo de GPU. Somado ao limite
conservador de US$ 0,3337 para os recursos já encerrados, o envelope fica em
US$ 0,4437. Não são iniciados novos treinos nem repetida a imagem já reparada.
Falhas nativas de normalização permanecem falhas, sem preenchimento como sinal.
