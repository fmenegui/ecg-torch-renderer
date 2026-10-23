Sampling/export M4 está concluído e as fontes aprovadas estão congeladas.

- Paridade: 280 cenas M3, seis variantes, dois orçamentos de bloco, 3.360 saídas, 16.944.480 valores RGB, zero diferença de uint8. Abrange todos os pixels do ROI41×41 arquivado, não o quadro inteiro.
- Memória: coordenadas explícitas limitadas ao bloco, no máximo917504 bytes no teste16384 pixels, independentemente de n=2/4/8. Fonte e saídas continuam ocupando memória proporcional às imagens.
- Custo negativo mantido: checkpoint final1024×768 deu cache0,171/0,623/2,185s e streaming0,287/1,066/3,813s. RSS cache173/337/941MB, streaming81MB. AMD EPYC9V74, OpenCV/BLAS/OMP/MKL1thread. Host compartilhado. As evidências adicionais calculadas pelo streaming também custam.
- RGB e confiança são independentes. Não finitos recebem fundo por máscara explícita. RGB fora da fonte usa fundo constante. RGBA usa premultiplicação, fundo transparente e alpha separado de confiança.
- PNG: temporário único, fsync, CRC de todos chunks, IEND, decode completo e igualdade de cada pixel antes da publicação. Destino existente é recusado por padrão. Substituição só com SHA256 esperado e lock cooperativo. Escritor externo que ignore o lock ainda pode correr contra substituição explícita.
- Contraexemplo mantido: minificação32:1 da faixa fonte1px retorna branco em2×2/4×4/8×8. Concordância entre grades não certifica suficiência.
- Controle positivo separado e dirigido por essa falha: área da reconstrução constante por célula fonte com inversa afim diagonal retorna247,03125→247 no caso32:1. Família60casos: zero desaparecimentos contra29/17/9 em2/4/8. Não é holdout fresco, modelo bilinear, área radiométrica ou solução geral. Qualquer shear/rotação é rejeitado.
- Testes finais:19 sampling/export +6 área =25 aprovados. O primeiro erro real de cancelamento alpha ficou preservado em axis-tests-initial.log. QA independente revisou separadamente os componentes.
- Próxima decisão: footprint/Jacobiano e integração/filtro anisotrópico com critério explícito de erro e custo, mantendo geometria e retenção separadas. Não promover fixed8 como universal.

Fontes finais:

| Arquivo | SHA256 |
| --- | --- |
| streaming_sampler.py | b9a7a732533090271455d42fdc3b4784bf299e14f0b4fc82c564fadccc31f503 |
| pngio.py | 816c319d8bdd47ef82aff7057a4461d7b2b3249ce95d60a7a7fa3b93b3fd38b1 |
| axis_aligned_area.py | d7b347d7733d91530babf518ab2987c2558b4aebed3b4f37773f6b8fee49cc26 |

O manifesto public-manifest.json lista todos os arquivos que podem ser copiados ao dossiê, com tamanho e hash. Todos são deste diretório. Os componentes e os testes são independentes de M3. Os scripts evaluate.py e benchmark.py leem intencionalmente o baseline e os NPZ M3 nos caminhos de pesquisa originais, conforme README. Se forem movidos, o launcher precisa fornecer esse contexto ou ajustar os caminhos de replay, sem alterar os protocolos nem as fontes aprovadas. O primeiro código executado no replay está preservado em results/streaming_sampler_at_first_run.py, com distinção explícita em receipt.json. As alterações posteriores foram só validação de argumentos, sem mudança do caminho numérico pontuado.
