# Compatibilidade NumPy 2.5, sem mudança metodológica

A CI detectou uma incompatibilidade de API no guard-v2. O NumPy 2.5 removeu o suporte de `numpy.cross` para vetores de duas dimensões, conforme as [notas oficiais de versão](https://numpy.org/devdocs/release/2.5.0-notes.html#expired-deprecations). A versão aprovada usava essa operação para calcular a orientação de triângulos.

A única mudança na fonte geométrica substitui o produto 2D por sua expressão escalar:

```python
edge_a = vertices[:, 1]-vertices[:, 0]
edge_b = vertices[:, 2]-vertices[:, 0]
cross = edge_a[:, 0]*edge_b[:, 1]-edge_a[:, 1]*edge_b[:, 0]
```

Não houve alteração de candidatos, hiperparâmetros, loss, folds, seleção, tolerâncias ou semântica das guardas. Não houve monkeypatch, pin ou downgrade de dependência como solução. O arquivo `requirements.txt` preexistente permaneceu inalterado e não foi usado para contornar a incompatibilidade. Nenhuma fonte M2/M3 foi modificada.

A fonte guard-v2 aprovada continua preservada em `geometry_selector_guard_v2_historical.py`, SHA256 `0ce6062797c5e25dd9ce3d7b26a6a9fabe7779ca72b5ea4e55fcda564bed9e87`. A fonte atual tem SHA256 `421ebe98bdd9b5c18daf039837297d8592e9fdb4f937b466311033a9f6be7fe0`. A revisão foi registrada antes dos retestes em `numpy-compatibility-revision.json`, SHA256 `0eee6ca9b6b91303e54911491957391e1b48b167c604bcf76d6817a852013fc4`. O verificador atesta que a diferença da fonte é exatamente a substituição acima.

O mesmo experimento foi repetido integralmente em Python 3.12.14 nos dois ambientes. O NumPy 2.5.4 foi carregado de uma instalação isolada por `PYTHONPATH`, sem mudar o ambiente principal.

| Execução | Diretório | Verificações | Seleções idênticas | Arrays de candidatos idênticos | Arrays totais idênticos | Registros primários de scores idênticos |
|---|---|---:|---:|---:|---:|---:|
| NumPy 2.3.5 | `results-numpy-compatible` | 3.230 | 19/19 | 342/342 | 722/722 | 76/76 |
| NumPy 2.5.4 real | `results-numpy254` | 3.230 | 19/19 | 342/342 | 722/722 | 76/76 |

A comparação com `results-guard-v2` foi exata, incluindo NaNs e máscaras, sem afrouxar tolerâncias. Não houve diferença numérica de coordenadas ou scores. Os dois recibos são `numpy-compatibility-retest.json` e `numpy254-compatibility-retest.json`. A referência física continuou excluída da seleção, e os novos selos de decisão permaneceram intactos depois dos scores.

A busca sintática no caminho atual não encontrou outro uso de `np.cross`: `geometry_selector.py`, o adaptador esparso, o runner, os verificadores atuais, a composição M3 importada e `cell_mesh.py` M2 importado. Snapshots históricos conservam deliberadamente a chamada antiga como evidência e não integram o caminho atual de execução. A verificação independente de QA e a matriz externa de CI são evidências separadas dos dois experimentos locais descritos aqui.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 swarm-venv/bin/python m4/geometry/run_experiment.py --out m4/geometry/results-numpy-compatible
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 swarm-venv/bin/python m4/geometry/check_numpy_compatibility.py

PYTHONPATH=/workspace/scratch/a57599a3e70e/m4-numpy25-runtime OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 swarm-venv/bin/python m4/geometry/run_experiment.py --out m4/geometry/results-numpy254
PYTHONPATH=/workspace/scratch/a57599a3e70e/m4-numpy25-runtime OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 swarm-venv/bin/python m4/geometry/check_numpy_compatibility.py --after-dir m4/geometry/results-numpy254 --receipt m4/geometry/numpy254-compatibility-retest.json
```
