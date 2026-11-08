# Independent NumPy compatibility follow-up

Date: 2026-10-23. Scope: the single 2D cross call in `guard_complex`.

**Decision: approve the narrow compatibility repair.** The previous methodological limitations, negative results and conditional approval remain in force. The repair does not add a method, tune a candidate or improve a physical score.

The historical guard-v2 identity is `0ce6062797c5e25dd9ce3d7b26a6a9fabe7779ca72b5ea4e55fcda564bed9e87`. The compatible identity is `421ebe98bdd9b5c18daf039837297d8592e9fdb4f937b466311033a9f6be7fe0`.

The complete literal source diff is exactly:

```diff
-    cross = np.cross(vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0])
+    edge_a = vertices[:, 1]-vertices[:, 0]
+    edge_b = vertices[:, 2]-vertices[:, 0]
+    cross = edge_a[:, 0]*edge_b[:, 1]-edge_a[:, 1]*edge_b[:, 0]
```

For two edge vectors a and b this is their signed 2D determinant, the same scalar previously returned by the 2D cross. Both edge differences and the multiply/subtract ordering remain explicit. Guard tolerances and all other file bytes are unchanged. The historical source and historical approval receipts are preserved, rather than retrospectively relabelled as the new source.

| Independent check | Result |
|---|---|
| Entire source equals the old source with only the declared substitution | Pass |
| Actual historical NumPy 2.3.5 cross reference | 8,204 float64 triangles saved |
| New assignments versus that reference in NumPy 2.3.5 | Bit-identical |
| New assignments versus that reference in actual NumPy 2.5.4 | Bit-identical |
| Existing 26 independent methods on Python 3.12.14, NumPy 2.3.5 | 26 pass |
| Existing 26 independent methods on Python 3.12.14, isolated NumPy 2.5.4 | 26 pass |
| Previous approval receipts and sources stable during tests | Pass |
| Both 19-scene reruns versus independently audited guard-v2 | 3,096 checks pass |
| Arrays compared across both reruns | 1,444 byte-identical arrays |
| Per-rerun selections, candidates and complete score records | Same 19 selections, 342 candidates, 76 records |

The numeric reference includes broad finite exponents, near-degenerate orientation thresholds, exact degeneracy, signed zero and subnormal products. Finite equivalence controls complement the direct formula/source review. They do not claim exhaustive verification of every IEEE754 input. The current physical negative controls, including the oblique crease, coherent alias and zero-coverage rejection, are unchanged.

No NumPy API shim, monkeypatch of `np.cross`, dependency pin, downgrade or principal-environment modification was used. The existing tests still use their original PNG export-failure injections. The actual target installation is `/workspace/scratch/a57599a3e70e/m4-numpy25-runtime`, selected through `PYTHONPATH`. Remote CI across its Python versions remains an integration check separate from this local Python 3.12.14 evidence.

Receipts are in `receipts/numpy-compat/numpy235.json`, `numpy254.json`, `results-parity-audit.json`, `historical-cross-reference.json` and `prior-approval-bindings.json`. The compressed historical reference is retained beside them. `review_numpy_compatibility.py` and `audit_numpy_compatibility_results.py` reproduce this review without modifying M2, M3 or the test implementations.
