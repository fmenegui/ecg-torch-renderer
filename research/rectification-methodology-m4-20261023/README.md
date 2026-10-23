# Rectification M4: methodology and conditional components

This dossier extends the frozen M3 comparison without changing the production
rectifier, the renderer, weights, or M2/M3 outputs. It is a development study,
not evidence of universal or clinical superiority.

The Kaggle census read all 24 public writeups returned by the official forum
listing on 2026-10-23. Only the third-place shared stages were executed natively
in M3. New M4 executions cover conditional geometry, identifiability, sampling,
file export and independent adverse controls. Powerful Medical was not executed.

## Reproduce core checks without external weights

From an intact source checkout with Python, NumPy, SciPy, OpenCV and Pillow:

```bash
python research/rectification-methodology-m4-20261023/replay_components.py --task qa
python research/rectification-methodology-m4-20261023/replay_components.py --task sampling-tests
python research/rectification-methodology-m4-20261023/replay_components.py --task area-tests
python research/rectification-methodology-m4-20261023/replay_components.py --task identifiability
```

The launcher preserves exact original source bytes in a disposable workspace.
It links this checkout for the pinned cell implementation and copies the
published M3 comparator. Receipts are created in that workspace and do not
replace archived evidence. To retain them, supply a new `--workspace` path.
`--task geometry` reruns the 19 analytical scenes and 18-candidate selector.

`tests/test_rectification_m4_components.py` runs the four core tasks during
unittest integration when optional dependencies are present. The independent
QA script includes the retained fixed-quadrature failure as an expected attack,
not as a content-preservation pass. Exact-axis-area is a restricted positive
control under diagonal affine maps and constant source-cell reconstruction.

## Evidence and path conventions

Compact text evidence is committed here. The full measurement package delivered
with the report contains all 76 geometry outputs, all 342 candidates, arrays,
old and corrected guard results, identifiability witnesses, sampling comparisons,
cost checkpoints and failure receipts. Original per-role manifests describe that
full archive, not just this compact GitHub subset.

Study scripts retain their original workspace-relative paths and hashes. The
neural-free core tests above are portable via the launcher. Full sampling parity,
cost and result-audit scripts also need the archived M3 sampler, ROI arrays and
result context in their original `m3/sampling` paths. These are input artifacts,
not regenerated expected fixtures. The report/figure builders target the full
study workspace. No third-party source cache, checkpoint or article is included.

## Read first

- `integration/methodological-decisions-pt.md`: choices, competing alternatives,
  implementation status and external benchmark gate.
- `integration/decision-contract.json`: distinct unit, orientation, support,
  visibility and image-preservation evidence.
- `qa/final-review.md`: independent finite approval, corrected topology defect,
  retained failures and gaps.
- `geometry/analysis.md`, `sampling/README.md`, `identifiability/notes.md`:
  exact conditional claims and source identities.

The 48-family matrix is finite. It distinguishes executed, partial, constructive
and unexecuted evidence. Counts of tests, candidates or pixel queries are not
additional patients. Origin-gauge controls and raw component scores must not be
confused with M3 translation-aligned metrics. CPU resource measurements use a
shared host. Sparse field-fit cost is not full-selector or image latency.
