# M3: methodological audit and native rectification comparison

This study compares the existing B0 and M2 rectifiers with the actual published
Open-ECG-Digitizer rectification stage and the public third-place Kaggle stage0/1
pipeline. It separately audits the publicly documented Dotter/Gridder mechanism.
No production rectification defaults or M2 source files change.

The methodological review is in [methodological-decisions-pt.md](methodological-decisions-pt.md).
The frozen assessment contract is [protocol.json](protocol.json). Every case,
query archive, source, checkpoint, configuration and runtime version is bound by
[execution-lock.json](execution-lock.json).

## Scope and comparisons

- Primary diagnostic panel: 32 existing M2 captures, eight patient groups,
  four views per patient, balanced across two render backends. These synthetic
  captures were inspected previously. This is not a new unseen holdout.
- Development feasibility: one predeclared capture, kept outside diagnostic
  aggregates. The remaining 23 development captures are not a completed panel.
- Five native arms: `our-B0`, `our-M2`, `npj-generic`,
  `npj-george-moody-2024`, `kaggle-third`. No preset is chosen per image using
  reference errors. The Kaggle arm represents Tang's third-place stage0/1
  adaptation of Heng, not every Kaggle solution or the competition winner.
- Commercial Dotter/Gridder is **not executed**. Its documented API requires
  authorized access and does not expose the geometric intermediate outputs used
  by this comparison. Publicly documented geometry and our primitive controls
  must not be labelled commercial implementation results.
- Analytical component studies retain all 10 scenes × 3 interpolation arms
  and all 12 scenes × 5 composition arms, including negative controls.

Primary scoring permits only one translation fitted on 64 disjoint frozen
anchors. It preserves native scale and orientation. The separate D4 diagnostic
does not establish automatic orientation. Neither track fits scale or shear.
Paper is the primary fixed query domain, including headers and gridless margins.
Grid is secondary. The 0.5 mm threshold is an engineering endpoint.

Correct coverage G, wrong coverage W, total coverage C and risk R=W/C must be
read together. R is undefined at C=0. Claimed-eligible nonfinite outputs count as
wrong. Failed scheduled cases remain in the denominator. Native trusted
coordinates, unrestricted finite field and actual RGB resampling are distinct
tracks. An original-preserved fallback cannot count as successful rectification.
Inverse-valid geometry alone does not certify waveform or text preservation.

## Methodological findings

1. A coherent 10 mm/5 mm pitch alias passes local cycle checks and fixed-ID
   refinement. Better interpolation does not repair incorrect identities.
2. Selecting the largest lattice component loses half the observed points in a
   controlled disconnected-grid example. This is lost coordinate evidence, not
   proof that the original RGB is erased.
3. Affine triangles have projective interpolation bias even with exact corners.
   Independent per-cell homographies remove planar bias but can disagree along
   shared edges on curved surfaces. Both interior accuracy and continuity matter.
4. The separate `T ∘ H` prototype retains shared continuous edges and planar
   projective accuracy. Its noise and coherent-corruption regressions prevent
   unconditional adoption. Its tests have no CNN, RGB or clinical interpretation.
5. Orientation semantics, scale, support provenance and actual sampling status
   need independent contracts. A smooth invertible map is not necessarily a
   physically correct map.

## Reproduction

Use the already locked M2 bank and `joint-1973.pt` checkpoint, and retrieve the
external source/checkpoints specified in `npj-availability.json` and
`kaggle-availability.json`. Those external checkpoints, source trees and PDFs are
not redistributed here. The original Kaggle reference grid asset is required.

Runtime used: CPU, Torch 2.8.0+cpu, torchvision 0.23.0+cpu, two Torch threads,
one OpenCV thread, serial inference. Other package versions are in the lock.
The 180 second deadline uses cooperative SIGALRM. Native calls may defer handling,
so observed elapsed time and overruns are retained. These are shared-host CPU
measurements, not GPU performance claims.

Set the paths below to the local copies, then create a **new** execution lock
for that checkout. The recorded historical lock intentionally binds the original
paths and bytes and must not be edited to make a different run appear identical.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python examples/compare_rectification_methods.py freeze \
  --bank build/nextgen-bank \
  --checkpoint build/study-inputs/joint-1973.pt \
  --metric-core examples/rectification_comparison_metrics.py \
  --npj-adapter examples/npj_rectification_adapter.py \
  --npj-repo /path/to/Open-ECG-Digitizer \
  --kaggle-adapter examples/kaggle_rectification_adapter.py \
  --kaggle-repo /path/to/physionet-ecg-image-digitization-3rd-place \
  --kaggle-assets /path/to/authentic-dataset \
  --protocol research/rectification-comparison-m3-20261023/protocol.json \
  --schedule research/rectification-comparison-m3-20261023/schedule.json \
  --summarizer research/rectification-comparison-m3-20261023/summarize_m3.py \
  --out build/m3-reproduction
```

The schedule expects its prepared `queries/` alongside it. Copy the archived
query NPZ files there before freezing. Run the same invocation with `run` in
place of `freeze` and `--split test`, then:

```bash
python research/rectification-comparison-m3-20261023/summarize_m3.py \
  --root build/m3-reproduction --out build/m3-reproduction-summary \
  --metric-core examples/rectification_comparison_metrics.py --split test
python -m unittest discover -s tests -p 'test_rectification_comparison*.py' -v
```

The summary refuses incomplete selected panels or changed locks/predictions.
The inference runner imports only original RGB into each native model. Gold
arrays enter only the separate evaluator. Our B0/M2 share fresh checkpoint
inference, never cached historical timing. The reported standalone estimate
includes that detector time for each arm.

For the component controls, pass the existing M2 geometry directory explicitly:

```bash
python research/rectification-comparison-m3-20261023/structural_controls.py \
  --geometry-dir ecg_gpu/digitization --out build/m3-structural.json
python research/rectification-comparison-m3-20261023/composition_probe.py --help
python research/rectification-comparison-m3-20261023/component_control.py --help
```

Upstream OpenCV/NumPy random state was not freshly seeded uniformly for every
competitor call in the historical run. Preserve execution order when replaying.
No bitwise equivalence across resumed/reordered executions is claimed.

## Attribution

`examples/npj_rectification_adapter.py` instruments
Ahus-AIM/Open-ECG-Digitizer at commit
`97a15087d4abcda843da8c58ee74b1d8f47e6f9a`, exposing coordinate transforms
and omitting downstream lead-layout/waveform modules. That file remains under
CC-BY-SA-4.0. The full license is [CC-BY-SA-4.0.txt](CC-BY-SA-4.0.txt).

The Kaggle adapter calls Tang/Heng's externally retrieved implementations and
weights, pinned in its source. It does not redistribute those assets. The
checkpoint dataset's license metadata is Unknown. See the availability notes.

This comparison does not establish superiority over clinical data, the Kaggle
leaderboard, the npj end-to-end waveform endpoint, or the commercial PMcardio
service. Generalization requires a new independent real-image assessment.
