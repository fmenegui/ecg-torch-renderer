# M3: methodological audit and native rectification comparison

This study compares the existing B0 and M2 rectifiers with the actual published
Open-ECG-Digitizer rectification stage and the public third-place Kaggle stage0/1
pipeline. It separately audits the publicly documented Dotter/Gridder mechanism.
No production rectification defaults or M2 source files change.

The methodological review is in [methodological-decisions-pt.md](methodological-decisions-pt.md).
The frozen assessment contract is [protocol.json](protocol.json). Every case,
query archive, source, checkpoint, configuration and runtime version is bound by
[execution-lock.json](execution-lock.json).

## Completed diagnostic results

The 160 scheduled case/configuration outputs are complete and independently
audited. See [results-interpretation-pt.md](results-interpretation-pt.md),
[results-summary.json](results-summary.json) and [native-cases.csv](native-cases.csv).
The primary paper-domain results preserve native orientation/scale and remove
only the predeclared global translation:

| Configuration | G correct | W wrong | C coverage | R = W/C | RGB emitted |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 55.02% | 15.95% | 70.97% | 22.48% | 18/32 |
| M2 | 55.27% | 15.66% | 70.94% | 22.08% | 18/32 |
| npj Generic | 10.67% | 77.02% | 87.69% | 87.84% | 32/32 |
| npj George Moody 2024 | 10.63% | 77.06% | 87.69% | 87.88% | 32/32 |
| Kaggle third-place | 0% | 0% | 0% | undefined | 3/32 |

Rounded percentages need not sum exactly. These are already-inspected synthetic
format-stress data. The predeclared Kaggle-compatible stratum has only one
capture/patient/backend and cannot assess its original supported task. Its three
emitted rasters have no observed grid nodes, all 2,508 nodes filled with zero
coordinates and 2,408 degenerate cells each. Emission is not correct geometry.
The two configurations of each architecture are correlated operating choices,
not independent replications. Different trained frontends remain confounded in
the native lane; matched-observation component lanes isolate geometric choices.

M2's gain over B0 is 0.252 percentage points of G, with a descriptive paired
patient-bootstrap interval [0, 0.690] pp. D4 diagnostic help raises M2 G to
68.17%, showing a much larger unresolved orientation issue. In the actual-RGB
full-field endpoint M2 G is 46.93%, W 9.32%, C 56.25% and R 16.57%.
Partial coordinates are useful but the 14 original-preserved cases are not
successful full-image rectifications.

Independent QA passed 50,223 saved-result checks and 821 post-summary checks,
including the 20,000 paired bootstrap draws. All 131 emitted/retained selected
PNGs decode fully after one disclosed fallback-file recovery from a verified
native duplicate. The truncated bytes and recovery receipt are retained in the
measurement archive. No inference, metrics, times, statuses or frozen files
changed. There were no wall or standalone-pipeline budget overruns.

These results do not establish external, clinical, leaderboard or commercial
superiority. The supplied archive retains every failure, full precision,
predictions, queries and native raster outputs.

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
6. The actual bilinear sampler at 4 px/mm can erase 0.1 mm strokes despite
   exact affine geometry and no confidence-mask erasure. A separate 280-scene
   sampling probe retains 1,960 measurements and all negative controls.
   Area integration removes the four native thin-strip disappearances in the
   140 declared thin-strip controls, but costs more and does not improve every
   endpoint monotonically. This is a component prototype, not a default change.

See [sampling/README.md](sampling/README.md) for the explicit synthetic RGB
formation/reconstruction contract, quality/cost table and limitations. The
independent audit passed 32,558 checks. Source quantization ties of one uint8
level and the first run's unbounded BLAS timing were retained and disclosed.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python research/rectification-comparison-m3-20261023/sampling/sampling_component.py \
  --geometry-dir ecg_gpu/digitization --out build/m3-sampling
```

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
