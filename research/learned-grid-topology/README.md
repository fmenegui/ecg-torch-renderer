# Learned subpixel detector and relative lattice prototype

This first implemented and trained automatic rectifier is not better than the
published baselines. Native-distribution point detection and topology work on
online test images, but physical-render transfer and complete waveform extraction
fail. The negative results and false topology acceptance are part of the result.

## Implemented boundary

`ecg_gpu/digitization/lattice.py` takes unordered detected pixel centers and
confidence scores, estimates dominant axes/spacing, builds mutual axial-neighbor
edges, infers relative integer lattice coordinates and checks cycle conflicts,
unique node ownership and local convex cell orientation. It never receives
source grid IDs, canonical vertices, lead names or signal values. It retains the
largest component and returns explicit rejection, conflicts, support and missing
cells. Absolute origin is not recovered. Quarter-turn orientation remains
ambiguous. Local checks do not establish global injectivity or correct major-grid
identity. Confidence scores are not calibrated error probabilities.

Accepted cells are remapped from the original RGB at a declared 8 pixels/mm,
assuming detected points represent major 5 mm intersections. Invalid and absent
cells stay white/unsupported. Both printed labels and waveform use this single
RGB warp. There is no learned dense field or explicit crease head in this first
prototype. The geometry is limited to moderate perspective and approximate
alignment. A frozen npj extractor tests whether rectified output is useful.

## Training and source-separated point evaluation

The small existing GridUNet is trained from a fixed initialization for 1,000
AdamW steps on CPU, using full-resolution heatmaps and bounded x/y offset heads.
There are 96 verified cached captures from 24 PTB-XL patients: 16 train, four
validation and four test patients, two views and two rendering backends each.
The equal-budget mixed regime uses 32 of the 64 training captures, selecting one
backend for each source/view. There are 4,000 sampled training crop examples.
Training takes 164.38 seconds in the recorded environment. Model threshold 0.85
is chosen only on the 16 validation captures, never on test or transfer results.
All cache file hashes, source IDs, patient groups and signal split ownership are
checked. Source provenance and the complete capture manifest accompany this note.

| Backend | Test images / patients | Precision | Recall | F1 | Median matched error | P95 matched error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Online torch renderer | 8 / 4 | 0.9529 | 0.9296 | 0.9411 | 0.385 px | 0.864 px |
| Offline physical renderer | 8 / 4 | 0.3664 | 0.1606 | 0.2233 | 0.949 px | 1.439 px |

Points are matched one-to-one at radius 1.5 pixels on the native 512x384 images.
Localization errors describe matched points only. Multiple views share patients
and are not independent clinical observations. These are synthetic captures of
real PTB-XL waveforms, not real photographs of ECG paper.

All eight online test lattices pass local checks. Their matched selected nodes
have 100% relative identity agreement, modulo origin and square-lattice symmetry.
That statistic is conditional on selected matched nodes, not complete-grid
recovery. Seven offline lattices are rejected. The eighth passes local checks
but has only 52.9% relative identity agreement. This false acceptance demonstrates
that local topology constraints are insufficient as a quality certificate.

## Image-to-signal transfer, unchanged model and thresholds

Four earlier diagnostic images use the artificial source signal, not the PTB-XL
training waveforms. They are development probes, not a new clinical test cohort.
Original 2048x1280 RGB is letterboxed to the detector size with pixel-center
coordinate inversion. This acquisition/scale distribution differs from the
native source test distribution, which explains why native F1 alone is inadequate.

| Diagnostic input | Geometry outcome | Supported output pixels | Complete signal outcome |
| --- | --- | ---: | --- |
| Standard 3x4 + rhythm | Partial warp | 13.20% | npj layout failure |
| Six rows | Partial warp | 13.19% | npj fallback/empty/inconsistent output rejected |
| Physical flat | Two supported cells | Not warped | Explicit geometry rejection |
| Physical fold | Cycle conflicts | Not warped | Explicit geometry rejection |

Complete usable signal recovery is 0/4 on these probes. A diagnostic score on
0.93% printed coverage for the failed standard case is not a successful result.
The partial images were visually inspected and show severe fragmentation.
No arbitrary-layout robustness, clinical validity or superiority is claimed.

The preceding oracle experiment shows a large recoverable geometry gap. This
prototype shows that point detection accuracy on familiar rendering conditions
alone does not close it. Next work should add scale/appearance diversity in the
training captures, improve offline grid detection, and assess component linking
and major-grid identity before densifying or interpolating across gaps. Keep
these development probes separate from a new confirmatory source/capture split.
Do not relax cell or conflict gates merely to improve their measured scores.

## Reproduce

Prepare the verified source cohort and cached renders following
`research/grid-benchmark-v0120/README.md`, then run:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python examples/train_grid_rectifier.py \
  --cache build/grid-benchmark-pilot \
  --out build/learned-grid-topology --steps 1000

python examples/evaluate_learned_grid.py \
  --checkpoint build/learned-grid-topology/model.pt \
  --cache build/grid-benchmark-pilot \
  --inputs build/paper-adaptation-inputs \
  --npj-source build/upstream/open-ecg-digitizer \
  --out build/learned-grid-evaluation
```

The exact small checkpoint is versioned as base64 text because the connected
repository writer supports text files. Decode it with
`base64 --decode research/learned-grid-topology/checkpoint.pt.b64 > model.pt`,
then verify SHA-256
`959506fbd414a1cfeb0d3bb28a604c7678570a796e6dce5c206dda8e4a2b2b97`.
The evaluator loads tensor-only checkpoint data with `weights_only=True`.

Five new geometry tests cover shuffled/rotated points, local deformation,
missing intersections, degeneracy and restricting warps to accepted cells.
The final 61-test digitization suite passes. Full discovery runs 357 tests:
349 pass, six skip and two error on the pre-existing missing original raster
fixture. It is not regenerated. E/F lint and diff checks pass. No renderer
source changes, CUDA execution or new paid GPU resources are involved.
