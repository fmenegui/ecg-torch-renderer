# Image-only hybrid rectification study

Status: proposed larger experiment, partial implementation and CPU preview.
No new GPU training has run. Do not present this directory as a completed GPU
comparison or evidence of superiority over Kaggle, npj or Powerful Medical.

## Scientific target

Recover the flat RGB image from a photograph of a gridded ECG, without relying
on lead layout, supplied generator coordinates, waveform extraction or a
diagnostic classifier. A regular-looking grid is insufficient if the method
cuts away the trace, swaps rows or maps overlapping regions onto each other.

## Proposed combination

1. Estimate coarse orientation and perspective from the page and grid. Use
   label-based anchors as an optional baseline, never a required lead layout.
2. Detect major-grid crossings with local appearance and subpixel offsets.
   Preserve both raw information and locally normalized contrast. Compare raw
   and contrast variants rather than assuming contrast always wins.
3. Associate the detected crossings using local directions, mutual neighbors,
   integer relative identities, collision checks and cycle consistency.
4. Fit a robust grid. Compare no smoothing with confidence-weighted curvature
   regularization. A future crease-aware version should reduce regularization
   across genuine folds. A globally smooth TPS is a separate control, not a
   mandatory extrapolator across occluded paper.
5. Apply the inverse geometric map to the original RGB in one final sampling
   operation. The trace, labels and grid must undergo the same transformation.
6. Keep a support mask and reject ambiguous regions. Never use ground truth to
   choose an inference transform. A failed or cropped output stays in the
   denominator, and local convexity is not a global injectivity certificate.

This shares the Dotter / Gridder / per-cell undistortion principle described
by Powerful Medical. It is not a claim to have invented this family of methods.
The Kaggle reference is Heng stage0/1 as used by Tang's third-place solution.
Its high-resolution mapping is useful, while its lead-label initialization is
a potential layout dependence to measure directly.
The npj reference contributes a grid-segmentation, graph optimization and TPS
control. Its published nonlinear path normally warps signal probabilities.
Our explicit adapter applies that same fitted warp to RGB for visual inspection.

The implemented prototype currently covers point detection, relative topology,
robust second-difference refinement and supported per-cell RGB resampling.
It does not yet implement a learned coarse orientation head, high-resolution
patch detector, crease-aware loss or a complete comparison statistic pipeline.

## Larger cohort, selected before training

The delivered data artifact's `selected-cohort.json` identifies 1,280 unique PTB-XL patients from metadata
1.0.3. There are 1,024 training, 128 validation and 128 test patients. Folds
1–8, 9 and 10 determine the split. One original 12-lead recording per patient.
Seventy-seven previous or physical-reference patients were excluded before
sampling. Original ADC-to-mV conversion only, no invented waveforms/diagnoses.

Planned captures: three views through each of two backends, 7,680 images in
total, compared with 192 in the previous study. Online torch captures and
offline Mitsuba CUDA captures share geometric supervision but not renderer
appearance. Offline physical captures use 128 samples per pixel. Layouts
alternate 3x4+r1, 6x2 and 12x1. Capture geometry families still overlap across
splits, so this alone is not a held-out-camera benchmark.

**Metadata selection is complete. Signal download is incomplete.** Do not
interpret the selected IDs or cached partial WFDB files as an available trained
dataset. PhysioNet returned HTTP 403 for some requested signal files. The
local downloader was interrupted and produced no final 1,280-record signals.npz.
No failed acquisition is silently substituted with a different patient.

The train driver reads cached captures in bounded batches rather than keeping
the full image and target bank on GPU. Geometric targets are rasterized on the
device. The fixed proposed budget is 20,000 AdamW steps per raw/contrast
variant, each with seeds 1973 and 20261021. Validation selects the threshold
from the previously declared grid and the model by backend-macro F1. Synthetic
test and real captures do not enter training or threshold selection.
Training stores unselected recovery weights every 1,000 steps. Recovery weights
are not deployable selected checkpoints. Exact optimizer/RNG restart is not
implemented, so recovery currently means preserving weights rather than a
bitwise continuation guarantee.

## Comparison protocol to finish before test evaluation

Run identical source RGB inputs through these independent arms:

| Arm | Purpose |
| --- | --- |
| Previous frozen point detector | Existing baseline |
| Larger raw detector with strict cells | Data-volume and contrast control |
| Larger contrast detector with strict cells | Matched appearance ablation |
| Same selected detector with robust smoothing | Geometric fit ablation |
| Heng stage0/1 | Published Kaggle rectification |
| npj perspective only | Published projective control |
| npj perspective plus graph/TPS | Nonlinear reference |
| Known-grid remap | Geometric oracle, not a learned method |

Compare native methods first. An optional Kaggle stage1 experiment with supplied
orientation must be labeled as a different, assisted control. A stage0 failure
is a native pipeline failure, not proof that stage1 cannot fit the grid.
The current reference adapter executes native Kaggle stage0/1 and nonlinear
npj only. The perspective-only arm and fair common-coordinate quantitative
comparison still need implementation and verification before the larger run.

Required metrics: point precision/recall/F1 at the declared image scale,
conditional localization error, relative-ID agreement, retained source-page
area, accepted cells, foldovers and rectified geometric error in millimeters.
Use a similarity alignment only for evaluation to remove arbitrary output
origin, isotropic scale and lattice-axis symmetry. Do not fit a general affine
transform that could erase aspect-ratio or shear errors. Pair errors on common
source support, show each method's all-case coverage and retain every failure.
Cluster uncertainty intervals and paired deltas by patient, with training seeds
reported separately. Avoid treating photographs of the same print as patients.

Real images are external checks only. Available physical scans/photographs of
four independent printed ECG records provide 12 checks. Their source waveforms
are real, but this is not a cohort of original clinical paper photographs.
No manually verified grid annotation is available for these 12 images, so
prediction counts and topology acceptance are not accuracy estimates. Public
reference-model training overlap is unknown. A future clinical-image subset
with blinded manual grid annotations remains necessary.
The previous model accepted topology in 10 of these 12 captures and rejected
two smartphone photographs. This does not establish correct relative IDs.

## Execution actually completed

Runpod MCP accepted initialization but `list-pods` returned HTTP-style status
500 with `detail: authentication failed` and an upstream failure hint. The
instructed single retry returned the same failure. No Pod or volume was created.
There is no configured Runpod CLI/API-key fallback and local CUDA is unavailable.
Restoring the Runpod account connection is required for the requested GPU run.

The existing selected contrast checkpoint, SHA-256
`e09218759a52bdf7324f88d7ef7615820f7d06908e9c4000ab93c1e50e57a54d`,
was evaluated on the 32 old synthetic test images using CPU. No training ran.
All 32 cases appear in the gallery. With fixed smoothing 0.05, 31/32 cases are
accepted in both arms. Mean all-case support of visible true grid nodes is
92.03% without smoothing and 92.13% with smoothing. Median per-case conditional
p95 geometric residual is 0.1974 mm and 0.1709 mm respectively. This is an
exploratory result on a previously examined test set, not a validation-selected
improvement or an independent estimate of generalization. Grid-node support is
not complete original-page or waveform coverage. The preview uses 4 px/mm for
display. Its millimeter residual uses physical 5 mm reference identities.

Two old synthetic examples also exercise the geometry-only reference adapters.
Kaggle failed stage0 on both. npj returned RGB warps on both. These are smoke
checks with no paired geometric scoring, not a two-example leaderboard.

## Runnable pieces

```bash
python examples/prepare_grid_benchmark.py \
  --metadata ptbxl_database.csv --out build/hybrid-cohort \
  --train 1024 --validation 128 --test 128 --seed 20261022 \
  --excluded-patients research/hybrid-rectification/excluded-patients.json

python examples/hybrid_rectification_study.py \
  --cohort build/hybrid-cohort --out build/hybrid-study

python examples/rectification_references.py \
  --images locked-images.json --out build/rectification-references \
  --source /path/to/pinned-npj --assets /path/to/locked-kaggle --device cuda
```

The reference image list is a JSON array with `id`, relative `file` and `sha256`.
No command silently downloads pretrained weights. Existing source/weight locks
remain mandatory. `rectification_references.py` invokes no waveform extractor.

## Primary sources

- https://www.medrxiv.org/content/10.1101/2024.08.31.24312876v1.full
- https://github.com/tanghaozhe/physionet-ecg-image-digitization-3rd-place
- https://github.com/Ahus-AIM/Open-ECG-Digitizer
- https://www.nature.com/articles/s41746-025-02327-1
- https://physionet.org/content/ptb-xl/1.0.3/
- https://www.kaggle.com/datasets/physionet/ecg-image-database/
