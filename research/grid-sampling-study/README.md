# Offline sampling and subpixel localization study

This extends the v0.12.0 exploratory pilot. It evaluates frozen models, not
new training runs. No result here establishes photographic realism, physical
subpixel accuracy or complete ECG digitization.

## Protocol and measured results

All eight offline test views from four PTB-XL patients were generated at 16,
32, 64 and 128 samples per pixel (SPP). The physiological signals, printing,
camera, mesh, geometry seed, lattice identities and visibility were unchanged.
All 32 captures have exactly equal geometric targets and validity. Repeated
16-SPP RGB tensors are bitwise equal to the original pilot cache.

Seven fixed checkpoints retain their original validation-selected thresholds.
All peaks are scored without truncation. Matching is recomputed independently
at 0.25, 0.5, 1 and 1.5 output pixels. This is necessary because restricting a
wider-radius assignment is not generally optimal at a narrower radius.

| SPP | Online F1 | Offline F1 | Mixed F1 | Mixed sigma .6 F1 | Median capture seconds, CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.0253 | 0.2080 | 0.2567 | 0.2839 | 1.905 |
| 32 | 0.0237 | 0.2394 | 0.3321 | 0.3589 | 2.246 |
| 64 | 0.0229 | 0.2739 | 0.4222 | 0.4501 | 2.836 |
| 128 | 0.0304 | 0.3228 | 0.5257 | 0.5481 | 4.239 |

The table uses a 1.5-output-pixel tolerance. Mixed F1 improves by 0.2690 from
16 to 128 SPP. An exploratory paired patient bootstrap gives a 95% percentile
interval of [0.2259, 0.3115]. All views and SPP levels remain paired in each of
2,000 resamples. Only four patients are available. The interval does not cover
training-seed, device, site or renderer-seed uncertainty.

The mixed model at 128 SPP reaches F1 0.0481 at 0.25 px, 0.1643 at 0.5 px,
0.3636 at 1 px and 0.5257 at 1.5 px. Its p95 error among matches at 1.5 px is
1.4082 px. Thus the sampling gain does not solve subpixel localization. The
sigma-.6 model has F1 0.1648 at 0.5 px, so its advantage at the wider tolerance
does not establish an advantage for precise centers.

The existing physical scan silver ROI was also rescored at all tolerances.
The online baseline's F1 changes from 1.000 at 1.5 px to 0.496 at 0.5 px. The
mixed no-offset model reaches 0.888 at 0.5 px in that same ROI. Reference
uncertainty is uncalibrated, and the ROI is clean and above the waveforms.
These conditional observations do not establish physical subpixel accuracy.

Timing uses two CPU threads after one discarded warmup, with one observation
per image and SPP setting. It includes the complete `_capture` call, excluding
serialization, inference and matching. The ascending SPP order can confound
timings. These are warm-process wall times, not isolated integrator timings,
steady-state throughput or GPU measurements. All models were trained at
16 SPP, so this study cannot answer whether higher-SPP training is better.

## External expansion and annotation queue

Four additional PTB-XL patient groups were selected before inference from the
archived dataset-v2 file listing. They are the first four complete scan/photo
triplets after excluding record 22 and all synthetic cohort groups. This is a
convenience sample, not a representative or multicenter test.

Records 66, 77, 85 and 107 add 12 physical captures: color scan, grayscale scan
and phone photograph for each patient. All seven models completed inference,
producing 84 model-image results. Together with the original pilot there are
15 physical captures from five patients. No new complete point annotations
were created. Detection counts are not accuracy metrics. Different patients
do not prove different acquisition devices or sites.

`external-manifest.json` records original image hashes and paths. Originals and
adapted physical overlays stay outside the repository and downloadable archive.
The source is [ECG-Image-Database v2](https://www.kaggle.com/datasets/physionet/ecg-image-database),
CC BY-ND 4.0, DOI 10.1088/1361-6579/ae85b2. Synthetic signals retain PTB-XL
CC BY 4.0 attribution, DOI 10.1038/s41597-020-0495-6.

The next annotation step is to define complete ROIs spanning clean grid,
waveform overlap and difficult photo regions. Annotators should work without
model proposals, mark ambiguous/occluded intersections separately, and repeat
a subset independently to quantify disagreement in original pixel units.
Record the exact pixel-center resize transform when reporting output errors.
Do not silently exclude missed nodes or ambiguous regions to inflate F1.

## Reproduce

Use the original cohort, capture cache and seven checkpoints from the pilot.

```bash
python examples/grid_sampling_study.py \
  --cohort build/grid-benchmark-cohort-v2/signals.npz \
  --original build/grid-benchmark-pilot \
  --evaluated build/grid-benchmark-final \
  --external build/grid-benchmark-inputs/external.json \
  --out build/grid-sampling-study
```

`report.json` includes 896 image/model/SPP/tolerance rows, 112 aggregates,
paired intervals, thresholds, source and checkpoint hashes, render timing and
external silver-reference rescoring. `metrics.csv` provides the image rows.
`summary.json` omits image rows for convenient visualization.

The exact executed driver and evaluator are preserved as
`study-driver-executed.py` and `evaluation-executed.py`. Subsequent source
cleanup sorted imports, added parent-input hash preflight checks, and
strengthened bootstrap validation to reject missing or duplicated views. All 224 saved predictions and 32 capture hashes were
verified. Recomputing all 112 aggregates and paired intervals with the stricter
validator gave identical results, recorded in `replay-verification.json`.

`examples/prepare_grid_external.py` prepares listed public triplets, validates
patient separation, byte lengths and image decoding, and saves provenance.
`examples/grid_external_review.py` performs frozen inference using that manifest.
Both refuse nonempty output directories. Source images must be obtained from
the original dataset. No cloud/GPU execution or retraining occurred.

## Next experiment

Use this pilot as development evidence. Predeclare three training seeds and
compare standard versus narrow heatmaps, with and without offsets, at the
intended output resolution. Select thresholds and sampling budgets using
validation. Preserve a new untouched patient-disjoint final test, because the
present pilot test has now informed development. Grid ordering, robust
rectification without oracle lattice IDs, waveform extraction and signal-level
evaluation remain separate unfinished milestones.
