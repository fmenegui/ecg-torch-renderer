# Paper-configuration adaptation experiment

This work separates an implemented calibrated decoding boundary, frozen-model
transfer tests, profile-based corrections and the remaining training problem.
An arbitrary-layout RGB digitizer has not been trained or validated here.

## Implemented and tested boundaries

`ecg_gpu/digitization/regions.py` removes a fixed number of rows/leads from
waveform decoding. Every printed instance retains named lead, polarity,
calibration, start time, sample rate and missingness. Separate x/y pixels/mm
support anisotropic crops. The decoder does not discover those fields from RGB.
Independent analytical tests cover fractional positions, physical gain/speed,
inverted aVR, repeated leads, missing intervals and resampling without invention.

The original Heng/Tang reference and its four-row decoder are preserved. New
region decoding is a foundation for a future region-conditioned neural head,
not a claim that the frozen network supports new layouts by changing a flag.

## Fixed experimental design

The original eight online PNGs and source signal are copied byte-for-byte.
Two additional physical path-traced captures use the same source signal:
standard 3x4 + rhythm on flat paper and 6x2 on folded paper, 2048x1280 at 64 SPP.
The experiment deliberately holds signal morphology fixed to isolate transfer
and paper-configuration failures. It is not a clinical population study.

Three arms run on the same device with float32 and TF32 disabled:

1. Heng Stage0/1 with Tang's single run15/fold0 waveform checkpoint, unchanged.
   The complete 15-model ensemble shares Stage0/1 and cannot fix an upstream
   reference-localization failure by changing waveform-model averaging.
2. Open-ECG-Digitizer with the published competition inference settings and the
   authors' full common layout catalog. Output is resampled to the supplied
   ten-second convention at 500 Hz.
3. The same npj neural weights with a configured paper interface: no forced
   portrait-to-landscape rotation, grid search 15-120 major lines, a shared
   Cabrera 3x4 + rhythm candidate and the declared nominal gain. This arm is
   profile-assisted, not automatic gain recognition. It does not receive exact
   grid vertices, baselines, lead boxes or reference waveforms.

Profiles are fixed before cloud inference. No checkpoints are modified, no
training is performed and thresholds are not tuned on output scores. The code
and weight hashes of the external references remain locked. The npj source
stays external under its own CC-BY-SA-4.0 license, with paper attribution in
`research/digitization-npj-probe/README.md`.

The same missing-aware score evaluates all three arms. Time alignment is limited
to +/-100 ms and both DC means are removed. Scores are conditional on finite
printed samples. Printed-time coverage, raw RMSE, layout identity, fallback and
empty output remain separate diagnostics. We do not call this the Kaggle metric.
Failures stay in case counts, rather than disappearing from a population score.

## Reproduce locally or on a prepared CUDA machine

```bash
python examples/digitization_layout_probe.py --out build/online --phase prepare
python examples/prepare_digitization_offline.py --inputs build/online \
  --out build/paper-inputs --spp 64
# Explicitly fetch the single Kaggle assets and prepare the pinned npj checkout
# as described in documentation/pages/digitization.md and its research protocol.
python examples/run_paper_experiment.py --inputs build/paper-inputs \
  --out build/paper-results --assets build/reference-assets \
  --npj-source build/upstream/open-ecg-digitizer --device cuda
```

Without CUDA, use `--device cpu`. With one arm only, invoke
`examples/digitization_npj_profiles.py --arms configured`.

## Recorded CUDA results, 2026-10-21

All 30 case/arm combinations completed on NVIDIA A40, Torch 2.8.0+cu128,
CUDA 12.8, float32, TF32 disabled. No network weights were trained or changed.
The following SNR values are conditional on observed printed samples.

| Case | npj published SNR / coverage | npj configured SNR / coverage |
| --- | ---: | ---: |
| standard | 22.36 dB / 81.72% | 22.36 dB / 81.72% |
| without-rhythm | 22.32 dB / 89.38% | 22.32 dB / 89.38% |
| six-rows | 21.06 dB / 92.57% | 21.06 dB / 92.57% |
| portrait | No score / 0% | 6.29 dB / 98.31% |
| cabrera | Wrong layout, 15.08 dB / 63.54% | 22.11 dB / 93.90% |
| gain-5 | 5.69 dB / 67.06% | 21.36 dB / 67.06% |
| speed-50 | 4.27 dB / 100% | 21.08 dB / 100% |
| phone | 13.77 dB / 79.95% | 13.77 dB / 79.95% |
| offline-flat | 18.26 dB / 99.95% | 18.26 dB / 99.95% |
| offline-fold | -7.72 dB / 98.20% | -7.72 dB / 98.20% |

The configured catalog selected the expected layout family in all ten cases.
That is not sufficient for correctness: phone and gain-5 still assign about 20%
of finite output to unprinted lead/time support. Portrait visibly loses peak
amplitude. Folded paper has severe time/baseline deformation despite almost
complete finite coverage. No accuracy acceptance threshold was invented after
observing the scores. Unaligned waveform overlays expose these failure modes.

Heng/Tang single failed in Stage0 for all ten images with fewer than four
document references detected. Neither its waveform head nor the complete
15-model waveform ensemble was evaluated on these images, because the ensemble
shares this failed geometry stage. This is evidence of transfer failure on this
generated set, not evidence that the published method fails on its competition
distribution. It also rules out CPU bfloat16 as the cause of this observed
reference-localization failure: the matched CUDA float32 run fails identically.

The npj CUDA calls took 6.21-7.86 seconds each, except the published portrait
fallback at 24.82 seconds. CPU configured runs provide a separate replication,
not an identical software-stack comparison. GPU peak allocated memory and
individual timings are recorded per case, not promoted as training capacity.

The successful temporary pod existed for 282.60 seconds and was deleted with
read-back confirmation. Three created A40 pods across setup attempts correspond
to approximately US$0.069 at the observed US$0.59/hour compute rate. This is a
catalog-based duration estimate, not an invoice, and excludes storage/billing
granularity. The pre-existing stopped pod was not modified. The cheaper
US$0.28/hour candidate had no capacity at creation time and was never created.

Validation: 49/49 digitization tests passed on the cloud environment with the
official reference artifacts. Full local discovery ran 345 tests: 337 passed,
6 skipped and 2 errored because the original binary regression fixture
`tests/fixtures/raster-v021.npz` is absent. It was not regenerated. The 16-case
adverse capture gallery was regenerated and visually inspected. No real-image
clinical benchmark was run.

The self-contained comparison page, input PNGs, prediction NPZs, unaligned SVGs,
JSON reports, hashes and sanitized lifecycle evidence are exported together.
Generate the page from completed results with:

```bash
python examples/report_paper_experiment.py --inputs build/paper-inputs \
  --results build/paper-results --out build/paper-results/index.html
```

## What remains for unknown paper layouts

| Module | Frozen Kaggle dependency | Required change |
| --- | --- | --- |
| Page alignment | Nine particular printed lead-name landmarks | Infer sheet/grid geometry independently of the printed arrangement |
| Local rectification | Fixed 44x57 indexed lattice | Subpixel nodes, local lattice adjacency and variable grid extent |
| Lead ownership | Three predefined short rows and long II | Detect a variable number of named lead instances and their polarity |
| Waveform network | Four position-dependent channels | Train shared region-conditioned extraction with overlap ownership |
| Units | Fixed canvas/mV constants | Infer or accept declared speed, gain and local grid scale |
| Time | Fixed sequential 2.5-second columns | Preserve declared/recognized simultaneous and sequential windows |

The generator supplies paired images, per-instance masks/boxes, grid IDs and
subpixel positions, physical scales, polarity, time windows and visibility.
These are sufficient supervision interfaces for training the missing components.
They are not substitutes for image-only predictions at deployment.

Start with an explicitly supported family of paper profiles and an independent
real-image evaluation. Then remove the supplied metadata one field at a time.
A visible grid determines millimetres, but not the speed, gain, lead identity or
absolute source time on its own. Missing paper or unprinted signal is retained
as missing rather than represented as a complete recovered acquisition.
