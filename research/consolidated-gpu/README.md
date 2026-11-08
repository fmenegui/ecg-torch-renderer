# Consolidated GPU rectification study

Training, held-out localization, frozen signal extraction and reference comparisons are complete. The result supports a synthetic-domain pipeline improvement, not general or clinical superiority.

## Locked comparison

The three variants share GridFieldNet's architecture, initialization per seed,
full-image batches, image pairs and optimizer budget. `raw-point` receives RGB
without local contrast normalization, `contrast-point` adds that normalization,
and `contrast-global` additionally supervises continuous major-grid coordinates.
The visibility head has the same auxiliary supervision in all variants.

- Seeds 1973 and 20261021, 1,800 AdamW steps each, batch two, learning rate .001
- 768×512 captures, 128 samples/pixel for offline physical rendering
- 32 training, eight validation and eight test patients from PTB-XL 1.0.3
- Four captures per patient, two torch and two Mitsuba views, 192 total
- Previously exposed patients and the exploratory physical ECG's patient excluded
- Thresholds .3, .5, .7, .85 and .93 selected using validation backend-macro F1
- The downstream model is selected using validation only, before test evaluation
- Radius matching at 1.5 capture pixels with one-to-one assignment
- Patient-cluster percentile bootstrap, 2,000 draws, for localization F1 summaries

All captures and deterministic target caches are prepared before training. Test targets do not enter gradient updates, threshold selection or downstream model selection. Test metrics are computed only after all models and thresholds are frozen.

The RGB-only topology functions never receive generator identities. Identity
agreement is calculated afterward using test truth, modulo an origin and lattice
symmetries, and is conditional on selected nodes matched to reference points.
Neither the local nor chart-based association certifies global injectivity.

## Signal extraction

The held-out patients' view-zero online and offline captures give 16 cases.
Each patient retains a separate 12-lead truth signal and declared paper profile.
100 Hz source signals are resampled to 500 Hz only for the frozen digitizer's
output convention. Rendering uses the original 100 Hz physical waveforms.

The frozen npj weights process original RGB, known-grid oracle rectification,
and the validation-selected learned rectification. The oracle and learned
warps transform all RGB content together, including lead names. Learned warps
use supported inferred cells without extrapolation. Rejected geometries remain
in the denominator. Gain is supplied as a declared convention.

Additional matched original-RGB runs use the npj nonlinear dewarper enabled and
the locked Heng/Tang Kaggle single-checkpoint profile. Its fixed 3×4+r1 layout
assumption is recorded for each input rather than treated as support for every
layout. Conditional missing-aware SNR is not the Kaggle competition metric.

Pairwise raw RMSE uses identical finite printed-time samples for original and
intervention. All-case coverage assigns zero to rejection or wrong layout.
An oracle control cannot be presented as an automatically inferred method.

## Execution and recovery

The user's existing stopped A4000 could not start because its host lacked
capacity and remains unchanged. Temporary Secure RTX 3090 preparation attempts
in EU-CZ-1 cost $0.50/GPU hour at catalog read. The first failed on a PhysioNet
timeout before training. Its logs and partial raw files were recovered, and its
owned compute and volume were confirmed deleted. A complete fresh cohort was
then prepared and WFDB checksums verified locally.

A second preparation encountered slow dependency downloads. An attempted
supervisor transfer failed before training and the owned temporary compute was
removed. No trained checkpoint was lost. Its volume was subsequently removed.
This infrastructure failure is retained in the execution record.

The final run uses a Secure A40 in CA-MTL-1 at $0.59/GPU hour and a 10 GB network
volume. All dependencies, input data and pinned reference assets were prepared
locally and uploaded as authenticated chunks, then verified against the complete
bundle hash. Installation and inference need no artifact downloads. The Python
package stack preserves the provided matched CUDA torch/torchvision build.

The staged run has a 9,000-second job watchdog and a 9,800-second supervisor
limit, keeping total GPU time bounded after the preparation attempts. Results
are recovered and checksum-verified before deleting owned compute and storage.
The existing user pod is never deleted or reconfigured.

Staged input bundle SHA-256:
`72db953544b50b380271aa95aabd817b82e3b3e0b06615828d68cbeae95c4176`.
The private bundle includes external inference assets and is not redistributed.
The deliverable results archive excludes those assets and physical photographs.

Reproduction uses the corrected `job.py` with the source package, prepared
`cohort/`, local wheels and pinned inference assets. The exact split execution
scripts are retained as `training-job.py` and `signal-job.py`. `sources.json` records raw WFDB checksums and selected IDs. The public
metadata and signals are PTB-XL 1.0.3, CC BY 4.0. External model sources and
weights are pinned and downloaded outside model inference.

## Limits

This is a small synthetic-capture study, with two seeds and eight test patients.
Patient-bootstrap intervals condition on each trained model and do not measure
training-seed uncertainty. The 100 Hz source cannot validate frequencies above
50 Hz merely by being interpolated to the digitizer's 500 Hz output convention.
Three previously examined physical captures of one printed synthetic ECG are
exploratory only. One ROI has automatic silver annotations, and two images
have no reference points. They cannot establish independent clinical accuracy.
No adapted versions of those source photographs or overlays are redistributed.

Local integration: 364 tests, 350 passed, 12 skipped and two existing errors
caused by the absent original `tests/fixtures/raster-v021.npz`. The fixture was
not regenerated. Seven focused model, denominator and source-identity tests pass. The adverse
capture gallery was rendered and visually inspected.

## Completed detector results

The validation-selected trial is `contrast-point-20261021`, threshold 0.7.
On eight held-out patients it reaches online F1 0.99418 and offline F1 0.97664.
The offline patient-bootstrap 95% interval is [0.96873, 0.98391]. Its offline
matched-point p95 error is 0.79669 capture pixels. The local organizer accepts
15/16 offline images, with at least 99% relative identity agreement among
selected matched nodes in those 15 cases. This is not whole-warp certification.

Contrast normalization does not show a stable improvement over raw input across
both seeds. Its paired offline F1 delta is -0.01191 for seed 1973 and +0.00393
for seed 20261021, with the latter interval crossing zero. The auxiliary global
chart lowers localization F1 in both seeds. All 64 images accepted by its local
chart-cell check fail the >=99% relative-identity criterion. The experimental
CLI therefore refuses chart checkpoints rather than presenting them as a
validated rectification option.

These values must not be attributed to a single change relative to the prior
CPU pilot, whose cohort, architecture, resolution, training budget and physical
sampling differed. The three current variants are the matched ablation.

A clean isolated reproduction confirmed that absent Git LFS filters make the
original binary weights appear modified to git. The strict upstream integrity
gate correctly stopped signal extraction. Configuring LFS locally fixes the
repository status without changing any weight bytes. The continuation verifies
all frozen checkpoint hashes and reuses the existing original, oracle and
learned PNGs, without retraining or regenerating captures. It adds a bilinear
2x resize-only control before observing any signal predictions.

`examples/rectify_field_image.py` provides CPU or CUDA RGB-only inference for
the point-based checkpoints, with a support mask and explicit rejection. This
is an experimental diagnostic, not automatic calibration of gain or speed.
The four previously examined synthetic probes now produce inferred image
rectifications. Their recorded canvas support is not a measurement of original
page completeness, identity correctness or recovered signal quality.

## Portable checkpoint check

From the recovered result directory and this repository:

```bash
python examples/rectify_field_image.py \
  --checkpoint results/models/contrast-point-20261021.pt \
  --image input.png --out build/rectified --device cpu
```

The command writes rectified RGB, interpolation support, predicted points and
a JSON record containing the input and checkpoint hashes. It refuses global
chart checkpoints because their held-out identity checks failed. Support is
measured on the inferred canvas and cannot establish original-page coverage.

The offline setup uses the exact additional package versions listed in
`resolved-dependencies.txt`, alongside the matched CUDA torch/torchvision stack
of the declared official Runpod image. Prepare these wheels outside rented
compute with `pip download --no-deps -r resolved-dependencies.txt --dest wheels`.
External reference assets remain subject to their original provenance and
licenses and are not included in the public result archive.

## Completed signal comparison

All 96 digitizer image attempts are retained, including failures. The configured
npj neural weights stay frozen and the paper gain is supplied in every npj arm.

| Input / reference | Correct-layout accepted outputs | All-case mean printed-time coverage |
| --- | ---: | ---: |
| Original RGB, npj dewarping off | 3/16 | 13.92% |
| Original RGB, npj dewarping on | 4/16 | 17.34% |
| Bilinear 2x RGB, npj dewarping off | 10/16 | 54.69% |
| Learned RGB rectification, npj dewarping off | 10/16 | 58.14% |
| Known-geometry control, npj dewarping off | 7/16 | 36.38% |

The learned and resize-only arms have nine cases with common valid printed-time
support, representing 31.96% of all printed samples. On that identical subset,
pooled raw RMSE is 0.09746 mV learned versus 0.22913 mV resize-only. No temporal
alignment or DC removal enters this comparison. Against npj dewarping on, the
learned arm has 0.08498 versus 0.16987 mV on four common cases, only 7.84% of
all printed samples. These are conditional errors, not whole-population scores.

Kaggle single returns no waveform in 16 attempts, of which only four inputs
match its 3x4+r1 layout assumption. That is a limited transfer diagnostic on
two compatible-layout patients, not evidence about the competition ranking.
Conditional SNR here is not the official Kaggle metric.

The 2x control recovers much of the acceptance and coverage gain. It does not
exactly match the density or crop of every inferred warp, so this study does
not isolate geometric correction from all scaling and framing effects. The
known-geometry control is also not guaranteed to maximize downstream extraction
because its density/crop differ. Six learned cases still fail acceptance or
layout correctness. Even a correct layout does not guarantee complete signals.

Three physical photographs provide exploratory transfer only. The one annotated
ROI gives F1 1.0 for the contrast-selected model, but its references are automatic
silver annotations from a previously examined printed synthetic ECG. The other
two photographs have no point labels. These observations are not independent
clinical validation.

The four older synthetic transfer probes now yield image rectifications. A
geometry/source-hash-verified audit gives offline-flat F1 0.97448 and offline-fold
F1 0.89504, with 100% relative identity consistency among selected matched nodes
in both. These old probes have no new signal-extraction claim.

The final result archive passed its transport checksum, and all six checkpoint
byte streams match the training outputs. All 192 capture hashes were verified.
All owned temporary pods and volumes were confirmed absent afterward. The
original stopped A4000 remains unchanged. Estimated GPU cost from catalog rates
and observed lifecycle durations is $0.4532, excluding storage and not an invoice.
