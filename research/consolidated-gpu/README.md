# Consolidated GPU rectification study

Execution and analysis in progress. No superiority claim is supported yet.

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
capacity. It remains unchanged. A temporary Secure RTX 3090 in EU-CZ-1 costs
$0.50 per GPU hour at the catalog read. A 10 GB standard network volume stores
captures and checkpoints. Compute has a three-hour watchdog and is removed
after checksum-verified result recovery. The owned volume is removed only after
recovery. Neither action deletes the existing user pod.

The first attempt verified CUDA and passed 13 focused tests, then failed on a
PhysioNet connection timeout before training. Its partial data and logs were
recovered, and its temporary pod and volume were confirmed deleted. The fresh
48-patient cohort was then completed and WFDB checksums verified locally before
the second upload. No models or test scores existed during that repair.

Reproduction uses `job.py` with the source package and prepared `cohort/`
directory. `sources.json` records raw WFDB checksums and selected IDs. The public
metadata and signals are PTB-XL 1.0.3, CC BY 4.0. External model sources and
weights are pinned and downloaded outside model inference.

## Limits

This is a small synthetic-capture study, with two seeds and eight test patients.
Three previously examined physical captures of one printed synthetic ECG are
exploratory only. One ROI has automatic silver annotations, and two images
have no reference points. They cannot establish independent clinical accuracy.
No adapted versions of those source photographs or overlays are redistributed.

Local integration: 362 tests, 348 passed, 12 skipped and two existing errors
caused by the absent original `tests/fixtures/raster-v021.npz`. The fixture was
not regenerated. Five focused model and denominator tests pass. The adverse
capture gallery was rendered and visually inspected.
