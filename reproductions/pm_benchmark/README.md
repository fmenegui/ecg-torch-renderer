# Complete PM-ECG-ID reference catalog and physical clock ablation

This reproduction separates reference preparation, image-only inference and
post-seal scoring. The complete public release contains 6,000 exam entries,
6,600 page images, 100 original recordings and 97 patients. Its 60 conditions
share 700 waveform variants. Signal-noise variants retain their own provided
reference, not the corresponding clean waveform.

## Rebuild the complete reference catalog

Use the pinned environment in `reproductions/requirements-common.txt`. Prepare
the public ZIP index, metadata and signal banks with
`examples.prepare_pm_ecg_external`. Do not download the entire image archive
onto a host without sufficient disk space. Image acquisition supports CRC
checked ZIP member range reads.

```bash
python -m reproductions.pm_benchmark.catalog \
  --root /data/pm-source \
  --ptb-metadata /data/ptbxl_database.csv \
  --originals /data/pm-original-wfdb \
  --out /data/pm-complete \
  --download-originals
```

The builder checks all 1,200 short-lead windows against the 100 original
PTB-XL 500 Hz recordings, with a unique time match and a constant DC residual
tolerance of 1e-6 mV. This DC comparison locates source windows only. It does
not alter the PM reference waveform used for scoring. WFDB byte counts,
calibration, initial values and checksums are verified. Rhythm lead identities
must be unique matches within their own source bank.

Two-page 3x4, two-page 6x2 and four-page 3x4 exams must be grouped before exam
scoring. A single-page layout's time offsets cannot be assumed for a
multi-page print. `contracts.join_pages` accepts only predicted lead ownership
and sample coordinates. It abstains on conflicting samples and never fills
missing samples with zero or averages overlapping pages.

Source-window examples, used for scoring only:

| Public format | Pages | Original time windows verified from source waveforms |
| --- | ---: | --- |
| 3x4 | 4 | All 12 leads span the complete 10 seconds |
| 3x4 | 2 | Groups I/II/III and V1/V2/V3 start at 0 s, other groups at 5 s, with 5 s per short lead |
| 6x2 | 2 | All 12 leads span the complete 10 seconds |
| 6x2 | 1 | First six leads start at 0 s, last six at 5 s, with 5 s per short lead |
| 3x4+1R | 1 | Four groups of three short leads start at 0, 2.5, 5 and 7.5 s, plus a 10 s rhythm lead |

These are verified examples, not reference metadata supplied to a digitizer.
The generated catalog stores the individual result for every original record.

`training-exclusions.json` excludes all 97 benchmark patients, including their
other recordings and every synthetic rendering. Use
`validate_training_patients` before training an eligible new candidate. The
ongoing exact-source npj training is an historical reproduction and has not
been certified as a patient-disjoint candidate under this new protocol.

## Frozen bounded Runpod integration experiment

`clock-inputs.json` contains only image identities, archive byte addresses and
public axes. It schedules 12 exams, 22 pages, six distinct patients and two
conditions. `clock-selection-audit.json` records the score-free SHA256
selection, covering every one-, two- and four-page profile. This is an
engineering ablation on a public benchmark, not a new independent test.

The original npj network runs once per page at its 3,000-pixel GPU profile.
`NativeTraceObserver` captures its pre-resampling pixel columns without changing
native outputs. The alternative uses measured horizontal pixel density and
25 mm/s instead of stretching the surviving columns to the requested sample
count. Native vertical amplitude density and per-row centering remain fixed
to isolate the temporal change. This experiment does not solve baseline or
ST-level preservation. One first-page rerun must give bit-exact plain/observed
native outputs or the experiment fails.

The alternative's x=0 is the aligned image frame's zero. It is explicitly
**not an independently verified printed time origin**. No reference signal,
true layout, patient ID or reference time window enters inference. The worker
records pixel observations and seals page and assembled exam predictions
before any waveform scoring. Unsupported layouts or ambiguous rhythm
ownership become explicit failures.

`.github/workflows/physical-clock-runpod.yml` runs CPU gates before allocation,
uses live capacity and pricing, and caps this separate experiment at 1,800
pod seconds and USD 0.30 GPU time. Temporary container disk is additional.
There is no new training or persistent volume. Cleanup verifies the owned pod
is absent. An identical plan is admitted only once. The existing full npj
training is a separate execution and is preserved.

## Evidence and limits

See `research/hybrid-digitizer-20261030/physical-clock` for the complete catalog
summary, source hashes, CPU bit-exact integration receipt and experiment
results. The catalog marks image acquisition separately from reference
completeness. It does not imply that all 6,600 images have been evaluated.

The complete reference bank, original WFDB sources, metadata, ZIP member index
and CPU control are available in the
[reference bundle](https://chatgpt.com/library?file_id=libfile_4180411680e48191abe9bba1466f21b9&account_id=personal).
Its outer SHA256 is
`b3bf69af5aee92969ae065ce168cd07e5b1d65d69f2963502ad50539352951df`.
The bundle includes 915 files and does not contain all image bytes or the
subsequent GPU experiment. Its README is the pre-upload snapshot.

Post-seal evaluation verifies the frozen page schedule, image identities,
prediction hashes, sample rates and reference hashes before scoring:

```bash
python -m reproductions.pm_benchmark.score_clock \
  --catalog /data/pm-complete \
  --saved /data/physical-clock/outputs \
  --catalog-sha256 CATALOG_SHA256_FROM_RECORDED_SUMMARY \
  --out /data/physical-clock/scored.json
```

All 12 exams and all 22 pages remain scheduled, including complete failures.
The report includes patient-macro error and support decomposition, as well as
condition and page-count strata. It fits no time shift or amplitude scale.
Any internally DC-centered, zero-shift diagnostics are secondary to the raw
physical-axis measurements.

The author-specific alignment and failure protocol is not fully reproduced.
Raw physical-axis scores and support audits must be reported before any
secondary alignment diagnostic. No result here establishes superiority to
PMcardio, Kaggle winners or the original npj method.

The first GPU attempt, Actions run 41133083365, passed CPU admission but Runpod
rejected pod creation with HTTP 402 because account balance was too low. It
produced no new GPU predictions. After the account permits allocation, rerun
the failed benchmark job 124330441527 with the same immutable plan. Do not
change the cohort or manufacture an accuracy report from CPU controls.
