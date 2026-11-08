# Roadmap to a state-of-the-art ECG image generator

Goal: the generator whose synthetic-only training transfers best to real ECG
images, demonstrated on public benchmarks. Every item has an acceptance
criterion; "done" means the criterion was measured, not that code exists.

## The bar

| Measure | Reference point | Target |
| --- | --- | --- |
| Synthetic-only training, SNR on real images | PhysioNet Challenge 2024 winner (ECG-Image-Kit data): synthetic CV 12.6-17.0 dB vs hidden real 4.2-12.2 dB; phone photos below 0 dB (arXiv 2410.14185) | Beat ECG-Image-Kit at equal budget on ECG-Image-Database and Kaggle 2025 images |
| Clinician real-vs-synthetic test | GenECG round 3: 53.3 % accuracy (medRxiv 2023.12.27.23300581) | Not distinguishable from chance with a preregistered protocol |
| Patch-level distribution distance | None published for ECG images | KID decreasing release over release |

## Status legend

`planned` · `in progress` · `implemented` (code and analytic tests) · `measured` (effect shown on real images)

## 1. Evaluation

| # | Item | Acceptance | Status |
| --- | --- | --- | --- |
| 1 | Sim-to-real waveform benchmark: train on synthetic only, SNR on real | Reproducible script, patient-disjoint, 3 seeds, CIs | in progress: scorer and protocol in 0.22.0; needs real images and a trained digitizer |
| 2 | Same-budget comparison with ECG-Image-Kit and GenECG-style captures | Same network, signals and steps | planned |
| 3 | Ablation per realism factor | Each factor's SNR delta with CI | planned |
| 4 | Clinician test and patch KID per release | Protocol and results in `research/` | planned |
| 5 | Paired real dataset: print signals on real devices, photograph and scan | Consent, licence, device list, ground truth | planned |

## 2. Printed document

| # | Item | Acceptance | Status |
| --- | --- | --- | --- |
| 6 | Fixed paper width; speed sets time window | 12.5/25/50 mm/s show 20/10/5 s on 250 mm | implemented, 0.14.0 `page_model="physical"` |
| 7 | Stroke and grid widths in mm | Flat-line coverage integral equals width at every resolution | implemented, 0.14.0 |
| 8 | Isotropic strokes | Perpendicular width within 5 % for slopes 0-10 | implemented, 0.15.0 `stroke="isotropic"` |
| 9 | Trace-to-grid registration error | Feed-speed error | implemented: feed error 0.14.0, trace skew 0.25.0 |
| 10 | Lead-switch marks and segment breaks | Marks in masks' complement, separate target | implemented, 0.16.0 `lead_marks=True` (marks); breaks come from sequential windows |
| 11 | Generated text: header, measurements consistent with signal, statements, footers, languages | OCR boxes and transcripts as targets | implemented, 0.19.0 `report_text=True` (rate from signal; other intervals fictional; no statements by policy) |
| 12 | Vendor-style layouts, calibration pulse variants | At least 8 generic styles | planned |
| 13 | Multi-page prints | Page index and cross-page time mapping | planned |
| 14 | Print technologies: laser halftone, inkjet, dot matrix, fax, photocopy generations | Visual references per technology | in progress: laser halftone, photocopy, fax in 0.23.0 `print_technology=True`; inkjet, dot matrix planned |
| 15 | Handwriting, stamps, stickers, barcodes | Separate occlusion target | in progress: barcodes, QR, stamps in 0.29.0 `office_marks=True`; handwriting with words, stickers planned |

## 3. Signal

| # | Item | Acceptance | Status |
| --- | --- | --- | --- |
| 16 | Diverse real sources (PTB-XL, CODE-15 %, MIMIC-IV-ECG, Chapman) | Loader and provenance | planned |
| 17 | Printed reference: device filters, clipping, alongside the untouched source | Both references returned | implemented: filters 0.18.0, clipping 0.28.0 (`device_filters=True`) |

## 4. Paper and geometry

| # | Item | Acceptance | Status |
| --- | --- | --- | --- |
| 18 | Content-displacing deformation in the training path from simulated sheet banks | Grid lines measurably bend; target alignment tests | in progress: pinhole parallax 0.31.0 `perspective_relief=True`; bank mixing by default needs Mitsuba banks and a GPU transfer measurement |
| 19 | Held pages, stacks, overlaps, curved supports | Recipes with aligned targets | planned |
| 20 | Mold, scorch, scanner-jam damage, tape, staples | Recipes referenced to ECG-Image-Database | in progress: mold and scorch in 0.26.0 `archive_damage=True`; scanner jam, tape, staples planned |

## 5. Capture

| # | Item | Acceptance | Status |
| --- | --- | --- | --- |
| 21 | Phone ISP: demosaic, denoise, sharpening, tone mapping, real codecs, lens profiles, glare, shake | Each stage toggleable and tested | in progress: aberration, sharpening and glare in 0.20.0 `phone_isp=True`; demosaic, tone mapping, codecs planned |
| 22 | Scanner modes: grayscale, bilevel, 72-600 dpi, skew, lid shadow; fax | Recipes | in progress: modes and dpi 0.17.0, fax 0.23.0, lid shadow 0.30.0; skew planned |
| 23 | Screenshots and EHR viewers; realistic display banding and moiré | Recipes | planned |
| 24 | Licensed real background bank | Attribution file | planned |

## 6. Data-driven realism

| # | Item | Acceptance | Status |
| --- | --- | --- | --- |
| 25 | Fit sampler ranges to statistics of real images | Documented fitting and before/after KID | planned |
| 26 | Real-vs-synthetic discriminator as a diagnostic | Ranked list of giveaways per release | planned |
| 27 | Optional structure-preserving appearance refinement | Target alignment preserved, KID improves | planned |

## 7. Supervision, performance and adoption

| # | Item | Acceptance | Status |
| --- | --- | --- | --- |
| 28 | Pixel→(lead, time, mV) map, text targets, page corners, depth, pulse and label masks | Contracts and tests | in progress: time/voltage maps 0.21.0, text targets 0.19.0, lead marks 0.16.0, page corners 0.27.0, pulse masks 0.28.0; depth planned |
| 29 | GPU profiling, sparse raster, stage compilation, compact targets | See `OPTIMIZATION_PLAN.md` | planned |
| 30 | PyPI release, versioned docs on Pages, calibrated presets | Published | planned |
| 31 | Released datasets and a paper with the benchmark | Public | planned |
