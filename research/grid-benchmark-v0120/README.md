# Grid detector pilot, v0.12.0

This is a measured exploratory training experiment. It does not establish a
renderer realism ranking, a clinical digitization result or subpixel physical
accuracy. The full JSON/CSV measurements accompany this report.

## Protocol

Twenty-four independent PTB-XL 1.0.3 patients, one recording per patient,
were selected deterministically with seed 1973. Published folds 1–8 supply
16 training records, fold 9 supplies four validation records and fold 10
supplies four test records. Original 100 Hz WFDB signals were decoded to mV
after checking the initial sample and checksum of each channel. No waveform
augmentation was applied.

Two views per record and backend produce 96 cached captures at 512 by 384.
The online distribution uses torch scan/photo/fold recipes. The offline
distribution uses Mitsuba flat/fold/curl captures at 16 samples per pixel.
They share source/view keys and layouts, but not identical camera/material
distributions. This comparison therefore includes recipe confounding.

Three GridUNet baselines use the same initialization, crop random sequence,
32 training images each, 300 AdamW steps, batch four and 192-pixel crops.
The mixed model alternates backends by source/view key. Its additional
ablations use sigma .6 or 2, no offset regression, or radius-2.5 disks.
Baseline sigma is 1.2 with offsets. One training seed was run. Total measured
CPU training time was 345.84 seconds, excluding rendering and evaluation.

Thresholds come exclusively from validation. Matching maximizes one-to-one
cardinality within 1.5 output pixels, then minimizes total matched distance.
An initial 2048-node decoder cap was found to truncate pages. All saved
weights were recalibrated on validation and rescored on unchanged test
partitions using every peak and sparse matching. No optimizer steps were
repeated. The archived training driver hash and final evaluator hash are
recorded separately. All final test images have zero truncation.

## Measured outcomes

| Training | F1 on torch test | F1 on ray-traced test |
| --- | ---: | ---: |
| Online | 0.916 | 0.025 |
| Offline | 0.380 | 0.208 |
| Mixed | 0.747 | 0.257 |

The mixed model improves transfer into this ray-traced distribution while
losing performance on the torch distribution. This is a substantial domain
gap under this small training budget. It does not show that either renderer
is more realistic. Patient-bootstrap intervals keep both views together and
are included in summary.json. They cover sampling of only four test patients,
not uncertainty from model initialization, site or capture distribution.

The mixed sigma-.6 ablation reaches F1 0.284 on ray-traced images. Sigma 2
reaches 0.097, no offsets 0.227 and disks 0.133. Wide markers can merge nearby
nodes at this resolution. These target choices need validation at the intended
resolution, rather than a single global preferred width.

## External physical captures

Three physical captures of record 00022 were downloaded from
[ECG-Image-Database version 2](https://www.kaggle.com/datasets/physionet/ecg-image-database).
The corresponding PTB-XL patient group is excluded from all synthetic source
partitions. They comprise a color scan, grayscale scan and phone photograph.
Original images and adapted overlays are excluded from this repository and
the deliverable archive. References retain source paths, hashes, dataset
version, license and the independently computed point coordinates.

A clean red-grid region above the waveforms of the color scan contains 234
reference intersections. Profile peaks and local red-line intensity centers
proposed the points, followed by visual review of the original crop. These
are silver references. Annotation uncertainty is uncalibrated. The other two
images have predictions and qualitative inspection only.

The online baseline matches 234/234 points with no extra detections in that
region at the 1.5-pixel tolerance. Its median error is 0.510 output pixels and
p95 is 1.088 pixels. Perfect tolerance-based F1 does not mean exact centers.
The mixed no-offset ablation has median error 0.278 pixels and p95 0.577 in
this same region. Learned offsets are not yet uniformly better than Gaussian
peak fitting. Neither result generalizes to ink overlap, damaged paper,
additional patients, other devices or physical subpixel calibration.

## Paired sampling sensitivity

One held-out offline view was rerendered with 128 samples per pixel, keeping
the signal, mesh, camera, seed, grid coordinates and validity unchanged.
Thresholds were frozen. The mixed model's F1 changes from 0.265 to 0.530.
The offline model changes from 0.178 to 0.256. The online model remains poor,
changing from 0.009 to 0.003. This single-image sensitivity indicates that
render-sampling noise matters, while other transfer limitations remain.
It is not a cohort-wide improvement estimate.

## Reproduction and software evidence

```bash
python -m pip install -e '.[benchmark,pathtrace]'
python examples/prepare_grid_benchmark.py --metadata ptbxl_database.csv --out build/cohort
python -m ecg_gpu.grid_benchmark --signals build/cohort/signals.npz --out build/pilot --ablations
python examples/summarize_grid_benchmark.py --report build/pilot/report.json --out build/summary --captures build/pilot/captures
```

Use the explicit external manifest only after obtaining the original images
and verifying their hashes. Follow dataset licenses when sharing images or
overlays. PTB-XL is CC BY 4.0, DOI 10.1038/s41597-020-0495-6.
ECG-Image-Database is CC BY-ND 4.0, DOI 10.1088/1361-6579/ae85b2.

Ten new unit contracts and a complete artificial-data integration/replay run
pass. Branch-aware coverage is 87% across the two new package modules.
The local 296-test full discovery has 288 passes, six CUDA skips and two
errors caused by the absent original raster fixture in this partial checkout.
Full-checkout CI is required before merge. Lint, package wheel and the
21-page documentation build pass. A 16-scenario adverse gallery was inspected.
These checks do not establish project-wide coverage or photographic realism.

The rectification diagnostic uses oracle lattice identities and held-out
checkerboard nodes. Grid ordering without reference identities, robust
unwarping, waveform extraction, lead/time calibration and signal-level SNR
remain unfinished. CUDA execution and GPU performance remain unavailable.
The next study should expand real annotations and evaluate higher sampling
and resolution with repeated training seeds and matched capture recipes.
