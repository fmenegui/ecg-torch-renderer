# Heng/Tang image-to-signal reference

This is an integration/development evaluation, not a held-out clinical or private
competition benchmark. Public checkpoint training overlap is unknown.

The declared cohort contains PTB-XL-derived records 22, 66, 77, 85 and 107, each
with the digital raster, color scan, grayscale scan and phone photograph from
ECG-Image-Database v2. No image was excluded after observing its score. All
references are original WFDB16 ten-second signals at 500 Hz. Printed short leads
use their sequential 2.5-second windows and lead II uses the complete rhythm strip.

`patients.csv` is a five-row, two-column extract of PTB-XL 1.0.3 metadata, licensed
CC BY 4.0. Source: https://physionet.org/content/ptb-xl/1.0.3/.
Credit: Wagner et al., PTB-XL, Scientific Data (2020), DOI 10.1038/s41597-020-0495-6.
The extract preserves the original record/patient identifiers and contains no
new clinical labels. It lets CI identify its declared patient without repeatedly
downloading the entire metadata table.

## Reproduce

```bash
python -m pip install -e '.[digitization]'
ecg-digitize fetch --assets build/reference-assets
python examples/prepare_digitization_reference.py \
  --out build/digitization-data --metadata research/digitization-reference/patients.csv
ecg-digitize run --assets build/reference-assets \
  --manifest build/digitization-data/manifest.json --out build/digitization-single \
  --precision bfloat16 --threads 2
```

For the full published ensemble, explicitly fetch `--profile full` and run the
same option. The first color scan, record 22, was selected as the full-ensemble
execution probe before observing the 20-image evaluation. A full-ensemble result
on one image must be compared with the same image in the single profile. It does
not estimate an improvement over the whole cohort.

The single-profile run used CPU bfloat16, two threads and no waveform TTA.
Its timings include concurrent work and are not isolated throughput estimates.
The full probe uses four threads and runs without another inference process.
CUDA is unavailable in this environment.

## Integrity and numerical audit

The official metric v7 is hash pinned, including its 1e6 linear-ratio cap.
Failures remain in the score denominator. Raw RMSE is reported alongside aligned
SNR because the official metric permits temporal/DC correction.

The original notebook's high-resolution rectification is bitwise equal in the
independent function-level test. Quadratic waveform decoding differs by less
than 4e-7 mV in the tested synthetic probability maps. Lead fusion agrees within
floating-point roundoff. See `notebook-parity.json`.

The training repository uses PCHIP gap filling, whereas notebook v15 uses linear
interpolation. The adapter follows the notebook. One low-confidence column in
one image was initially processed with the repository's variant. It was replayed
from the saved pre-fusion, Fourier-upsampled rows, with both original artifact
hashes retained. Fourier upsampling is invertible here because 5000 saved samples
preserve all 3924 original columns. The replay rejects downsampled rows and is
tested against a directly computed reference. The aggregate SNR is unchanged at
the reported precision. Original predictions and report remain separate.

Source images, rectified scans, external code and checkpoints are not redistributed
here. Image provenance: https://www.kaggle.com/datasets/physionet/ecg-image-database,
version 2, CC BY-ND 4.0. Public checkpoint datasets report an unknown license.

## What this does not establish

- Reproduction of all upstream training or the private competition score.
- Patient independence from the checkpoint authors' training samples.
- Clinical waveform fidelity across devices, sites, layouts or calibration settings.
- Accurate localization of our own learned grid detector on real documents.
- An advantage of the experimental composed-geometry mode.

The lowest-scoring record is retained and discussed even though inference
completed and the waveform heatmaps were highly confident. Confidence alone
does not prove faithful signal recovery.

## Measured results

The single checkpoint completed all 20 images with aggregate official SNR
22.6897 dB. The lowest image score was 12.7708 dB. Record 85 scored approximately
13 dB across all four capture types, with the main raw errors in V2 and V3.

| Capture type | Images | Official aggregate SNR (dB) |
| --- | ---: | ---: |
| Digital raster | 5 | 23.9485 |
| Color scan | 5 | 22.1343 |
| Grayscale scan | 5 | 21.8053 |
| Phone photograph | 5 | 22.5488 |

On the same color scan, D0_000022_04, the single checkpoint scored 21.7456 dB
and the full 15-checkpoint ensemble scored 24.0709 dB, a 2.3253 dB gain in that
one example. The full run took 500.6 seconds on CPU, four threads, bfloat16,
with process peak RSS 5.09 GB. It is an execution probe, not a cohort estimate.

The experimental single-sampling geometry scored 21.8569 dB on that same image,
0.1113 dB above the single-checkpoint reference geometry. One image is insufficient
to establish an improvement, so the original geometry remains the default.

The machine and dependency versions are recorded in `protocol.json`.
The CPU float32 default was also executed on the same color scan: SNR 21.6333 dB,
82.3 seconds and peak RSS 5.68 GB. The smoke test uses this portable default.
The precision-related score difference on this example is comparable to the
small composed-geometry difference, reinforcing the need for broader paired
experiments before claiming that the geometric change improves digitization.
The first full attempt was interrupted by the OS memory limit while other jobs
were running. The completed full run used an isolated inference process. A
metadata correction records TTA accurately per model group without changing the
predictions. These execution limitations are retained in the protocol.

Open [the complete results page](results/index.html) for every image score,
including the difficult record. Machine-readable results are in
`results/metrics.csv` and `results/summary.json`.
