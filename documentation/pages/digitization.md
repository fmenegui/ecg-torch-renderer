# Reference ECG image digitization

The optional digitization package integrates Heng's learned document/grid
rectification with Tang's published waveform models. The renderer remains an
independent image/target generator. No public checkpoints are bundled in the
wheel and inference never downloads code or weights implicitly.

## Install and run

Install a matching Torch/torchvision pair for the chosen CPU or CUDA platform
first. A CUDA torchvision wheel cannot be used with a CPU-only Torch build.

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[digitization]'
ecg-digitize fetch --assets build/reference-assets
python examples/prepare_digitization_reference.py --out build/digitization-data
ecg-digitize run --assets build/reference-assets \
  --manifest build/digitization-data/manifest.json --out build/digitization-run \
  --device cpu --precision bfloat16 --threads 2
```

`fetch` is an explicit setup action. `check` verifies the downloaded artifacts
without downloading anything. All artifact byte lengths and SHA-256 hashes
are fixed in the packaged `reference-lock.json`. Failed downloads leave no
completed artifact. Mismatched existing files are rejected, not overwritten.

## Profiles and geometry

| Option | Meaning |
| --- | --- |
| `--profile single` | run15/fold0 ConvNeXtV2 waveform checkpoint, no waveform TTA by default |
| `--waveform-tta` | Four sequential flips for the single profile |
| `--profile full` | Notebook v15's 15 waveform checkpoints: run15 five folds with four flips, run13 five folds, HRNet run10 five folds, weights 0.5/0.3/0.2 |
| `--geometry reference` | Preserve Tang's high-resolution homography, grid warp, final resize and historical interpolation conventions |
| `--geometry composed` | Experimental inverse-map composition, sampling the original rotated image once with consistent pixel-center conventions |

The full profile requires a separate explicit download:

```bash
ecg-digitize fetch --assets build/reference-assets --profile full
ecg-digitize run --assets build/reference-assets --profile full \
  --manifest build/digitization-data/manifest.json --out build/digitization-full \
  --device cuda --precision float16
```

The single artifact set is about 540 MB. The full set is about 5.98 GB.
Memory requirements depend on precision and image dimensions. Inference is
sequential across TTA/folds, but full-resolution decoder activations remain
large. Run the full profile in isolation on a memory-limited machine. Reports
record precision and device, and CPU measurements are not GPU benchmarks.

`composed` is an experiment, not a silent correction to the reference. Applying
the inverse homography to sparse nodes and then interpolating those transformed
nodes is generally wrong. We interpolate in the normalized-image coordinate
system before applying the inverse projective map to the dense field.

## Physical and temporal contracts

This adapter assumes the competition's 3x4 layout plus a ten-second long lead II.
It uses the published 4400x1700 rectified canvas, 4352x1696 network crop, waveform
columns 236:4160, zero-mV positions [703.5, 987.5, 1271.5, 1531.5] and 79 px/mV.
These are model-specific constants, not calibration inferred from arbitrary ECGs.

Outputs contain twelve `LeadSignal` objects with finite immutable mV samples,
sample rate and printed start time. Short leads retain their sequential time
windows. Lead II uses the complete rhythm strip, with the published short/long
overlap fusion. Einthoven correction is restricted to the simultaneous first
segments of I, II and III. It is not applied across different temporal columns.

Waveform decoding refines each column's peak with a quadratic fit and resamples
in the Fourier domain. Gaps use linear interpolation to match notebook v15.
The training repository's PCHIP variant is a separately selectable decoder
option, not silently substituted into reference inference. Interpolated
low-confidence columns remain marked in
`observed_columns`. A completely missing row fails explicitly. The adapter does
not fabricate a successful zero waveform when a stage fails.

The current manifest reference loader supports the supplied interleaved WFDB16
records, explicit gain/baseline in mV, twelve named channels and ten seconds at
integer sample rate. It validates original byte length, first sample and channel
checksums before selecting printed windows. Arbitrary layouts, cropped ECGs and
competition CSV ingestion are separate extensions.

## Arbitrary calibrated lead regions

The frozen Kaggle networks retain their published fixed layout. A separate
layout-independent boundary can decode any number of individually detected,
rectified lead regions, including subsets and repeated rhythm occurrences:

```python
from ecg_gpu.digitization import TraceCalibration, decode_trace_region, resample_trace

calibration = TraceCalibration(
    instance_id='rhythm-V1', lead='V1',
    pixels_per_mm_x=8., pixels_per_mm_y=4.,
    speed_mm_per_s=25., gain_mm_per_mv=10.,
    baseline_y=70., start_s=0.,
)
observation = decode_trace_region(region_probability, calibration)
at_500_hz = resample_trace(observation, 500.)
```

Calibration uses the heatmap's own pixel-center coordinates. X and Y may have
different scales after network resizing. Speed controls time, gain controls mV,
and `polarity=-1` restores an inverted printed lead such as Cabrera's -aVR.
Identity, calibration, baseline and printed start time must come from a paper
profile or separately trained image detector. This is not an automatic unknown
layout recognizer and does not modify the published checkpoints.

`TraceObservation` represents one printed occurrence. Its vectors are immutable.
Weak, boundary-clipped and ambiguous columns remain NaN with `observed=False`.
Resampling interpolates only within contiguous observed runs, preserving missing
gaps and avoiding extrapolation. No twelve-lead or ten-second completeness is
implied. The existing `DigitizationResult` remains the stricter twelve-lead
contract of the published reference.

`examples/digitization_npj_profiles.py` compares published and configured npj
inference on the same images and device. The configured arm expands the common
layout catalog, preserves portrait orientation, broadens the grid search and
passes a declared paper gain. It rejects empty signals, layout fallbacks and
inconsistent row counts. Its gain is supplied metadata, not automatic OCR.

`examples/prepare_digitization_offline.py` extends the online probe with two
actual 64-SPP path-traced captures while retaining the exact first eight PNGs.
`examples/run_paper_experiment.py` runs npj profile controls and Heng/Tang with
the same float32 device and the same missing-aware evaluation convention.

## Manifests, outputs and evaluation

`examples/prepare_digitization_reference.py` writes a ready-to-run manifest with
image hashes, original WFDB hashes, patient identity, capture kind and source
license. It declares the selected records as development cases. Patient identity
must not cross train/validation/test partitions. This cannot establish separation
from the public checkpoint authors' training data, which remains unknown.

`run` writes one NPZ per successful image and an atomically updated `report.json`.
NPZ files contain mV leads, sample rate, start times, learned grid, homography,
row waveforms and original-column observation masks. Source and rectified images
are excluded. Existing nonempty output directories are rejected.

Reports include per-image/per-lead errors, per-capture-type results, failures,
stage timings, process peak RSS and source/model provenance. A failed image with
truth stays in the score denominator using an explicitly labeled zero-prediction
penalty. It does not receive a prediction artifact. The CLI returns nonzero if
any image failed. A report still marked `running` indicates an interrupted process.

The official metric is loaded from the exact hash-verified Kaggle metric v7
source. Its ratio cap is 1e6. Older copies with a cap of 384 are not equivalent.
It aligns within 0.2 seconds and removes a DC offset. We therefore also report
unaligned RMSE, bias, maximum error and estimated alignment shift for each lead.
Aggregation averages the capped linear SNR ratios across images before taking
10*log10. Averaging image scores in dB is not the competition score.

## Verification and scope

```bash
python -m unittest discover -s tests -p test_digitization.py -v
python examples/digitization_smoke.py --assets build/reference-assets \
  --manifest build/digitization-data/manifest.json --out build/digitization-smoke
```

The independent tests cover fractional peaks, projective-map composition,
border validity, temporal ownership, mV conversion, original WFDB checksums,
patient leakage, artifact corruption and explicit failure accounting. CI runs
an actual published checkpoint through a declared real-image development fixture.
An SNR floor on that fixture is an integration guard, not clinical validation.
The CI smoke defaults to float32, matching the CPU CLI default and avoiding a
dependency on efficient native bfloat16 instructions in the runner hardware.

The current work reproduces published inference components. It does not reproduce
all training, the private competition test or clinical generalization. Upstream
datasets report an unknown artifact license. The code, weights, source images
and adapted scans remain external rather than being relicensed or redistributed.

## Synthetic paper-layout transfer probe

`examples/digitization_layout_probe.py` tests the frozen single profile against
eight images generated by the package, varying layout, portrait orientation,
gain, speed and phone capture. Automatic prediction receives RGB only. Generator
metadata is reserved for evaluation. Reports retain explicit failures and measure
printed-time coverage and unsupported output alongside raw waveform errors.

```bash
python examples/digitization_layout_probe.py --out build/layout-probe
python examples/digitization_layout_probe.py --out build/layout-probe --phase oracle
```

The separately labeled oracle control supplies exact geometric calibration for
three clean standard-layout images and tests only waveform-network transfer.
It skips the learned geometry stages and is not automatic digitization. Open
`build/layout-probe/index.html` for embedded images and unaligned signal overlays.
See `research/digitization-layout-probe/README.md` for the protocol and the
architectural changes required for paper configurations beyond the Kaggle layout.

`examples/digitization_npj_probe.py` evaluates the separately published
Open-ECG-Digitizer from Stenhede et al. (npj Digital Medicine, 2026) on the exact
same inputs. It requires an explicit unmodified, pinned external checkout and
Git-LFS weights. Its partial canonical output retains NaNs, converts microvolts
to mV and reports coverage, layout fallback and a separately named paper-style
aligned score. See `research/digitization-npj-probe/README.md` for setup and scope.

## Primary sources

- [Heng training repository, MIT](https://github.com/hengck23/kaggle-physionet-ecg-image-digitization)
- [Heng stage0/1 source and weights, dataset version 3](https://www.kaggle.com/datasets/hengck23/hengck23-demo-submit-physionet)
- [Tang third-place writeup](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/3rd-place-solution)
- [Tang inference notebook, reviewed version 15](https://www.kaggle.com/code/hirotetsu/physionet-submission-3rd-place)
- [Tang source and checkpoints, dataset version 35](https://www.kaggle.com/datasets/hirotetsu/physionet-data)
- [Official Kaggle ECG signal extraction metric, version 7](https://www.kaggle.com/code/metric/physionet-ecg-signal-extraction-metric)
- [ECG-Image-Database, version 2](https://www.kaggle.com/datasets/physionet/ecg-image-database)
