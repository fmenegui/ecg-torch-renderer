# Validation & testing

Software tests check contracts, numerical regression, visibility, seeds, packaging and downstream training integration. They do not substitute for evaluation against real images and known waveform references.

## Current verification record

{{verification_summary}}

Inspect the [machine-readable verification](https://github.com/fmenegui/ecg-torch-renderer/blob/main/VERIFICATION.json). CUDA-dependent tests are present and skipped when hardware is absent. A skipped CUDA test is not a GPU pass.

## Numerical regression

The frozen 0.2.1 fixture covers all 30 standard layouts. Eager masks are checked bit-for-bit against that reference. Compiled masks are checked on six representative layouts with absolute tolerance `2e-6`. The six CPU public-layer benchmark cases also compared RGB values and observed maximum error zero.

## Supervision contracts

Tests cover one render per paired forward, independent crossing channels, soft/binary targets, visibility under opaque occlusion, reordered/subset channels, repeated rhythm windows, empty slots, pixel-edge boxes and checkpoint continuation.

The hot-path test rejects `.cpu()`, `.numpy()` and `.item()` during rendering and target generation. Separate integration checks run a network backward step and verify finite gradients and changed weights. The CLI exports are checked as arrays, JSON and lossless grayscale files.

## Run the checks

```bash
python -m unittest discover -s tests -v
python examples/segmentation.py --steps 3 --compile-raster
python documentation/validate.py
```

Documentation validation checks local links and anchors, required figures, gallery hashes, Python snippet syntax and generated API metadata. Browser appearance and interactions should also be checked in your deployment browser.

## Evaluate a digitizer separately

Build a held-out patient-disjoint set of actual ECG images with source waveform references. Measure signal reconstruction and calibration error, lead identification, visibility handling and failure rates. Include photographs, scans, varied manufacturers, crops, compression and genuinely damaged paper.

Synthetic test images establish software behavior. They do not establish that the augmentation distribution matches all clinical acquisition conditions, or that a trained inverse model will outperform another system.
