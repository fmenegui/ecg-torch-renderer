# Validation & testing

Software tests check contracts, numerical regression, visibility, seeds, packaging and downstream training integration. They do not substitute for evaluation against real images and known waveform references.

## Current verification record

{{verification_summary}}

Inspect the [machine-readable verification](https://github.com/fmenegui/ecg-torch-renderer/blob/main/VERIFICATION.json). CUDA-dependent tests are present and skipped when hardware is absent. A skipped CUDA test is not a GPU pass.

## Numerical regression

The frozen 0.2.1 fixture covers all 30 standard layouts. Eager masks are checked bit-for-bit against the immutable 0.2.1 implementation executed in the same runtime. The stored cross-runtime float32 fixture uses absolute tolerance `2e-6`. Assertion failures retain the actual numeric difference and layout group. Compiled masks are checked on six representative layouts with absolute tolerance `2e-6`. The historical six CPU public-layer benchmark cases observed maximum RGB error zero before the paper model changed. Current paired benchmarks compare backends within 0.5.0. Camera rounding permits one 8-bit RGB code of difference. Geometry and trace-mask tests use much tighter float32 tolerances.

## Paper geometry

Tests cover flat identity, unit normals, bounded material strain, dense versus coarse inverse mapping, grazing-angle lighting, visibility outside an external camera bank, scene compilation, random-generator continuity and backend-safe checkpoints. Every procedural family preserves the shared inverse field between RGB and dense targets.

## The uploaded T4 run

The 0.4.0 Colab run executed 54 tests. 53 passed and one bit-exact stored-fixture comparison failed. The original log did not retain the numeric difference. This failure was not reproduced on the current CPU runtime. The new two-part regression check keeps exact same-runtime legacy comparison and a documented float32 fixture tolerance. A fresh Colab run must confirm the change on that Torch/CUDA build.

The original U-Net smoke test completed 20 steps, with loss 0.731 to 0.319. This checks executable training, not held-out waveform accuracy. [Historical T4 evidence](PERFORMANCE_T4_V040.json) preserves measured performance and corrected profiler counts.

## Supervision contracts

Tests cover one render per paired forward, independent crossing channels, soft/binary targets, visibility under opaque occlusion, reordered/subset channels, repeated rhythm windows, empty slots, pixel-edge boxes and checkpoint continuation.

The hot-path test rejects `.cpu()`, `.numpy()` and `.item()` during rendering and target generation. Separate integration checks run a network backward step and verify finite gradients and changed weights. The CLI exports are checked as arrays, JSON and lossless grayscale files.

## Run the checks

```bash
python -m unittest discover -s tests -v
python examples/segmentation.py --steps 3 --compile-raster --compile-scene
python documentation/validate.py
```

Documentation validation checks local links and anchors, required figures, gallery hashes, Python snippet syntax and generated API metadata. Browser appearance and interactions should also be checked in your deployment browser.

## Evaluate a digitizer separately

Build a held-out patient-disjoint set of actual ECG images with source waveform references. Measure signal reconstruction and calibration error, lead identification, visibility handling and failure rates. Include photographs, scans, varied manufacturers, crops, compression and genuinely damaged paper.

Synthetic test images establish software behavior. They do not establish that the augmentation distribution matches all clinical acquisition conditions, or that a trained inverse model will outperform another system.
