# Independent engineering quality review

This review evaluates software contracts and numerical behavior. It does not
certify clinical validity or perceptual indistinguishability from real ECG
photographs. Synthetic unit signals are useful for identifying errors, but do
not substitute for evaluation against independent clinical acquisitions.

## Executed environment and results

- Python 3.12, PyTorch 2.14.1+cpu, no CUDA device.
- Integrated discovery: `python -m unittest discover -s tests -v` completed
  **91 tests in 67.234 s, passed with 3 CUDA skips**. The discovery started
  before the final scenario tests and final adversarial additions were saved.
- Initial independent adversarial checks: 9 tests passed in 1.058 s.
- Scenario-specific tests: 3 passed in 0.746 s.
- Integrated final discovery: **101 tests in 22.633 s, passed with 3 CUDA
  skips**, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- Expanded adversarial checks: 15 tests passed in 1.422 s after precision
  guard, constructor validation, provenance and scanner illumination changes.
- Final focused adversarial checks: **16 passed**, including tiny Python
  fractions in direct source lengths, categorical controls and booleans.
- Module/CLI integration rerun after constructor hardening: **13 passed in
  4.428 s**, covering checkpoint replay, AMP, downstream training and exports.
- Final focused adversarial checks: **16 passed**, including tiny Python
  fractions in direct source lengths, categorical controls and booleans.
- Module/CLI integration rerun after constructor hardening: **13 passed in
  4.428 s**, covering checkpoint replay, AMP, downstream training and exports.

The original frozen raster fixture and immutable legacy ZIP were recovered
before integrated discovery. Missing assets during partial materialization
were an environment staging issue, not a renderer failure.

## Findings and changes

| Finding | Consequence | Resolution / evidence |
| --- | --- | --- |
| Integer-valued source lengths were cast before validation | `[20.9]` silently became `[20]`, changing the valid waveform window | CLI now validates numeric, finite, integral, in-range lengths before int64 conversion. Tests reject fractional, NaN, infinity, complex, string and boolean lengths, and accept integral floats. Renderer guard handled by realism reviewer |
| Categorical parameter overrides were cast before validation | `layout=-0.5` or `gain_index=1.9` silently selected another identity or physical scale | Renderer, fixed layer values and Choice banks validate integrality before integer casts. Boolean controls require 0 or 1. Independent regression covers layout, gain, orientation and rhythm lead identity |
| Finite float64 source can overflow renderer float32 | `torch.full((1,12,20),1e40,dtype=torch.float64)` produced nonfinite RGB and trace targets | Renderer now checks finite working precision and bounds pixel coordinates by the safe float32 range. Regression passes |
| WFDB provenance hashed only the `.hea` file | Two different `.dat` waveforms sharing the same header had indistinguishable input-file hash | CLI now also exports `normalized_input`, a canonical digest of decoded voltages, rates and valid lengths, with individual array digests |
| CI pinned only particular current versions after the minimum | Subsequent compatibility would not be exercised | CI retains Python 3.10 / Torch 2.6.0 minimum, adds latest stable CPU Torch on Python 3.12 and 3.13 |
| Scanner recipes retained sampled scene lights | Nominal flatbed scans could receive directional paper illumination | Non-photo recipes set scene illumination to unity and disable directional/flash/specular effects. Independent scene-map test confirms uniform illumination within float32 epsilon |

`normalized_input` uses C-order arrays, explicit little-endian float32
voltages/rates and int64 lengths. The combined SHA-256 includes the schema,
array shapes/dtypes and array bytes in signals/rates/lengths order. It records
the actual decoded physical input. It does not claim to archive every WFDB
sidecar or the original raw encoding. The raw `input_sha256` is retained.

## Independent adversarial coverage

The 16 additional tests exercise:

- Every supported layout with the shortest legal waveform, on an odd canvas.
- Exact image and mask isolation from finite values beyond each padded length.
- Exact pixel and canonical lead identity invariance across lead chunk sizes.
- No mutation of a noncontiguous source tensor and no source gradients.
- No consumption of the global Torch RNG when a generator is explicit.
- Fractional identity rejection and working-precision overflow rejection.
- Exclusive-edge boxes for all four image corners and absent channels.
- CLI length validation and canonical physical-input fingerprints.

The scenario review confirms that all default scenario layouts include the
twelve canonical leads and explicitly disable stochastic lead, label and
calibration-pulse dropout. Damage can still hide printed content and must be
assessed with the visibility masks. Explicit missing-lead layouts remain
available in the general renderer and are distinct from these recipes.

## Acceptance limits

CPU contract success does not establish CUDA parity or a new GPU throughput
number. The existing GPU tests must run on real hardware. Latest-version CI
is a compatibility monitor and does not provide bitwise cross-version
reproducibility. Freeze software, assets, parameters and device context for a
specific reproducible study, and retain the normalized-input fingerprint.

The renderer reproduces supplied 12-channel voltage data. It does not generate
or verify the physiological consistency of those channels, diagnose disease,
or guarantee that a real-world ECG digitizer will generalize to its images.
The procedural document model is 2.5D and the JPEG stage simulates decoded
compression artifacts. External assets and independent real examples remain
necessary for a defensible hyperrealism evaluation.

## Final rerun

The integrated suite passed after the numerical guards, scanner recipes and
fingerprint integration. Constructor controls also preserve Python float
precision before validation, preventing values such as `1.00000001` from
rounding to an integer category or `1e-50` from rounding to a boolean zero.
Direct renderer overrides and non-tensor source lengths apply the same policy.
The final focused adversarial and module/CLI suites passed after this last
precision hardening. Direct renderer overrides and non-tensor source lengths apply the same policy.
The final focused adversarial and module/CLI suites passed after this last
precision hardening. The numerical render remains unchanged for valid controls.
