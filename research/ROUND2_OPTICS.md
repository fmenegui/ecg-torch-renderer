# Round 2: capture radiometry review

The previous camera performed illumination, defocus, shutter averaging and
signal-dependent noise directly on display-encoded RGB. That is a physical
error. A dark pixel and a bright pixel carry a nonlinear relationship to their
code values. Blurring their codes loses light at high-contrast ECG edges.

## Implemented correction

`CameraAppearance` receives sRGB-encoded scene colors. For an active sensor its
observation chain is now:

1. Decode the standard piecewise sRGB transfer into linear RGB.
2. Apply exposure, directional illumination, vignetting and broad shadow
   attenuation to irradiance.
3. Integrate the existing finite Gaussian defocus plus constant-velocity shutter
   PSF in linear light. Keep highlight energy until after optical integration.
4. Add independent read noise and a Gaussian approximation to shot noise, with
   variance `read_noise**2 + shot_noise**2 * linear_signal`.
5. Saturate the normalized sensor, apply digital channel white balance, encode
   sRGB, then apply the optional display `gamma` adjustment.
6. Apply the existing decoded JPEG approximation and 8-bit observation
   quantization.

The geometric targets never enter this chain. Blur changes the image and does
not expand the geometric trace mask or change the original mV waveform.

`sensor_active=0` retains the direct document display path, including historical
display-space noise coefficients. Fractional activation blends the two encoded
observations. That blend is an augmentation convenience, not a physical mixture
of two camera sensors. A fully active camera interprets noise parameters in
normalized linear sensor units. Existing parameter names and shapes are kept.
This behavior change must be represented by the renderer fingerprint/version.

## Independent checks

`python -m unittest tests.test_capture_physics tests.test_realism -v`

All 15 checks passed locally on CPU after the appearance change. Six new checks
exercise independent scalar transfer references, one-stop exposure, additive
photon energy at a black-white edge, highlight transport before saturation,
linear read/shot variance with digital white-balance amplification, and the
inactive-sensor compatibility path. Existing checks independently integrate a
moving point-source Gaussian with 4096 shutter samples and verify DC, variance,
orientation, reference-waveform preservation and geometric-target invariance.

The one-stop reference is especially diagnostic: a white patch at half
irradiance produces sRGB 0.73535698 before quantization. Multiplying its display
code by 0.5, as the previous camera did, gives a materially incorrect exposure.

## Cost and limits

An isolated camera-component microbenchmark with `B=1`, `768x1024`, two torch CPU
threads and five timed repetitions after two warm-ups measured median 121.7 ms
for the previous implementation and 164.2 ms for this linear implementation.
These are local CPU component timings, not end-to-end or GPU throughput claims.
The torch build was `2.14.1+cpu`. CUDA was unavailable. Additional transfer
functions and the direct-document compatibility branch have a measurable cost.

The PSF remains a normalized finite 9x9 discrete approximation. Its existing
Gaussian defocus model does not simulate spatially varying aberrations,
focus-depth dependence, an aperture Airy pattern or finite pixel-area
integration. Large parameter overrides can make finite-support truncation
substantial and should not be treated as calibrated long-exposure photography.
The sampled scenario range is much narrower than arbitrary overrides.

The RGB sensor has no Bayer mosaic, demosaicing, rolling shutter, color
cross-talk, lens-specific chromatic aberration or vendor ISP. Shot noise is a
Gaussian approximation suitable away from photon starvation, not exact Poisson
photoelectron counts. Digital white balance is separate from sensor clipping,
but there is no fitted full-well capacity, ISO conversion gain or spectral
response. Grid and thermal-print formation remain simplified document models.

These corrections remove identifiable physical errors. They do not establish
equivalence to real scans or phone photographs. That claim needs a held-out
real-image benchmark and blinded visual evaluation.

## Primary references

- [W3C CSS Color 4](https://www.w3.org/TR/css-color/), sRGB transfer functions and
  linear-light RGB color spaces.
- [EMVA 1288 Linear, release 4.0](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf),
  linear sensor response and read/shot/quantization variance model.
- [PBRT, fourth edition: Film and Imaging](https://pbr-book.org/4ed/Cameras_and_Film/Film_and_Imaging),
  radiometric sensor measurement and camera image formation.
