# Open-ECG-Digitizer transfer probe

Stenhede E, Bjørnstad AM, Ranjbar A. *Digitizing paper ECGs at scale: an
open-source algorithm for clinical research*. npj Digital Medicine 9, 145 (2026).
DOI: https://doi.org/10.1038/s41746-025-02327-1.

Original code: https://github.com/Ahus-AIM/Open-ECG-Digitizer, CC-BY-SA-4.0.
Reviewed commit: `97a15087d4abcda843da8c58ee74b1d8f47e6f9a`.
External source and weights are not redistributed or relicensed in this package.

## Protocol

The eight exact PNG files and artificial source signal are the same as in
`research/digitization-layout-probe/results/manifest.json`. No source waveform,
true geometry, true lead identities or actual layout are passed to inference.
The common requested output duration is ten seconds at 500 Hz. This is a supplied
time convention, not inferred speed recognition.

The base configuration is the authors' competition inference YAML. Before
inference, we change both device fields to CPU, request 5000 samples and select
the authors' full thirteen-template layout catalog. Paths become absolute.
No weights, thresholds, segmentation architecture or extraction algorithm change.
The effective configuration is included in the results. Code is verified as an
unmodified checkout and both weights are checked against their Git-LFS SHA-256.

The published configuration retains perspective correction, no experimental
dewarping, 30-70 major grid lines per axis, portrait-to-landscape rotation,
two valid rows for signal trimming and fixed 0.1 mV/mm amplitude conversion.
These are meaningful restrictions when interpreting the gain, speed and portrait
cases. The full catalog does not include 3x4 Cabrera with a rhythm strip.

The authors individually set both U-Nets to evaluation mode. We preserve their
CLI path rather than calling `.eval()` on the wrapper. In this upstream commit,
that call traverses an unused Dewarper whose Module constructor was not called
and raises an AttributeError. No upstream source change was made to avoid it.

## Metrics and missingness

- Canonical output is published in microvolts. The adapter divides by 1000 to
  obtain mV before comparing to the immutable renderer source.
- NaNs are preserved in NPZ predictions. They are never filled, interpolated or
  replaced with artificial zero leads for a successful prediction.
- Printed-time coverage uses unique named lead/time support from the renderer.
  Missing samples remain in its denominator. Unsupported output is also counted.
- Raw overlap RMSE/bias use only finite printed samples with no alignment.
- The aligned score follows the paper's comparison principle: search at most
  +/-100 ms, remove both DC means and choose maximum per-lead SNR. True unprinted
  time windows are masked before alignment. At least 250 ms of paired samples
  are required for a lead score. Mean SNR is arithmetic over scored leads and
  is conditional on observed samples. The lead count is explicitly reported.
- This is not the official Kaggle metric, a reproduction of the article's full
  clinical evaluation, or a claim of clinical validity.
- Inference returning a tensor is reported as `returned`. Correct layout,
  fallback selection and finite printed coverage are separate fields. An empty
  canonical tensor or wrong layout is never counted as demonstrated success.

The published identifier can silently fall back to its first template when
recognition fails. We retain the upstream output and flag that fallback based on
its nonfinite matching cost. The portrait case illustrates why a returned tensor
alone is insufficient. The candidate list is identical for all eight inputs.

## Recorded results

| Generated configuration | Expected layout selected | Printed-time coverage | Mean observed-lead aligned SNR |
| --- | --- | ---: | ---: |
| Standard 3x4 + long II | Yes | 81.7% | 22.36 dB |
| Standard 3x4, no rhythm | Yes | 89.4% | 22.32 dB |
| 6x2 | Yes | 92.6% | 21.06 dB |
| Portrait 12x1 | No | 0.0% | No score |
| Cabrera 3x4 + rhythm | No | 63.5% | 15.08 dB |
| Standard with 5 mm/mV gain | Yes | 67.1% | 5.69 dB |
| Standard at 50 mm/s | Yes | 100.0% | 4.27 dB |
| Standard phone photo | Yes | 80.0% | 13.77 dB |

All eight inference calls returned. Six selected the expected layout, but the
portrait output contained no finite printed samples. Cabrera was assigned a
standard layout. Its numeric score cannot be interpreted as correct lead
identification. The ordinary papers demonstrate useful synthetic-domain transfer
in this small probe. They also demonstrate meaningful missingness.

Changing gain/speed hurts this configuration. In the 5 mm/mV case, the median
per-lead centered least-squares amplitude slope at the nominal printed times is
about 0.48. The published identifier defaults to 0.1 mV/mm, corresponding to
10 mm/mV, and the wrapper does not infer the alternative gain. The 50 mm/s
paper also exceeds the configured horizontal grid-line search prior. A causal
claim that changing that prior alone fixes the speed case requires a separate
controlled inference run, which has not been done here.

Heng/Tang failed before waveform extraction on all eight identical inputs. This
is evidence that Open-ECG-Digitizer transfers more usefully to these generated
examples under the tested settings. It is not a ranking on real patients, a
matched-precision performance benchmark, or a comparison of equal SNR metrics.

## Reproduce

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Ahus-AIM/Open-ECG-Digitizer.git build/open-ecg
git -C build/open-ecg checkout 97a15087d4abcda843da8c58ee74b1d8f47e6f9a
git -C build/open-ecg lfs pull
python examples/digitization_npj_probe.py \
  --source build/open-ecg \
  --inputs research/digitization-layout-probe/results \
  --out build/npj-probe
python -m unittest discover -s tests -p test_digitization_npj_probe.py -v
```

Install a compatible CPU Torch/torchvision pair first. The upstream inference
imports require yacs, scikit-image, scipy, matplotlib, networkx, scikit-learn,
torch-tps, PyYAML and Ray Tune, including a training utility imported indirectly.
They are research-environment dependencies, not additions to the renderer's
mandatory package dependencies. Exact installed versions are recorded in JSON.

`results/index.html` embeds the input examples, signal comparisons and aligned
segmentation previews. Red is waveform, green is grid and blue is text in those
segmentation previews. Signal plots use green for the original and orange for
the reconstruction. Blank reconstruction intervals remain blank.

## Limits

This is a development probe with one artificial nonstationary signal and one
fixed capture seed per configuration. It tests the online Torch generator.
It does not establish external device/patient performance, offline ray-traced
transfer or comparative population accuracy against Heng/Tang. No tuning was
performed on these images, and no missing output was discarded from coverage.
