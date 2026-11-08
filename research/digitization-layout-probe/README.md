# Frozen digitizer transfer to synthetic paper configurations

This exploratory experiment asks whether the published Heng/Tang single profile
already transfers to the package's generated images. It does not train or tune
models. It deliberately tests failures before proposing a general-layout system.

## Reproduce

Install the `digitization` extra and fetch the locked single-profile assets as
described in `documentation/pages/digitization.md`, then run:

```bash
python examples/digitization_layout_probe.py \
  --out build/layout-probe --precision bfloat16
python examples/digitization_layout_probe.py \
  --out build/layout-probe --phase oracle --precision bfloat16
python -m unittest discover -s tests -p test_digitization_layout_probe.py -v
```

CPU float32 is the default. This recorded exploratory run uses CPU bfloat16 with
two threads, the same profile/precision as the preceding real-image pilot.
The renderer code and public weights are unchanged. The ground-truth control
was added after the first automatic reference-localization failure was observed.
Its three cases were fixed before running that control.

## Inputs and evaluation

- Eight cases use one artificial 12-lead signal at 500 Hz for ten seconds. A
  shared time-varying amplitude envelope preserves the limb identities while
  making wrong time windows less likely to hide behind identical repeated beats.
- Configurations include standard 3x4 + long II, no rhythm strip, 6x2, portrait
  12x1, Cabrera, 5 mm/mV gain, 50 mm/s speed and a phone capture.
- The random seed is fixed. Most papers are 2048x1280, the portrait is 1280x2048.
  Font, grid phase and ordinary non-adverse parameters are controlled. This is
  not a population study or evidence for all fonts, pathologies or devices.
- Automatic prediction receives RGB and the requested output sample rate only.
  Generator metadata never enters the automatic predictor.
- Error evaluation uses only printed lead/time support. It reports unaligned
  RMSE, raw overlap SNR, printed-time coverage and unsupported output. It does
  not score an inferred rhythm strip against a signal that was never printed.
  Repeated printed lead/time samples count once in coverage.
- All failures remain in the report. A numerical array is not proof of correct
  lead assignment or full recovery. Unprinted signal cannot be recovered from
  an image by ordinary digitization.

`results/index.html` embeds all displayed images and waveforms. It can be opened
without a local server. The manifest pins input PNG and signal hashes and stores
the renderer's complete realized parameters and canonical instance metadata.

## Recorded result

All eight automatic cases failed in Stage 0 with fewer than four detected
document references. This includes the standard layout, so the observed failure
is not explained solely by changing the arrangement of leads. These results do
not evaluate the automatic Stage 1 or Stage 2 accuracy because they were never
reached. The 0/8 is execution completion on this convenience set, not an estimated
population sensitivity or a claim that every synthetic image fails.

The oracle-geometry control also exposes waveform-transfer errors:

| Clean standard paper | Raw RMSE, mV | Raw SNR, dB | Official aligned SNR, dB |
| --- | ---: | ---: | ---: |
| 25 mm/s, 10 mm/mV | 1.307 | -18.68 | -17.64 |
| 25 mm/s, 5 mm/mV | 2.302 | -23.60 | -12.88 |
| 50 mm/s, 10 mm/mV | 1.905 | -21.96 | -20.43 |

In the ordinary-gain/speed control, V1 and V4-V6 have raw RMSE of approximately
0.010-0.016 mV, while other leads have large errors. A negative SNR means error
power exceeds signal power for the recorded metric. We retain these values.
This control is a supplied-geometry diagnostic, not the original automatic
pipeline, and it cannot establish whether retraining one component would suffice.

The images were generated online with the Torch renderer. Offline ray-traced
captures and wider signal/typography diversity were not evaluated in this run.

## Diagnostic control

The control normalizes three clean standard-layout images using the exact
baseline, time scale and gain provided by the generator. It resamples RGB into
the frozen waveform network's canvas, then runs that network. It does not draw
pixels from the reference waveform. Both learned geometric stages are skipped.

This is an **oracle geometry control**, not an automatic layout detector or a
successful adaptation. It quantifies the remaining waveform-domain transfer
problem after giving the model the required geometry. An independent coordinate
ramp test checks the normalization's pixel map. The official aligned Kaggle
metric is reported only for these compatible standard-layout controls, beside
raw errors, because time/DC alignment can conceal mistakes.

## Architecture needed for varied paper configurations

| Responsibility | Current frozen reference | Required extension |
| --- | --- | --- |
| Page orientation and geometry | Nine particular lead-label landmarks | Sheet/grid geometry independent of printed lead arrangement |
| Grid representation | Fixed 44x57 indexed lattice | Subpixel nodes with local adjacency and a variable-size physical lattice |
| Printed regions | Four predetermined waveform rows | Variable number of named lead instances with bounds, polarity and baseline |
| Calibration | Fixed pixels/mV and ten-second convention | Grid scale, speed, gain and calibration-pulse evidence per region |
| Waveform extraction | Four position-dependent heatmap channels | Region-conditioned extraction with overlapping ownership preserved |
| Time axis | Sequential 2.5-second columns and long II | Explicit simultaneous/sequential windows, unknown timing reported |
| Training | Published competition-domain weights | Synthetic curriculum plus real-image adaptation and independent testing |

The generator already supplies exact grid coordinates/IDs, lead-instance
identity, polarity, canonical bounds/baseline, speed, gain and printed time
windows. Capture-aligned masks, instance boxes and validity are available through
the supervision APIs. These targets support training, but do not replace a
trained image-only detector.

Start with declared paper profiles and explicit metadata as a supported mode.
Then replace those supplied fields with learned estimators and evaluate the
incremental errors. A grid establishes millimetres. It cannot alone establish
mm/s, mm/mV, lead identity or whether columns are simultaneous. Those require
printed evidence, a known acquisition convention or a user-supplied profile.
