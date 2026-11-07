# Sim-to-real waveform benchmark protocol

Purpose: measure how well a digitizer trained only on generated images
recovers signals from real ECG images, and compare generators at equal
budget. The scorer is `ecg_gpu.waveform_evaluation` (`ecg-waveform-benchmark`).

## Data you need to supply

Real images with reference signals and patient identifiers, for example
ECG-Image-Database (PhysioNet) or the PhysioNet/Kaggle 2025 digitization
images. They are not redistributed here. Keep every image of a patient in the
same split and exclude those patients from synthetic training sources.

Manifest, one JSON object per line:

```json
{"id": "00022_scan_color", "reference": "refs/00022.npz", "patient": "00022", "capture": "scan"}
```

`reference` is an `.npz` with `signals` `[12,T]` in mV (standard lead order)
and `fs`. Predictions are `<id>.npz` files in the same format, at any sampling
rate; they are linearly resampled to the reference. A missing prediction
scores as zeros.

## Protocol

1. Train the same digitizer architecture, with the same steps, seeds and
   source signals, on each image generator under comparison (this one with a
   named recipe, ECG-Image-Kit defaults, and so on). Use at least three seeds.
2. Predict every real image. Score with
   `ecg-waveform-benchmark --manifest manifest.jsonl --predictions preds --out report.json`.
3. Report the options used. The defaults score raw signals (no offset removal,
   no time shift). `--remove-offset` and `--max-shift-s` make the score more
   forgiving and are not comparable with raw scores.
4. Report per-capture means with patient-level bootstrap intervals and the
   spread across seeds. Synthetic validation scores are not evidence of
   real-image performance.

## Status

Scorer implemented and tested on synthetic arrays. No real-image results yet:
they need the images above and a trained digitizer.
