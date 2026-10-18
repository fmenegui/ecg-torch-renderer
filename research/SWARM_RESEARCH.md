# Research review: ECG rendering fidelity and adverse capture

Date: 2026-10-18. Role: research lead. Scope: primary-source review and inspection of the available README, METHODOLOGY, renderer, layout, raster and appearance implementation. This document recommends acceptance gates. It does not report an external validation experiment or certify clinical performance.

## Assessment

The code already has an unusually broad engineering foundation: physical gain/speed coupling, explicit lead identities and time windows, Cabrera polarity, repeated rhythm strips, paired trace targets, approximate paper geometry, lighting and camera artifacts. The README correctly distinguishes these capabilities from demonstrated realism. The highest-value next step is to make adverse scenarios coherent and measurable, then test transfer to independently captured ECG images. More random artifacts alone cannot establish hyperrealism.

ECG-Image-Kit is the relevant open comparator. Its author repository describes rendering from time series with handwriting, paper damage and perspective augmentation [1]. Its publication evaluates digitization and recovered intervals, rather than relying only on attractive examples [2]. The PhysioNet Challenge used synthetic training imagery and real paper images for evaluation, providing a useful precedent for external validation [3].

The ECG-Image-Database publication describes 35,595 images derived from 1,977 recordings, including actual printing, scanning, photography and physical damage. It includes multiple phone cameras, stains and monitor photographs [4]. This makes it a candidate external reference after checking access and license terms. Do not assume its images are independent patients or that printed pages represent all native ECG manufacturers.

## Priorities and concrete acceptance gates

All numeric gates below are proposed engineering thresholds, not thresholds imposed by a clinical standard. They should be versioned and tightened using measured errors. Existing tests may already cover parts of a gate.

| Priority | Work | Acceptance evidence |
| --- | --- | --- |
| P0 | Preserve physical calibration | In canonical coordinates, verify all 3 speeds × 3 gains using independently measured impulse and ramp fixtures. Analytic positions must agree within 1e-4 pixel before rasterization. Raster centerline error should be at most 0.5 pixel where a unique centerline exists. |
| P0 | Calibration marks and grid | Pulse height equals gain × pixels/mm for 1 mV. Pulse plateau width equals 0.2 × speed × pixels/mm. Minor/major grid spacings equal 1/5 mm. Assess high-resolution binary geometry within 1 pixel at each endpoint, separately from antialias coverage. |
| P0 | Lead identity, timing and polarity | Test all built-in layouts with distinguishable per-lead fixtures and an independent layout manifest. Assert standard lead identity, repeated-lead identity, exact source windows and inverted aVR polarity. Validate sequential and simultaneous modes separately. Partial long-strip layouts must explicitly declare their lead subset. |
| P0 | Target alignment under capture | Transform fiducials and independent one-pixel fixtures through the public image/target paths. Compare projected centers within 0.5 pixel in visible nonsingular interiors. Verify opaque occlusion and out-of-frame support remove visible targets. Report invalid support and numerical failures instead of hiding them in a mean. |
| P0 | Reproducibility and provenance | Save source checksum, units, sample rate, length, renderer version, full sampled parameters, profile name, layout manifest and asset hashes. A replay must match on the same supported device/runtime. State that cross-device bitwise identity is not guaranteed. |
| P1 | Coherent capture profiles | Separate clean export, color scan, grayscale photocopy, phone capture, damaged archive and extreme stress. Each profile must have a deterministic replay example, rationale and a parameter-range test. Scanner defaults should not inherit a phone's perspective, flash or cast-shadow model. |
| P1 | Resolution and recoverability | Export actual pixels/mm, pixels/s and mV/pixel, with per-instance windows. Provide high-resolution lossless canonical reference examples. Label low-resolution training outputs and extreme occlusion as information-losing observations. |
| P1 | Current-version performance | Benchmark matched input/seed, image size, profile, outputs and backend. Report median/p90 latency, cold compile, peak memory and image error. Numerical speedups must pass fidelity gates before acceptance. Current paper-model GPU evidence needs a new measurement. |
| P2 | External realism | Lock an independent real-image evaluation set before tuning profiles. Group patient/recording and every repeated capture in one split. Report results by capture and damage class, not only a pooled score. |
| P2 | Advanced 3D and device styles | Add authorized UV/visibility geometry and reference-based device templates only after identifying an external failure they address. Verify geometry and font/asset licensing separately. |

## Resolution is part of fidelity

This is a derivation from the inspected renderer, not a new clinical recommendation. The canonical raster uses

```text
pixels_per_mm = (width - 1) / (duration_s * speed_mm_per_s + 21)
pixels_per_second = speed_mm_per_s * pixels_per_mm
millivolts_per_pixel = 1 / (gain_mm_per_mV * pixels_per_mm)
```

For width 1024, duration 10 seconds and speed 25 mm/s, this is approximately 3.775 px/mm, 94.37 px/s and 10.60 ms per column. At gain 10 mm/mV, one row corresponds to approximately 26.49 microvolts. At width 2048 those values become approximately 7.554 px/mm, 188.84 px/s, 5.30 ms per column and 13.24 microvolts per row.

These are raster spacings, not sampling frequencies promised for successful reconstruction. Multiple input samples can occupy a column. Preserving extrema retains the envelope but loses their ordering within that column. Paper motion, blur, compression and cropping further change observable information. The original sample rate in metadata must not be interpreted as the amount of information retained by the image.

The ECG-Image-Kit authors explain the distinction between signal sampling and scanned image resolution [5]. A DPI tag is not sufficient for arbitrary camera images. Pixels/mm refers to the canonical material coordinates here, while local scale varies after projection.

## External evaluation protocol to establish stronger claims

This is a proposed experiment. It has not been run in this review.

1. Register a benchmark manifest with record IDs, capture family, grouping rule, source license, waveform checksum and image checksum. Remove or mask nonessential identifiers consistently. Avoid diagnostic text leaking a classification target.
2. Match train-record counts and waveform sources for three conditions: clean synthetic, ECG-Image-Kit and this renderer. Keep network architecture, training budget, preprocessing and evaluation fixed. Group all captures of a source record together and use patient grouping where available.
3. Evaluate digitization on real images with paired waveforms. Report calibrated voltage RMSE, SNR, correlation and coverage. State the alignment policy. Do not silently realign amplitude, time scale or lead identity to conceal renderer/calibration errors. Report interval error only with a specified measurement tool and its failure rate.
4. Stratify clean scans, grayscale copies, phone photos, deformed paper, stains, crop/occlusion and low resolution. Include confidence intervals clustered by source recording. A 10-image gallery is not statistical evidence.
5. Predeclare the transfer claim. A suggested research target is a positive paired median-SNR improvement over the matched comparator with a record-bootstrap 95% confidence interval excluding zero, and no large prespecified loss in a clinically relevant stratum. This is a performance target, not a result.
6. Run blinded visual review of matched crops and whole pages, with category labels hidden and acquisition types balanced. Ask reviewers to record specific inconsistencies in grid, labels, ink, illumination and deformation. A real-versus-synthetic classifier can diagnose shortcut cues but cannot establish clinical usefulness.

Do not optimize only perceptual distance, photorealistic backgrounds or a classifier's inability to separate real and synthetic. Trace geometry, timing, scale and downstream transfer are independent requirements. Heavy blur can hide synthetic cues while destroying the information a digitizer needs.

## Honest limits at the inspected baseline

The paper implementation is 2.5D with approximate localized creases. It is not a cloth solver. Analytic shadows do not demonstrate real hand geometry. External geometry maps can extend visibility but need their own validation. The camera module simulates decoded JPEG artifacts, not a complete encoder. Glyph and texture realism depends on supplied assets. Geometric visibility does not imply trace readability.

No external-photo distribution comparison, real paired-waveform reconstruction study, cardiologist realism study or current-version GPU benchmark was executed for this report. The checked documentation explicitly acknowledges these boundaries. Preserve that distinction when describing this work as professional software.

## Primary references

1. ECG-Image-Kit author repository. https://github.com/alphanumericslab/ecg-image-kit
2. Shivashankara KK et al. ECG-Image-Kit: a synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization. Physiological Measurement 2024, 45:055019. https://doi.org/10.1088/1361-6579/ad4954
3. Official PhysioNet Challenge 2024 description, data formats and evaluation context. https://moody-challenge.physionet.org/2024/
4. Reyna MA et al. ECG-Image-Database: A Dataset of ECG Images with Real-World Imaging and Scanning Artifacts. 2024. https://arxiv.org/abs/2409.16612 and https://arxiv.org/html/2409.16612v1
5. Sameni R. A note on electrocardiogram image vs time-series resolutions. ECG-Image-Kit author documentation, January 2024. https://github.com/alphanumericslab/ecg-image-kit/blob/main/codes/ecg-image-generator/documentation/ECG_IMAGE_RESOLUTION.md

Sources were checked on 2026-10-18. Source summaries above are deliberately brief. Acceptance criteria and the evaluation design are research recommendations from this review.
