# Rectification geometry fixes and frozen CPU replay assessment

The explicit cell mesh fixes an actual forward/inverse topology defect. A validation-selected background-banding filter improves a narrow subset of detector failures. The frozen CNN remains the main limitation in difficult 3D captures. No new model was trained and no GPU was provisioned.

## Geometry correction

Forward evaluation previously triangulated observed points with Delaunay while inverse RGB sampling triangulated canonical points independently. Their simplices can differ even for a perfect grid under an affine shear. `cell_mesh.py` now constructs two fixed triangles per complete convex lattice cell and uses them in both directions. Missing corners remain missing. Strict interior overlaps are rejected. Inverse sampling also checks the return to each output pixel center. This is sampled ambiguity rejection, not a global injectivity certificate.

The independent 5,000-query affine-shear control changes support from 8.18% to 100%, with maximum coordinate error 1.14e-13 pixels and no fitted scale. `analytic-control.json` and six mesh tests preserve that control and coordinate-ramp, hole and overlap checks. In the actual replay set, the mesh-only aggregate gain is small. The control establishes correctness, not a large empirical detector gain.

## Data, selection and measurement

Public PTB-XL source bytes and WFDB checksums match the previously frozen cohort. This is a new CPU float32 and Mitsuba LLVM replay of its seeds, layouts, geometry and corruptions. It is not byte parity with the original CUDA bank. Six first validation patients, global indices 1024–1029, supply 48 development views. Six last test patients, indices 1274–1279, supply 48 reserved views. Previously inspected test-gallery patients are excluded. There are six torch captures and two true 3D ray captures per patient. `source-records.json` preserves original global indices for independent regeneration. All 96 selected archive hashes were verified.

Gold dense material coordinates come from the renderer inverse field, independently of the inferred mesh. The optional export rotates and letterboxes those coordinates with RGB, support and nodes. Gold fields never enter the image-only detector or organizer. The three arms use gold coordinates with IDs, shuffled gold points without IDs, and frozen network detections.

Up to 4,096 visible material pixel centers per image form a fixed query domain inside the visible major-grid extent. Good coverage is the fraction with finite output and physical error <=0.5 mm. Wrong coverage is the fraction with finite output and error >0.5 mm. Rejections contribute zero good coverage. The 0.5 mm threshold is an engineering criterion, not a clinical acceptance rule. Evaluation permits only signed axis permutation and integer grid origin. It fits no scale, shear or affine map. Reported overall means weight patients equally and weight the two capture backends equally.

Five profiles were evaluated on validation only: original detector with shared cells, threshold 0.7, conditional banding at original threshold 0.85, banding at 0.7, and a confidence-anchored cycle organizer at 0.5. Selection maximizes good coverage subject to wrong coverage <= original-detector/shared-cell baseline +0.005 overall and +0.01 per backend. The higher-recall candidates failed this error constraint. `selection.json` was frozen before test inference, records hashes, and selected `banding`. The assessment runner refuses changed frozen sources or weights.

The filter estimates a periodic sinusoid only from observed background outside a pilot-point region, with minimum spectral concentration, fit quality and amplitude bounds. It is conditional and may abstain. It changes detector input only. Final RGB always samples the original image. The optional restoration head is diagnostic. No missing grid node is synthesized. The cycle organizer remains an explicitly experimental, unselected CLI profile. Simple ROI zoom, a second pass through the restoration head, NLM and affine/projective reindexing were also explored on a few validation views and did not justify adoption. Their code is excluded from the deliverable.

## Measured results

| Split | Capture | Good before | Good shared mesh only | Good selected | Wrong before | Wrong selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | backend-balanced | 67.49% | 67.50% | 69.15% | 3.14% | 1.14% |
| validation | torch-cached | 86.28% | 86.29% | 88.83% | 3.11% | 0.78% |
| validation | offline | 48.70% | 48.70% | 49.48% | 3.17% | 1.50% |
| test | backend-balanced | 63.99% | 64.00% | 65.33% | 2.44% | 2.43% |
| test | torch-cached | 84.46% | 84.46% | 87.13% | 3.45% | 3.43% |
| test | offline | 43.52% | 43.53% | 43.53% | 1.44% | 1.43% |

The reserved overall good-coverage gain is +1.34 percentage points, paired patient-bootstrap 95% interval approximately [+0.004, +3.989] points, conditional on six patients and this fixed checkpoint. Twenty thousand resamples use seed 1973. The practical gain is concentrated in one capture, `01276-5`, which changes from 4.05% to 99.07% good coverage. One image improves by more than one point and none regress by more than one point. This is not evidence of a broad detector improvement.

The reserved 3D result stays at roughly 43.5% good coverage. Wrong coverage is 2.44% before and 2.43% selected overall. On common finite support, pooled physical RMSE remains essentially unchanged at 2.364 mm. Four reserved images still have less than 10% good coverage. These failures remain in the denominator and in the complete gallery.

The reserved diagnostic arms give 98.85% good coverage with gold points and IDs, 96.43% with anonymous gold points, and 65.33% from the frozen network under equal backend weighting. The gap identifies detection as the main remaining bottleneck, with smaller organizer losses on 3D. `summary.json` includes each backend, paired intervals, common-support errors and failure counts. `validation.json` and `test.json` retain all image-level metrics.

## Use and reproduction

Install the project with digitization and pathtrace extras. The exact versions used here are in `environment.json`. Supply the recovered `joint-1973.pt` checkpoint with SHA-256 `8115e99c3517dc0b99974b9e64744aab9eb961fd0da06f1e84a65a1e45c3dbf8`. Its weights are unchanged.

```bash
python examples/rectify_denoising_grid.py \
  --checkpoint joint-1973.pt --image input.png \
  --profile banding --device cpu --pixels-per-mm 2 --out build/rectified

python examples/rectification_diagnostic.py \
  --sources research/rectification-attack-2026-10-22/source-records.json \
  --root build/rectification-replay
python examples/evaluate_rectification_diagnostic.py \
  --root build/rectification-replay --checkpoint joint-1973.pt \
  --out build/rectification-evaluation --split validation
python examples/select_rectification_profile.py \
  --validation build/rectification-evaluation/validation.json \
  --checkpoint joint-1973.pt --out build/rectification-evaluation/selection.json
python examples/evaluate_rectification_diagnostic.py \
  --root build/rectification-replay --checkpoint joint-1973.pt \
  --out build/rectification-evaluation --split test \
  --selection build/rectification-evaluation/selection.json
python examples/summarize_rectification_diagnostic.py \
  --root build/rectification-evaluation --out build/rectification-evaluation/summary.json
python examples/report_rectification_diagnostic.py \
  --bank build/rectification-replay --evaluation build/rectification-evaluation \
  --research research/rectification-attack-2026-10-22 --out build/rectification-report.html
```

The inference CLI and evaluator agree under permutation on detected coordinates, selected relative IDs and cells for two validation controls. Maximum coordinate difference is 3.1e-5 pixel. Score ties can permute point ordering. Previews are JPEG reductions, while metrics use native arrays and fixed physical queries. Gray preview pixels indicate missing output support.

## Verification and remaining evidence

The 26 focused geometry/inference tests pass. The full CPU integration run executes 392 tests: 378 pass, 12 skip and two error because original binary fixtures are absent in the reconstructed local checkout. Original fixtures remain untouched on the remote tree. They were not regenerated. The 16-scenario capture gallery was exported and visually inspected. Fatal-error lint and the reference workflow’s full digitization E/F lint pass. Checks and logs are included.

The earlier PR CPU jobs failed because `offline_noise.blur_motion` needs OpenCV but their installation included only the benchmark extra. The CPU matrix now explicitly installs `opencv-python-headless>=4.11,<5`, keeping the actual corruption test enabled. RGB mesh tests also have proper optional-dependency guards for installations outside that matrix. Fresh CI must confirm the complete checkout.

Chromium could not be downloaded for browser automation. Report QA uses HTML structure, decoded embedded previews and direct image inspection. No CUDA execution, real-photograph benchmark, waveform extraction or clinical validation was performed. The next substantive detector study should address small/oblique 3D papers and heavy sensor noise on validation patients, then use a fresh assessment set. These 48 reserved images are now inspected and must not be reused as untouched final evidence.
