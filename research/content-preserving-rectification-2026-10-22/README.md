# Preserve visible ECG content when grid evidence is incomplete

The default RGB output previously treated incomplete grid-cell support as absent image content. It both whitened holes without four detected corners and limited the output canvas to the detected grid extent. A trace, header or calibration mark could therefore disappear even though it was present in the input. This is an output-contract defect, independent of detector accuracy.

The default now archives the exact decoded input RGB and proposes a full-frame warp. Existing observed-cell triangles keep their exact local transforms. Only their complement inside the entire input rectangle is triangulated. Forward coordinates and inverse RGB sampling share this completed mesh. Grid evidence and coordinate eligibility are separate images and never act as RGB cutout masks. If a consistent completion cannot be constructed, or its index or output canvas exceeds resource bounds, the original RGB is returned unchanged. Existing partial observed-cell coordinates remain available separately.

This fixes visible-content deletion. It does not establish that inferred geometry is correct. The full inferred field has substantial errors on difficult 3D captures. The output is explicitly labeled `rectification-proposal`, with `geometry_accuracy_verified=false`. Keep the original and use the separate conservative coordinate-eligibility field for measurement.

## Primary-source review

| Source | Geometry actually described or implemented | Consequence for this change |
| --- | --- | --- |
| [Stenhede et al., npj Digital Medicine, DOI 10.1038/s41746-025-02327-1](https://www.nature.com/articles/s41746-025-02327-1) | Signal/grid/text/background segmentation, two-stage Hough perspective correction and signal-informed crop | Preserve visible signal when determining the image boundary. A missing geometric observation does not justify deleting RGB. |
| [Ahus-AIM/Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/97a15087d4abcda843da8c58ee74b1d8f47e6f9a/src/model/dewarper.py) | Repository's experimental nonlinear dewarper uses grid crossings, graph neighbors, square-edge optimization and TPS with identity fallback | This component is distinct from the main paper's validated global perspective pipeline. Its fallback is reported separately. |
| [US20240144417A1](https://patents.google.com/patent/US20240144417A1/en), also EP4302256B1 / WO2022184740A1 | Dotter grid intersections, Gridder edge/cell organization, inference of missing vertices from surrounding grid points, and original-image sampling per unit cell | Fill missing geometry under explicit constraints, rather than treating missing vertices as missing visible content. |
| [Demolder et al., High precision ECG digitization using artificial intelligence](https://pubmed.ncbi.nlm.nih.gov/40010101/) and [primary preprint methods](https://www.medrxiv.org/content/10.1101/2024.08.31.24312876v1.full) | Dotter normalization followed by Gridder gap interpolation and four-corner correction per grid square | The technical description supports geometric interpolation. It does not provide an executable proprietary comparator here. |
| [US20260065633A1](https://patents.google.com/patent/US20260065633A1/en) | Per-column waveform candidates, lead baselines and continuity/path association | This is waveform extraction. It must not be mistaken for an image-rectification method or permission to reconstruct unseen RGB. |
| [Tang, Kaggle third-place source](https://github.com/tanghaozhe/physionet-ecg-image-digitization-3rd-place/tree/f2b4ac2823007e3fb387a7e188e748ad832bb2b7) | Stage 0 landmark homography, stage 1 indexed grid points, cubic interpolation of missing grid nodes and dense sampling | Interpolation is useful, but fixed grid/template assumptions and out-of-domain behavior need independent checks. |
| [Dang, Kaggle fifth-place source](https://github.com/dangnh0611/kaggle_ecg_digitization/tree/8358d5b09a5b9519275e9b54c11716da5752a9d6) | High-resolution indexed keypoint heatmaps, iterative Hungarian assignment, per-lead piecewise geometry and spline/linear/nearest interpolation | Dense evidence and correspondence recovery matter. Fixed reference geometry does not establish general layout robustness. |
| [Someya, official second-place write-up](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/2nd-place-solution) | Heng stage 0/1 geometry retained, with improvements mainly in waveform extraction and lead sharing | Its final waveform score is not an isolated rectification score. Cross-lead signal completion is not used to fill our RGB. |

The full npj article XML and methods/results/discussion were read through the official Europe PMC full-text endpoint, PMC12891466. The geometry and waveform patent methods/claims and the linked geometry source files were reviewed. The Demolder preprint methods and Someya write-up were available through indexed primary-source text. Their full original pages were not downloaded successfully. No full articles or upstream source copies are redistributed here. The name “Lauder” could not be identified unambiguously as a relevant rectification source. It is not silently attributed to an inventor or method.

## Geometry and output contract

`frame_mesh.py` constructs the union of observed cells and triangulates the polygonal gaps, including the outer frame. Constrained Delaunay triangulation preserves the boundary edges that ordinary Delaunay may replace. Candidate source and target triangulations are checked for orientation and complete area coverage. A robust finite projective fit supplies only outer-frame correspondences. A pole crossing the image, overlapping observed cells, incompatible gap completion or unsafe index causes abstention.

`complete_rectifier.py` uses one mesh in both directions. The evidence legend distinguishes observed cells, interpolation inside the detected hull, and extrapolation outside it. These labels describe provenance, not error guarantees. Coordinate eligibility retains observed-cell coordinates and permits limited interpolation near consistent detected vertices inside their hull. It excludes unrestricted extrapolation. It is a measured heuristic, not a calibrated probability or accuracy certificate.

`original-rgb.png` always preserves decoded input values. `output-original-rgb.png` contains a sampled full-frame proposal or the unchanged input. `geometry-evidence.png`, `geometry-eligibility.png` and `sampling-valid.png` are separate. White padding may occur outside the mapped input quadrilateral. Missing grid evidence does not whiten internal RGB. `--output-mode cells` explicitly requests the old partial-cell diagnostic.

`observed-grid-geometry.npz` contains selected source coordinates, corresponding physical coordinates, grid IDs, scores and allowed cells. The output helper writes this archive itself, so partial-coordinate fallback is reproducible for both checkpoint CLIs and standalone cached-point previews. Raw model-specific `points.npz` files are retained separately. A regression check reconstructs the fallback coordinate mask from the reported archive and compares it with the emitted eligibility image.

The default detector profile is the previously validation-selected `banding` profile. Weights, architecture and cached detections are unchanged. Background filtering influences detector input only. Original RGB supplies the output, and the optional restoration head remains diagnostic.

## Evaluation protocol and selection

The same 96 native CPU/LLVM replay captures and fixed physical query pixels from the preceding diagnostic are reused. There are six validation patients and six different retrospective test patients, with eight views each. Each patient has six torch captures and two true 3D captures. Gold material coordinates are used only for scoring, never fitting. Evaluation allows signed axis permutations and an integer grid-origin shift. It fits no scale, shear or affine transformation. The fixed query domain is the visible material inside the grid extent, not the entire photographic background.

Good coverage is the fraction of all fixed queries with finite coordinates and error <=0.5 mm. Wrong coverage is the fraction with error >0.5 mm. Unsupported queries remain in the denominator. Means weight patients equally and the two capture backends equally. The 0.5 mm threshold and error budgets are engineering choices, not clinical acceptance criteria.

Five eligibility radii were compared on validation only: 0.25, 0.5, 0.75, 1.0 and 1.25 local spacings. Selection maximized good coverage subject to wrong coverage no more than baseline +0.005 overall and +0.01 in either backend. Radius 0.75 was selected. Sources, validation results, input manifest, cached topology digest and unchanged checkpoint identity were frozen before the retrospective assessment. CLI and evaluator use this radius. The previously inspected test set is explicitly not an untouched independent test.

When a full map is rejected, RGB remains unrectified and its prior observed-cell coordinate field is retained separately. The coordinate aggregate measures that field, not a claim that the unrectified image has a globally uniform physical scale. Full-map proposal counts precede the renderer's density-dependent output-size guard. Gallery sampling uses 2 px/mm. At the CLI's default 8 px/mm, four validation proposals additionally exceed the output bound and keep original RGB.

## Fixed-query results

| Split and capture | Good before | Good eligible now | Wrong before | Wrong eligible now |
| --- | ---: | ---: | ---: | ---: |
| validation, backend balanced | 69.15% | 72.21% | 1.14% | 1.57% |
| validation, torch | 88.83% | 90.20% | 0.78% | 0.84% |
| validation, true 3D | 49.48% | 54.21% | 1.50% | 2.29% |
| retrospective test, backend balanced | 65.33% | 68.86% | 2.43% | 2.94% |
| retrospective test, torch | 87.13% | 89.51% | 3.43% | 3.66% |
| retrospective test, true 3D | 43.53% | 48.22% | 1.43% | 2.23% |

The retrospective gain is +3.53 percentage points in good coverage. Wrong coverage also increases by +0.51 points. Its increase is slightly above the +0.50-point overall development budget, so that constraint did not transfer exactly. Twenty-one of 48 images gain over one point in correct coverage and none lose over one point. The paired patient bootstrap interval for the good-coverage gain is approximately [+1.10, +6.98] points, conditional on just six patients and this checkpoint. It is not a population generalization guarantee.

Twenty-seven validation and 32 retrospective test images admit a complete mesh. The other 21 and 16 retain original RGB and separate partial coordinates. The maximum measured shared-mesh round-trip error on finite fixed queries is 1.05e-11 pixels. Numerical self-consistency does not establish physical correctness.

**Do not use the entire inferred field as trusted geometry.** On the retrospective test its unrestricted aggregate good coverage is 71.43%, but wrong coverage is 15.62%. For true 3D, unrestricted wrong coverage is 26.30%. Eligibility reduces this at the expense of leaving many physical queries unsupported. Full visible RGB and correct physical coordinates are different outcomes. These results do not establish superiority over npj, Dotter or Kaggle.

Eight older matched JPEG inputs are also replayed using the same cached detections and cached npj adapter outputs. They are visual diagnostics from one patient without shared geometric gold. The upstream networks and references were not rerun here. The npj RGB was sampled by our adapter through its estimated field, not an official paper RGB artifact. Its two identity fallbacks are distinguished. The cached Kaggle stage-0 adapter failed on all eight inputs. Those failures are retained, not evidence that a leaderboard solution is generally worse.

## Reproduction and verification

Install `.[digitization]`. Supply the unchanged selected checkpoint recorded in `selection.json`, or reuse the matching cached native topologies and queries from the preceding diagnostic.

```bash
python examples/rectify_denoising_grid.py --checkpoint selected.pt \
  --image input.png --profile banding --output-mode full --out build/full-frame

python examples/evaluate_complete_rectification.py --bank build/diagnostic \
  --previous build/diagnostic-evaluation --out build/complete --split validation
python examples/summarize_complete_rectification.py --root build/complete \
  --bank build/diagnostic --select
python examples/evaluate_complete_rectification.py --bank build/diagnostic \
  --previous build/diagnostic-evaluation --out build/complete --split test \
  --selection build/complete/selection.json
python examples/summarize_complete_rectification.py --root build/complete
```

Sixteen new tests cover missing-grid traces and outside-grid headers, original preservation, independent smooth physical deformations, exact retention of observed-cell transforms, full-frame invertibility, inconsistent IDs, poles, output bounds, index bounds and partial-coordinate fallback. Together with prior focused geometry tests, 27 pass. Fatal-error lint and strict E/F digitization lint pass. See the included logs for the full integration run and original missing binary-fixture limitations. The 16-scenario capture gallery was exported and visually inspected. No renderer hot path or physiological source was changed.

The completed local integration run executed 408 tests: 393 passed, 12 skipped, two errored because the reconstructed checkout lacks the original binary regression fixtures, and an existing Mitsuba exact-RGB parity assertion failed. That unchanged test passed immediately in an isolated repeat. Two subsequent combined runs stopped before producing a complete summary, so they are not reported as completed passes. Their logs are retained as incomplete diagnostics. Fresh complete-checkout CI is the integration gate. Original regression assets are preserved unchanged, not regenerated.

An inherited CLI metadata flag incorrectly claimed that the new sampler performed an explicit per-pixel return check. A metadata-only amendment now sets that flag to false. It changes no geometry, RGB sampling, eligibility, selected radius or physical metrics. `selection.json` records the amended CLI hash. All 48 retrospective fit and result records were repeated and are exactly equal before and after this correction.

This turn does not train a model, run CUDA, execute fresh pretrained competitors, or validate on real photographs or extracted waveforms. The remaining improvement with the largest evidence base is native-resolution or tiled multiscale grid detection, dense grid-line evidence and correspondence confidence. It needs fresh held-out patients, matched public reference runs with the same successful/failed denominators, physical-error metrics and waveform distortion checks before any claim of being better than all competitors.
