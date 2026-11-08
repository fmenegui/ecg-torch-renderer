# npj Digital Medicine: methodological audit and authentic executable comparator

Date: 2026-10-23. Owner: independent npj reference agent. No M3 bank image was scored by this agent before the root protocol freeze.

## Exact reference and evidence depth

Elias Stenhede, Agnar Martin Bjørnstad and Arian Ranjbar, **Digitizing paper ECGs at scale: an open-source algorithm for clinical research**, npj Digital Medicine, published 14 January 2026, DOI [10.1038/s41746-025-02327-1](https://www.nature.com/articles/s41746-025-02327-1). Official full text was downloaded and read from the Europe PMC API at https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12891466/fullTextXML. Nature direct opening redirects to an inaccessible identity endpoint. PMC's HTML route presented a browser challenge, so the official public full-text API was used.

The paper evaluates complete digitization on 37,191 images, including a 266-patient Ahus cohort represented by 1,596 images and a 35,595-image Emory set. Its primary endpoint is shifted waveform SNR, with correlation and signal RMSE. These are not pure geometric error or full-page coverage measurements. Training uses CODE15-derived synthetic images rather than this project's renderer. The publication explicitly discusses curved-paper and real-photo limitations. These observations come from the full article, not its abstract alone.

Primary code: https://github.com/Ahus-AIM/Open-ECG-Digitizer, pinned commit `97a15087d4abcda843da8c58ee74b1d8f47e6f9a`. Public HTTPS clone succeeded. Both actual LFS model files downloaded, with exact LFS OID and byte-count verification. This is an executable reference, not a reimplementation inferred from a paper.

Web refs for integrator verification: `turn78view1` official repository, `turn93search0` official Nature entry, `turn93search4` official README. The final integrator should open its own primary sources before citing child refs.

## Main method and experimental branch must be separated

The released rectification stage consists of a learned four-channel residual U-Net, grid-based global perspective estimation, signal-informed quadrilateral selection and independent grid-scale estimation along the two axes. It does not infer a flexible sheet mesh in its paper-evaluated default configuration.

The README explicitly marks `dewarper.py` as experimental and not formally evaluated. It is disabled in all supplied inference presets. It locates crossings with a fourfold harmonic kernel, builds/filters a graph, optimizes neighboring-node geometry and fits a TPS using a bounded random subset of control points. This branch can fall back to identity. Running that branch and calling it the validated npj method would be inaccurate. Our main comparator therefore keeps it off. A future TPS experiment needs a separate method label, seed, point budget and failure accounting.

## Native geometry stage, read from actual source

| Stage | Verified operation | Consequence for choosing our method |
|---|---|---|
| RGB preparation | Global min-max normalization. Images with maximum dimension above 3000 are downscaled, while minimum dimension below 512 is upscaled. The number 3000 is a maximum, not a forced target resolution. | Compare actual input resolution and photometric preprocessing. Avoid assuming a 3000-square inference cost. |
| Evidence | Residual U-Net with eight widths 32, 64, 128, 256, 320, 320, 320, 320. Four semantic classes are grid, text/background, trace and background. Softmax maps are mean-subtracted, clipped and normalized. | Line evidence and trace evidence provide complementary cues. Point-only inference should be tested against this richer segmentation evidence rather than compared only with itself. |
| Perspective | Grid pixels above an estimated 98th percentile enter a Hough transform. The algorithm searches a coarse angular interval, then a refined interval. Variance along lines in Hough space identifies two line families. Quantile sampling and large-grid sampling can use RNG. | A strong global stage can suppress local grid-point noise on planar documents. It cannot model a true fold or arbitrary local surface deformation. Fix RNG for reproducible comparison. |
| Extent selection | A trace-probability-informed quadrilateral uses binned probabilities and specified percentiles. Source corners define a global perspective warp. | Cropping is content-informed, which can preserve traces better than paper-edge guessing, but it is not a guarantee of preserving headers, annotations or every valid paper pixel. |
| Sampling and orientation | The cropper calls `torchvision.transforms.functional.perspective` with its actual **BILINEAR** default. A portrait output rotates clockwise by 90 degrees. A subsequent row crop uses signal-plus-grid projections. | Sampling conventions and all crop/rotation operations must be composed. The cropper does not merely return a corner homography that can be used without half-pixel bookkeeping. |
| Calibration | Autocorrelation of summed grid probabilities is matched against a major/minor template with a 5:1 spacing ratio. A zooming search estimates x and y mm per pixel separately. | Explicit physical calibration is valuable. Periodic aliases and expected grid-count bounds remain assumptions. Do not rescue a wrong inferred scale with gold regression. |
| Downstream digitization | Connected-component trace assembly and layout/template recognition happen after these geometric steps. | Geometric operation is not intrinsically restricted to the downstream lead-layout list. Conversely, success of a complete waveform pipeline cannot be attributed exclusively to rectification. |

The main native map is a projective source-to-output map followed by two scale factors. Its native physical origin is output pixel center `(0,0)`, an arbitrary phase relative to printed grid crossings. It exposes no inferred lattice identities. A translation-only evaluation gauge is therefore appropriate for comparing shape and units, provided every comparator gets the same privilege and the alignment anchors are disjoint from scored queries. Gold scale, shear or free rotation must not be fitted in the primary score.

## There is no single unspecified default preset

All values below were read from the pinned official YAML files. They must be named, rather than silently selecting the best preset for each image.

| Official preset | Crop alpha | Minor-grid template weight | Other main geometry parameters |
|---|---:|---:|---|
| `inference_wrapper.yml` / generic | 0.85 | 0.3 | 250 angular samples, 80 crop bins, 0.02/0.98 trace percentiles, 30 to 70 expected major lines, max dimension 3000 |
| `inference_wrapper_george-moody-2024.yml` | 0.85 | 0.5 | Same geometric search. Downstream layout and trace requirements differ but are not used in a rectification-only comparison. |
| `inference_wrapper_ahus_testset.yml` | 0.90 via Cropper default | 0.3 | Same geometric search. Ahus-specific downstream layout configuration is omitted from our pure geometry arm. |

Root's planned primary arm is Generic, with George Moody 2024 as a separate predeclared arm. Ahus is used for the analytical smoke only unless the root protocol explicitly includes it. No per-case oracle selector is supplied.

## Decisions for our rectifier, with falsification tests

1. **Retain a strong global projective hypothesis.** Our local refinement should not be assumed superior on flat paper. Test a common-evidence projective fit and this authentic npj stage on flat/scanned versus folded captures separately. Reject global-only as the universal model if local residual physical errors remain after correct units and origin alignment.
2. **Separate grid identity from localization.** npj's period search and our lattice IDs are different ways of imposing calibration. Our worst physical errors cannot be fixed by smoothing locations while preserving a wrong identity. Test factor-five scale aliases and false neighbor links explicitly, with no gold scale correction.
3. **Add independent extent and retention reporting.** npj can use trace evidence to choose extent. Our full-frame policy preserves more information but extrapolates farther. Return both whole-paper and trace-region coverage with physical error, and distinguish an original-preserved fallback from successful rectification. Neither more pixels nor a smaller cropped error alone proves superiority.
4. **Use geometry first to test geometry.** The paper's waveform scores allow up to 100 ms horizontal alignment and zero-centering. The released evaluation code ignores NaN pairs and separately reports their fraction. These choices can be sensible for signal reconstruction but conceal some geometric offset and missing-coverage consequences. A rectification benchmark needs fixed physical queries and all failures in the denominator. A complete signal comparison should additionally retain unshifted and shifted metrics with missingness explicitly reported.
5. **Preserve the original and compose sampling once when extending our system.** The native reference samples its normalized image, not our RGB-preservation policy. Do not silently add our full-page sampler or change its crop and then label the result original npj. A common-sampler stage-matched ablation can be useful, but it is a separate arm.
6. **Do not equate flexible geometry with better geometry.** More degrees of freedom may help folds and amplify false local correspondences. A predeclared flat/folded split, worst-tail physical error and wrong coverage can decide between hypotheses. Internal round-trip or positive local Jacobians alone do not answer that question.

## Runnable adapter and authenticity checks

`npj_adapter.py` imports the original UNet, PerspectiveDetector, Cropper and PixelSizeFinder modules from the pinned checkout without altering their files. It instruments the wrapper's geometric control flow, then stops before signal extraction and layout recognition. The pipeline instrumentation is marked CC-BY-SA-4.0 with upstream attribution. Original files and checkpoint are strictly SHA256 checked. Official preset YAML bytes are strictly checked. CPU device is explicit, with two threads used in the smoke.

API:

```python
from npj_adapter import NPJPerspectiveRectifier
rectifier = NPJPerspectiveRectifier(upstream_root='/path/to/Open-ECG-Digitizer', preset='generic')
mapping, normalized_rectified_rgb, metadata = rectifier.infer(image_uint8)
predicted_mm, eligible = mapping.predict(source_queries_xy)
```

The map exports `source_to_pixels`, `source_to_mm`, `mm_per_pixel`, `source_shape`, and `output_shape`. Eligibility requires finite coordinates inside both source pixel-center bounds and the actual cropped output pixel-center bounds. There is no extension outside the native output and no gold input. The adapter composes the original half-pixel resize convention, torchvision corner-coordinate perspective transform, possible clockwise rotation and actual row crop.

Three independent controls are in `test_adapter_controls.py`:

* Physical-unit and crop-bound checks using an analytical anisotropic map.
* Actual torchvision bilinear sampling of x/y coordinate ramps agrees with the exported inverse map to less than `1e-4` pixel on the interior.
* A portrait analytical image produces exactly equal corners, RGB output and learned x/y scale under the adapter and the actual upstream wrapper geometry methods. Only imports of unrelated training/config dependencies are removed from the dynamically executed wrapper AST. No function body is changed. This is geometric-stage parity, not execution of the full waveform/layout pipeline.

## Declared analytical smoke

`smoke.py` creates one 768×512 analytical chart with known 4 pixels per millimeter and several artificial sine/pulse strokes. It is not a clinical ECG and not a frozen-bank image. The purpose is model load, stage reachability and physical-unit sanity. It is not used to select a preset or tune thresholds. The exact input and output, metadata, timings and a translation-invariant diagnostic vector error are retained locally in the source cache. Results and source hashes are exported in `smoke-result.json` and `availability.json`.

No conclusion about accuracy on our bank can be drawn from that single chart. The parent benchmark is responsible for frozen queries, disjoint anchors, inclusion of failed stages, both named presets and paired patient-level reporting.

## Availability, licenses and reproduction boundary

The official main UNet weight is 90,464,067 bytes, SHA256 `17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b`. The 23,296,757-byte lead-name UNet is also publicly obtainable and hash-verified, but is not executed by this rectification adapter. Repository license is CC-BY-SA-4.0. Article license is CC-BY-4.0. Mandatory article citation is stated by upstream README. Do not redistribute the cached complete article or source repository as our original work. Preserve source identity and attribution if publishing the adapter.

The adapter uses Torch 2.8.0+cpu, torchvision 0.23.0+cpu, NumPy 2.3.5 and PyYAML. A separate environment at `m3/npj/venv` was created to avoid changing shared dependencies. Root may instead use its matched shared environment. CPU measurements are not GPU measurements.

## Primary source URLs

* Article: https://www.nature.com/articles/s41746-025-02327-1
* Official full text: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12891466/fullTextXML
* Repository: https://github.com/Ahus-AIM/Open-ECG-Digitizer
* Wrapper: https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/97a15087d4abcda843da8c58ee74b1d8f47e6f9a/src/model/inference_wrapper.py
* Perspective detector: https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/97a15087d4abcda843da8c58ee74b1d8f47e6f9a/src/model/perspective_detector.py
* Cropper: https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/97a15087d4abcda843da8c58ee74b1d8f47e6f9a/src/model/cropper.py
* Calibration: https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/97a15087d4abcda843da8c58ee74b1d8f47e6f9a/src/model/pixel_size_finder.py
* Evaluation: https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/97a15087d4abcda843da8c58ee74b1d8f47e6f9a/src/evaluate.py
