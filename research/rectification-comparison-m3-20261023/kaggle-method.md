# Kaggle rectification: source-grounded methodological review

Date: 2026-10-23. Scope: actual Heng stage 0/1 pipeline as reused by the third-place solution, plus first-place methodological changes. No claim that one implementation represents every Kaggle team.

## What was recovered

The widely shared base is **Heng CherKeng, `hengck23`**. The third-place solution is by **Tang**, repository `tanghaozhe/physionet-ecg-image-digitization-3rd-place`. Third place is not Heng's placement. Both first- and third-place write-ups explicitly credit Heng's rotation/homography and grid detector. This distinction matters when comparing an underlying rectifier with a team's full digitization score.

The official model dataset is publicly downloadable without credentials. Actual stage 0 and stage 1 checkpoints were downloaded and loaded strictly, with zero missing and unexpected keys. Stage 0 has 14,326,454 parameters and stage 1 has 24,414,710. The dataset is version 3, updated 2025-11-14. We did not retrain either network.

Pins:

* Heng training repository: `95cfc6fecee9f2ccbd20f1c9acfd3706728303fd`, MIT.
* Third-place repository: `f2b4ac2823007e3fb387a7e188e748ad832bb2b7`.
* Actual third-place single-image implementation SHA256: `130768d580e130706774bbc6fd97c337cf118a3d962af6e67beb0332d9303b3d`.
* Stage 0 checkpoint SHA256: `b7360dfce9c49305b54272cf242e93e5728823f439a8e5ad59c03b073988cfc9`.
* Stage 1 checkpoint SHA256: `313f1e5dafdc212977e4586ce118453866830be0fa32664e36054f7f4568a46a`.
* Canonical grid NPY SHA256: `e695f580bd0de4df2fe2e2ba8c89809af819127b2e5936c2716830f421cef4ad`.

The third-place repository has no root LICENSE file. Its preparation script attributes the original notebook to Apache 2.0, while the separately released Heng training repository is MIT. The model dataset's API reports license `Unknown`. These are separate provenance facts, not interchangeable licenses. The study uses external checkouts and downloaded assets locally, and should publish its own wrapper, retrieval URLs and hashes without repackaging the author's weights or unlicensed adaptation.

## Native pipeline and its assumptions

### 1. Orientation and coarse normalization

The third-place preparation script defines a ResNet18d encoder and U-Net-style decoder for 14 lead-marker classes, together with an eight-class orientation head. Input is resized to width 1440, padded to a multiple of 32, and evaluated under four flip augmentations. Predictions are transformed back and averaged. Marker components yield nine selected lead-name positions: aVR, V1, V4, aVL, V2, V5, aVF, V3 and V6.

The largest component for each selected class is used. A missing component defaults to coordinate zero. Only quarter-turn classes have explicit rotation decoding. Classes 4 through 7 trigger an unknown-orientation message and proceed through a quarter rotation rather than a reflection. The nine positions are matched to a fixed canonical arrangement and fitted by `cv2.findHomography(..., RANSAC)`. The coarse output is 1440 by 1152.

This is strong semantic orientation evidence when the page follows the learned arrangement. It is also a format dependency. The same nine spatial correspondences do not automatically describe a 6-by-2, 12-by-1, Cabrera or arbitrary rhythm-strip layout. A native homography failure must stay a failure, rather than be rescued with gold corners or our estimated geometry under the same comparator name.

An important general conclusion is independent of these network details: a square grid alone has rotational and reflection symmetries. Recovering its lattice does not determine the clinical reading direction. Orientation needs text, lead labels, acquisition metadata or another explicit cue. The old M2 axis/permutation evaluation gauge does not measure that task.

### 2. Absolute grid identification

Stage 1 uses a ResNet34 encoder and U-Net-style decoder. It predicts a crossing heatmap plus horizontal and vertical line classes tied to a fixed 44-by-57 lattice. Components of heatmap pixels above 0.5 are reduced to centroids and rounded to integer coordinates. Line-segmentation components are assigned their majority class. Additional line-fragment checks compare fitted line positions and merge or reject fragments. A crossing gets its row and column from the corresponding predicted line classes.

These semantic row and column heads can provide useful correspondence information that an unlabeled point detector lacks. They also impose an absolute template. Their learned identity prior must not be treated as an observation that remains reliable under every paper size, placement or layout.

**The third-place write-up's parabolic subpixel refinement concerns waveform peak extraction, not grid-localization decoding.** The actual grid decoder still rounds component centroids. First place separately describes probability-weighted grid-position estimates. These should not be conflated.

### 3. Filling gaps and warping

Missing lattice nodes are filled using cubic `scipy.interpolate.griddata`. Remaining NaN coordinates are replaced by source coordinate `[0,0]`. The result is a rectangular 44-by-57 backward map, upsampled bilinearly into a dense sampling map. It is not a triangle mesh and must not be silently replaced by piecewise affine interpolation in the evaluator.

The third-place high-resolution path scales the coarse homography and detected grid to the raw-image scale, performs a high-resolution `warpPerspective`, samples through the grid and finally resizes to 4400 by 1700. This reduces the damage of rectifying exclusively at low resolution, but it still contains successive interpolation operations. First place describes a direct-remapping alternative from the rotated source image, avoiding an intermediate homography raster.

The physical interpretation used for comparison is nominal 5 mm grid pitch, with a 280 by 215 mm lattice span. The final 4400-by-1700 raster has different pixel densities in x and y. That is not inherently a physical error if calibration is tracked separately on both axes. It is a methodological error to assume one isotropic pixels/mm value after such a resize.

### 4. Sampling convention and failure behavior

The actual source normalizes source pixel coordinates by `(W-1,H-1)` and uses `grid_sample(..., align_corners=False, padding_mode='border')`. These two conventions do not represent the same pixel centers. The wrapper deliberately retains this behavior for the authentic comparator. A corrected sampling-convention arm would be a separately named ablation.

The native source has no global physical-error certificate and no calibrated uncertainty. A rectangular output can still contain hallucinated or folded correspondences. Border padding duplicates edge pixels. Coordinates filled to `[0,0]` copy the source corner into unsupported output regions. Such pixels must not be advertised as recovered paper. This does not imply that every real input exhibits the problem, and it is not a measurement of the competition's overall performance.

## Concrete component controls executed

`analytic-controls.json` and `analytic_controls.py` retain the values and generation procedure. No capture bank or gold target was opened for these analytical controls.

1. Native bilinear-map inversion was checked on 6,000 continuous queries across an identity map, a smooth bilinear map, and composition with an independent projective transform. The largest inverse-coordinate error was about `1.42e-13` mm. This checks evaluator conformance, not real-world rectification accuracy.
2. Source-to-rotated pixel coordinates match `np.rot90` exactly for all four quarter rotations.
3. Independent QA caught artificial exclusion of source boundary pixel centers in the initial wrapper. The wrapper now uses inclusive bounds, with an explicit four-query edge control. This was fixed before any frozen assessment was scored.
4. An actual call to the author's sampler on an identity coordinate-ramp image produced maximum interior error `0.495819` pixel and 2D RMSE `0.406322` pixel before the final uint8 conversion. The controlled `align_corners=True` ablation reduces maximum error to `3.05e-5` pixel. This confirms a coordinate-convention defect in the component. It does not quantify its downstream ECG impact.
5. Removing outer grid observations from a regular analytical grid left 1,776 observed nodes. The native cubic-fill rule produced 732 zero-valued coordinates, or 29.19% of the complete 2,508-node array. This specifically exposes extrapolation outside the observed convex hull.
6. A folded analytic backward map has two conflicting inverse positions for a source location. The evaluator reports ambiguity instead of returning a favorable arbitrary preimage. This is evaluator geometry eligibility, not native model confidence.

## Comparison decisions to retain or challenge

| Decision | Evidence and tradeoff | Recommended choice and falsification |
|---|---|---|
| Estimate geometry economically, sample high-resolution RGB | First and third place both separate parameter estimation from high-resolution use. This is prior art, not a novelty claim. | Retain original-RGB sampling. Compare one-shot composition and the authentic interpolation cascade on identical traces and maps. |
| Treat source pixel conventions as part of the algorithm | Actual identity ramp fails by about half a pixel at some positions. | Require coordinate-ramp, border and odd-size controls. Keep original comparator and corrected ablation separately named. |
| Use semantic line identities | Absolute heads can resolve local correspondence ambiguity, while fixed template labels can fail under distribution shift. | Investigate relative/periodic line evidence plus explicit scale hypotheses. Compare under a matched-input decoder lane, apart from full native networks. |
| Use semantic orientation evidence | Lead labels provide information absent from a bare square lattice. Nine fixed spatial anchors limit layout scope. | Add layout-aware orientation evidence, with explicit uncertainty. Evaluate automatic orientation separately from a D4 diagnostic. |
| Fill missing regions | Interpolation can improve continuity within observed support. Zero-fill outside support produces a real source coordinate without evidence. | Separate evidence, geometric eligibility and RGB. Never interpret a sentinel as a measurement. Test holes, boundaries and disconnected components. |
| Smooth grid observations | First place describes weighted positions and surface-based rejection, while our M2 guard shows folds can conflict with smoothing. | Preserve physically plausible discontinuities, record abstention and expose errors in both added and retained support. Do not select by smooth appearance alone. |
| Fix output aspect ratio | Third place's 4400-by-1700 output is deliberately anisotropic. | Track x and y calibration explicitly. Isotropic comparison previews should be separate from native outputs. |
| Optimize a full digitization score | Signal prediction, TTA, ensembles, lead fusion and postprocessing all affect SNR. | Use geometric error, failure/coverage and trace preservation as direct rectification metrics. Measure the signal stage separately with one common downstream extractor. |
| Infer generality from one successful template | Native models were trained for a specific challenge distribution. Our models were trained with our renderer family. | Split native-compatible and format-stress lanes. Keep failures in denominators. A win on our synthetic bank is not broad superiority over Kaggle. |

First-place local examples report differences for high-resolution remapping and Fourier resampling from only one image. Those demonstrations motivate controlled ablations, but are not population estimates. Its validation description uses ten training sample IDs and 30 validation images. It does not establish the patient/template/camera separation needed for a broad rectification claim. Heng's training README describes use of downstream regression errors and visual inspection rather than a dedicated quantitative grid-validation set. Our patient-separated synthetic evaluation addresses one dimension of separation, while still leaving shared-renderer and clinical-photo generalization unresolved.

## Executable comparator contract

`KaggleThirdPlaceAdapter(source_repo, assets).infer(image_rgb_uint8, output_dir)` executes the author's actual stage definitions, decoding functions and high-resolution RGB functions. Adaptations are limited to CPU float32 instead of CUDA float16, supplying the missing canonical reference asset, disabling ImageNet preload, seeding the original stochastic line subsampling with 1973 and using a single-image driver. Checkpoints are loaded strictly. Source, checkpoint and reference hashes are verified.

The field returned by `infer` exposes `.forward(source_queries_xy)` returning physical coordinates in mm and a boolean geometry-eligibility array. It inverts the native bilinear grid in normalized high-resolution coordinates, retaining the author's pixel convention, then composes the global homography and quarter rotation exactly. It does not perform a new fit to gold. Nonconvex and ambiguous inverse geometry are reported explicitly. The native RGB output still uses the author's code, independent of this evaluator map.

The map represents the continuous native geometry, not every OpenCV interpolation-table quantization or final cubic image-filter footprint. The nominal grid pitch assumption is explicit. Coordinates outside the actual canonical crop must not be expanded into full-frame eligibility. This comparator must not be called a first-place reproduction, nor a corrected generic 'Kaggle algorithm'.

One development-only wrapper smoke used image `01024-0` without reading its targets. It failed natively in stage 0 because the estimated homography was invalid, with 26.63 seconds on two CPU threads. This is not a frozen-test result. No rescue or parameter tuning followed. An author-supplied annotated reference overlay produced a fully missing grid and is not a valid positive fidelity control. A real normalized example from the author's own public training-data dataset was subsequently used for a positive Stage 1 execution check. It produced all 2,508 observed finite grid nodes, a heatmap maximum of 1.0 and 32,932 pixels over the unchanged 0.5 threshold in 12.76 seconds on two CPU threads. This establishes positive execution fidelity on a native-distribution input, not benchmark accuracy, and no gold target was read. The full wrapper then completed on this same example with rotation 0, a finite homography, all 2,508 detected grid nodes, all 2,408 convex cells and a native 4400-by-1700 RGB output in 49.41 seconds. That input was already stage-0 normalized, so this is an execution sanity check rather than a test of a new raw photograph. RGB handling was verified against both author training datasets and the third-place preparation script.

## Primary sources and citation handles

* Heng author repository and README: https://github.com/hengck23/kaggle-physionet-ecg-image-digitization (`turn82view0`).
* Third-place author repository and README: https://github.com/tanghaozhe/physionet-ecg-image-digitization-3rd-place (`turn82view1`).
* Third-place original single-image implementation, pinned: https://github.com/tanghaozhe/physionet-ecg-image-digitization-3rd-place/blob/f2b4ac2823007e3fb387a7e188e748ad832bb2b7/data_preparation/prepare_inputs.py . Actual cloned file read in full by relevant sections.
* Third-place geometry helper: https://github.com/tanghaozhe/physionet-ecg-image-digitization-3rd-place/blob/f2b4ac2823007e3fb387a7e188e748ad832bb2b7/utils/geometry.py .
* Third-place grid helper: https://github.com/tanghaozhe/physionet-ecg-image-digitization-3rd-place/blob/f2b4ac2823007e3fb387a7e188e748ad832bb2b7/utils/grid_utils.py .
* Third-place official write-up: https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/3rd-place-solution (`turn79search8`, `turn82view2`). The official cached JSON is an internal source receipt, not for wholesale redistribution.
* First-place official write-up: https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/1st-place-solution (`turn104search0`, `turn82view3`, cached full official JSON receipt in the earlier source audit).
* Official checkpoint dataset: https://www.kaggle.com/datasets/hengck23/hengck23-demo-submit-physionet (`turn86view0`). Its public API file list, metadata and downloaded file hashes are in `availability.json`.

The root integrator must open web refs itself before using them in final-answer citations. Do not quote or republish the full Kaggle write-ups. Link to originals and retain only our analytical summary.
